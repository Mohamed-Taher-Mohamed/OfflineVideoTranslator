"""faster-whisper (CTranslate2) backend. Loads ONLY from a local folder - never from the hub."""
import math
from typing import List, Optional, Tuple

from .audio_io import read_wav_float32
from .base import ASRBackend, Segment


class FasterWhisperBackend(ASRBackend):
    name = "faster-whisper"

    def __init__(self, model_dir: str, device: str = "auto", compute_type: str = "auto", cpu_threads: int = 0,
                 beam_size: int = 5, sensitivity: str = "normal", multilingual: bool = False,
                 word_timestamps: bool = False, vad_min_silence_ms: int = 2000):
        self.beam_size = max(1, int(beam_size))
        self.sensitivity, self.multilingual = sensitivity, multilingual
        self.word_timestamps, self.vad_min_silence_ms = word_timestamps, int(vad_min_silence_ms)
        from faster_whisper import WhisperModel  # lazy: keeps import cost out of the CLI --help path
        if compute_type == "auto":
            compute_type = "int8" if device in ("auto", "cpu") else "float16"
        # model_dir is a directory path => faster-whisper does not contact the hub.
        from ..errors import ModelIntegrityError
        try:
            self._model = WhisperModel(str(model_dir), device=device, compute_type=compute_type,
                                       cpu_threads=cpu_threads, local_files_only=True)
        except Exception as e:  # CTranslate2 raises plain RuntimeError/ValueError
            raise ModelIntegrityError(
                "Could not load the speech model in '%s'. It may be damaged, incomplete, or not a "
                "faster-whisper (CTranslate2) model. Engine said: %s" % (model_dir, str(e)[:300])) from e

    def transcribe(self, wav_path, language: Optional[str] = None, on_segment=None) -> Tuple[str, float, List[Segment]]:
        audio = read_wav_float32(wav_path)          # ndarray in -> faster-whisper skips its PyAV decoder
        kw = dict(language=language or None, vad_filter=True, condition_on_previous_text=False,   # fewer runaway loops
                  beam_size=self.beam_size, word_timestamps=self.word_timestamps,
                  # faster-whisper's default (2000 ms) fuses a whole conversation into ONE speech block
                  vad_parameters=dict(min_silence_duration_ms=self.vad_min_silence_ms))
        if self.word_timestamps:
            kw["hallucination_silence_threshold"] = 2.0     # needs word timing: drops text invented over long silences
        if self.sensitivity == "high":
            # keep quieter / overlapping / background speech that the defaults discard. Values are my choice,
            # NOT tuned on real data: they trade missed speech for extra (sometimes wrong) lines.
            kw.update(vad_parameters=dict(threshold=0.3, min_silence_duration_ms=min(300, self.vad_min_silence_ms), speech_pad_ms=300),
                      no_speech_threshold=0.85, log_prob_threshold=-1.6)
        if self.multilingual and not language:
            kw["multilingual"] = True                  # detect the language of every 30 s window, not once per file
        segs, info = self._model.transcribe(audio, **kw)
        out = []
        for s in segs:                              # generator: decoded lazily, window by window
            text = s.text.strip()
            if not text:
                continue
            conf = math.exp(s.avg_logprob) if s.avg_logprob is not None else None
            words = [{"s": float(w.start), "e": float(w.end), "w": w.word, "p": float(w.probability)}
                     for w in (getattr(s, "words", None) or [])] or None
            seg = Segment(s.start, s.end, text, conf, s.no_speech_prob, words)
            out.append(seg)
            if on_segment is not None:
                on_segment(seg)                      # may raise CancelledError: stops decoding at once
        return info.language, float(info.language_probability), out
