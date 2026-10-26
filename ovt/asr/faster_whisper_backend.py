"""faster-whisper (CTranslate2) backend. Loads ONLY from a local folder - never from the hub."""
import math
from typing import List, Optional, Tuple

from .base import ASRBackend, Segment


class FasterWhisperBackend(ASRBackend):
    name = "faster-whisper"

    def __init__(self, model_dir: str, device: str = "auto", compute_type: str = "auto", cpu_threads: int = 0):
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

    def transcribe(self, wav_path, language: Optional[str] = None) -> Tuple[str, float, List[Segment]]:
        segs, info = self._model.transcribe(
            str(wav_path), language=language or None, vad_filter=True,
            condition_on_previous_text=False,       # reduces runaway repetition
            beam_size=5, word_timestamps=False)
        out = []
        for s in segs:                              # generator: decoded lazily, window by window
            text = s.text.strip()
            if not text:
                continue
            conf = math.exp(s.avg_logprob) if s.avg_logprob is not None else None
            out.append(Segment(s.start, s.end, text, conf, s.no_speech_prob))
        return info.language, float(info.language_probability), out
