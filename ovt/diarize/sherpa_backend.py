"""Offline speaker diarization with sherpa-onnx: pyannote segmentation + a speaker-embedding model + clustering.
Runs on CPU (about 0.13x real time on the dev box: 26 s of audio in ~3.3 s)."""
from pathlib import Path
from typing import Callable, List, Optional

from ..asr.audio_io import read_wav_float32
from ..pathutil import ascii_path
from .base import Diarizer, Turn, renumber


class SherpaDiarizer(Diarizer):
    name = "sherpa-onnx"

    def __init__(self, segmentation_model, embedding_model, num_speakers: int = 0, threshold: float = 0.8,
                 threads: int = 2, min_duration_on: float = 0.2, min_duration_off: float = 0.3):
        import sherpa_onnx                                   # lazy: only needed when speakers are requested
        self._sherpa = sherpa_onnx
        cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=ascii_path(segmentation_model)),
                num_threads=threads),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=ascii_path(embedding_model), num_threads=threads),
            clustering=sherpa_onnx.FastClusteringConfig(num_clusters=int(num_speakers) if num_speakers else -1,
                                                        threshold=float(threshold)),
            min_duration_on=min_duration_on, min_duration_off=min_duration_off)
        if not cfg.validate():
            from ..errors import ModelIntegrityError
            raise ModelIntegrityError("The speaker models could not be loaded (damaged or incompatible files).")
        self._sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)

    def diarize(self, wav_path, progress: Optional[Callable[[int, int], None]] = None) -> List[Turn]:
        audio = read_wav_float32(wav_path)

        def cb(done, total):
            if progress:
                progress(int(done), int(total))
            return 0                                          # this library version cannot be stopped from here
        result = self._sd.process(audio, callback=cb).sort_by_start_time()
        return renumber([Turn(float(s.start), float(s.end), int(s.speaker)) for s in result])
