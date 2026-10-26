"""ASR interface. Any engine (faster-whisper on desktop, whisper.cpp on Android, ...) implements it."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple


@dataclass
class Segment:
    start: float
    end: float
    text: str
    confidence: Optional[float] = None   # 0..1, from avg logprob when available
    no_speech_prob: Optional[float] = None

    def to_dict(self):
        return asdict(self)


class ASRBackend(ABC):
    name = "abstract"

    @abstractmethod
    def transcribe(self, wav_path, language: Optional[str] = None) -> Tuple[str, float, List[Segment]]:
        """Return (detected_or_given_language, language_probability, segments with times relative to the wav)."""

    def close(self):
        pass


def dedupe_repeats(segments: List[Segment], max_repeat: int = 2) -> List[Segment]:
    """Whisper can loop on one phrase. Keep at most `max_repeat` identical consecutive segments."""
    out, run, last = [], 0, None
    for s in segments:
        key = " ".join(s.text.lower().split())
        if key and key == last:
            run += 1
        else:
            run, last = 1, key
        if run <= max_repeat:
            out.append(s)
    return out
