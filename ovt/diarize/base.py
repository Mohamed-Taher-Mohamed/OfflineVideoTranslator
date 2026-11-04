"""Speaker diarization interface: 'who speaks when' from the audio itself."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Callable, List, Optional


@dataclass
class Turn:
    start: float
    end: float
    speaker: int            # 0-based, numbered by first appearance

    def to_dict(self):
        return asdict(self)


class Diarizer(ABC):
    name = "abstract"

    @abstractmethod
    def diarize(self, wav_path, progress: Optional[Callable[[int, int], None]] = None) -> List[Turn]:
        """16 kHz mono WAV -> speaker turns sorted by start time. NOTE: not interruptible once started."""


def renumber(turns: List[Turn]) -> List[Turn]:
    """Sort by time and number speakers 0,1,2... in order of first appearance (stable across models/runs)."""
    turns = sorted(turns, key=lambda t: (t.start, t.end))
    ids = {}
    for t in turns:
        ids.setdefault(t.speaker, len(ids))
    return [Turn(t.start, t.end, ids[t.speaker]) for t in turns]
