from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class Support:
    ok: bool
    route: str = ""        # "direct", "pivot via en", ...
    reason: str = ""


class Translator(ABC):
    name = "abstract"
    version = ""
    handles_mixed = False       # True if it can translate lines whose source language differs from line to line

    @abstractmethod
    def supports(self, src: str, tgt: str) -> Support:
        """Honest answer: is this exact language pair installed and runnable?"""

    @abstractmethod
    def translate(self, texts: List[str], src: str, tgt: str) -> List[str]:
        """Translate; MUST return exactly len(texts) strings."""

    def close(self):
        pass
