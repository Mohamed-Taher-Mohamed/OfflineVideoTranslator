"""Split recogniser segments into subtitle-sized cues using the timing of every word.

Why: Whisper often returns one long segment for a whole exchange (a 13-second "segment" holding three speakers'
lines). The real boundaries are the pauses between words, so we cut there. Without speaker identification this is
the practical approximation of "one line per speaker turn": a turn almost always starts after a pause.
Pure functions, no I/O: unit-testable."""
from dataclasses import dataclass
from typing import List

from .asr.base import Segment

_END = (".", "?", "!", "…", "。", "؟", "！", "？")


@dataclass
class SplitOptions:
    pause: float = 0.40            # a gap this long between two words always starts a new cue
    sentence_pause: float = 0.20   # a (shorter) gap after . ? ! also starts a new cue
    max_seconds: float = 6.5       # a cue never lasts longer than this
    max_chars: int = 84            # ... nor holds more characters (2 lines x 42)
    tail: float = 0.15             # seconds the cue stays on screen after the last word (never into the next cue)


def _text(words) -> str:
    return " ".join("".join(w["w"] for w in words).split())


def _ends_sentence(word: dict) -> bool:
    return word["w"].strip().endswith(_END)


def _gap_score(words, i: int) -> float:
    """How good is a cut between words[i-1] and words[i]: the bigger the pause the better, punctuation helps."""
    return (words[i]["s"] - words[i - 1]["e"]) + (0.3 if _ends_sentence(words[i - 1]) else 0.0)


def _split_long(words, opt: SplitOptions) -> List[list]:
    """Recursively cut an over-long group at its best internal boundary."""
    dur = words[-1]["e"] - words[0]["s"]
    if len(words) < 2 or (dur <= opt.max_seconds and len(_text(words)) <= opt.max_chars):
        return [words]
    lo = max(1, len(words) // 6)                         # avoid leaving a one-word orphan when a better cut exists
    candidates = range(lo, len(words) - lo + 1) if len(words) > 2 * lo else range(1, len(words))
    cut = max(candidates, key=lambda i: (_gap_score(words, i), -abs(i - len(words) / 2)))
    return _split_long(words[:cut], opt) + _split_long(words[cut:], opt)


def split_words(words: List[dict], opt: SplitOptions) -> List[list]:
    groups: List[list] = []
    for w in words:
        if groups:
            prev = groups[-1][-1]
            gap = w["s"] - prev["e"]
            if gap >= opt.pause or (_ends_sentence(prev) and gap >= opt.sentence_pause):
                groups.append([])
        else:
            groups.append([])
        groups[-1].append(w)
    out: List[list] = []
    for g in groups:
        out += _split_long(g, opt)
    return out


def resegment(segments: List[Segment], opt: SplitOptions = None) -> List[Segment]:
    """Segments with word timing are split at pauses; segments without it are returned unchanged."""
    opt = opt or SplitOptions()
    out: List[Segment] = []
    for seg in segments:
        words = [w for w in (seg.words or []) if w.get("w", "").strip()]
        if not words:
            out.append(seg)
            continue
        for g in split_words(words, opt):
            text = _text(g)
            if not text:
                continue
            probs = [w.get("p") for w in g if w.get("p") is not None]
            out.append(Segment(g[0]["s"], g[-1]["e"], text, (sum(probs) / len(probs)) if probs else seg.confidence,
                               seg.no_speech_prob, g))
    out.sort(key=lambda s: s.start)
    for i, s in enumerate(out):                           # short tail so the last word is readable, never overlapping
        limit = out[i + 1].start - 0.02 if i + 1 < len(out) else s.end + opt.tail
        s.end = max(s.end, min(s.end + opt.tail, limit))
    return out
