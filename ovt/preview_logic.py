"""Pure (no Qt) helpers for the preview page, so the sync logic is unit-testable."""
from bisect import bisect_right
from typing import List, Optional, Sequence


def cue_index_at(starts: Sequence[float], ends: Sequence[float], t: float) -> Optional[int]:
    """Index of the cue showing at time t (seconds), or None in a gap. `starts` must be sorted."""
    i = bisect_right(starts, t) - 1
    if i < 0:
        return None
    # a short cue may sit inside the span of a longer previous one: look back a little
    for j in range(i, max(i - 3, -1), -1):
        if starts[j] <= t < ends[j]:
            return j
    return None


def neighbour_start(starts: Sequence[float], t: float, direction: int) -> Optional[float]:
    """Start time of the previous (-1) or next (+1) cue relative to t."""
    if direction > 0:
        i = bisect_right(starts, t + 1e-3)
        return starts[i] if i < len(starts) else None
    i = bisect_right(starts, t - 0.5) - 1          # 0.5 s grace so "previous" does not return the current cue
    return starts[i] if i >= 0 else None


def clock(sec: float) -> str:
    s = max(0, int(sec))
    return "%d:%02d:%02d" % (s // 3600, s % 3600 // 60, s % 60) if s >= 3600 else "%d:%02d" % (s // 60, s % 60)


def tail_lines(lines: List[str], keep: int = 8) -> str:
    extra = len(lines) - keep
    shown = lines[:keep]
    if extra > 0:
        shown.append("… (+%d)" % extra)
    return "\n".join(shown)


def parse_clock(text: str) -> float:
    """'90' -> 90, '9:15' -> 555, '1:02:03' -> 3723, '0:09:15.5' -> 555.5. Empty -> 0.0. Raises ValueError."""
    text = (text or "").strip().replace("،", ".").replace(",", ".")
    if not text:
        return 0.0
    parts = text.split(":")
    if len(parts) > 3 or any(not p.strip() for p in parts):
        raise ValueError("bad time: %r" % text)
    total = 0.0
    for p in parts:
        v = float(p)
        if v < 0:
            raise ValueError("negative time")
        total = total * 60 + v
    return total


def range_tag(t0: float, t1: float) -> str:
    """File-name tag for a partial run, e.g. '.t0915-1215' ('' when the whole file is processed)."""
    if t0 <= 0 and t1 <= 0:
        return ""
    def mmss(x):
        x = int(x)
        h, m, s = x // 3600, x % 3600 // 60, x % 60
        return "%d%02d%02d" % (h, m, s) if h else "%02d%02d" % (m, s)
    return ".t%s-%s" % (mmss(max(t0, 0)), mmss(t1) if t1 > 0 else "end")
