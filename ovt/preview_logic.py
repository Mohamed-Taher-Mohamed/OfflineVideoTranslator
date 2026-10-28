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
