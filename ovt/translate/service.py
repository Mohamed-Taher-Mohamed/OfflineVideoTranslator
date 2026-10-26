"""Context-aware, resumable, verifiable translation of subtitle cues.

1. Cues are merged into sentence-like units (so MT sees whole sentences).
2. Units are translated in batches; every finished batch is reported (for resume).
3. Each unit's translation is re-split over its original cues (timing stays original).
4. Output is sanity-checked; problems become warnings, never silent fake translations.
"""
import re
from typing import Callable, Dict, List, Optional

from ..glossary import Glossary
from ..subtitles import Cue

_END = re.compile(r"[.!?؟…:;。！？]\s*[\"')\]»”]*$")
_ARABIC = re.compile(r"[\u0600-\u06ff]")
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


_WORD_LOOP = re.compile(r"(?<!\S)(\S+)(?:\s+\1){3,}(?!\S)")      # same word 4+ times in a row
_CHAR_LOOP = re.compile(r"(.)\1{7,}")                              # same character 8+ times in a row


def collapse_loops(text: str):
    """Shrink obvious MT repetition loops. Returns (text, changed). Normal doubled words are untouched."""
    new = _WORD_LOOP.sub(r"\1", text)
    new = _CHAR_LOOP.sub(r"\1", new)
    return new, new != text


def build_units(cues: List[Cue], context_merge: bool = True, max_gap: float = 1.0,
                max_chars: int = 240, max_cues: int = 4) -> List[List[int]]:
    units, cur, chars = [], [], 0
    for i, c in enumerate(cues):
        if cur:
            prev = cues[cur[-1]]
            joinable = (context_merge and not _END.search(prev.source_text.strip())
                        and c.start - prev.end <= max_gap and chars + len(c.source_text) <= max_chars
                        and len(cur) < max_cues)
            if not joinable:
                units.append(cur)
                cur, chars = [], 0
        cur.append(i)
        chars += len(c.source_text)
    if cur:
        units.append(cur)
    return units


def split_over_cues(translated: str, weights: List[int]) -> List[str]:
    """Distribute translated words over len(weights) cues, proportional to source length."""
    n = len(weights)
    if n == 1:
        return [translated.strip()]
    words = translated.split()
    if len(words) < n:
        parts = [" ".join(words)] + [""] * (n - 1)
        return parts
    total, out, pos, acc = float(sum(weights) or 1), [], 0, 0.0
    for k, w in enumerate(weights):
        acc += w
        end = len(words) if k == n - 1 else max(pos + 1, min(len(words) - (n - 1 - k), round(len(words) * acc / total)))
        out.append(" ".join(words[pos:end]))
        pos = end
    return out


def translate_cues(translator, cues: List[Cue], src: str, tgt: str, glossary: Optional[Glossary] = None,
                   context_merge: bool = True, batch_size: int = 16,
                   cache: Optional[Dict[str, str]] = None,
                   on_batch: Optional[Callable[[Dict[str, str], int, int], None]] = None,
                   check_cancel: Callable[[], None] = lambda: None) -> List[str]:
    """Fills cue.text for every cue; returns warnings. `cache` maps unit-index -> translated unit."""
    glossary = glossary or Glossary()
    cache = cache if cache is not None else {}
    warnings: List[str] = []
    units = build_units(cues, context_merge)
    tables: Dict[int, dict] = {}
    todo = [u for u in range(len(units)) if str(u) not in cache]

    for b in range(0, len(todo), batch_size):
        check_cancel()
        idxs = todo[b:b + batch_size]
        protected = []
        for u in idxs:
            text = " ".join(cues[i].source_text.replace("\n", " ").strip() for i in units[u])
            ptxt, table = glossary.protect(text)
            tables[u] = table
            protected.append(ptxt)
        out = translator.translate(protected, src, tgt)
        if len(out) != len(protected):
            raise RuntimeError("translator returned %d results for %d inputs" % (len(out), len(protected)))
        for u, o in zip(idxs, out):
            restored, lost = Glossary.restore(o or "", tables[u])
            cache[str(u)] = restored
            if lost:
                warnings.append("cue %d: glossary term lost by the engine: %s" % (units[u][0] + 1, ", ".join(lost)))
        if on_batch:
            on_batch(cache, min(b + batch_size, len(todo)), len(todo))

    for u, idxs in enumerate(units):
        # glossary terms of cached units (resumed run) are already restored in the cache
        pieces = split_over_cues(cache[str(u)], [max(1, len(cues[i].source_text)) for i in idxs])
        for i, piece in zip(idxs, pieces):
            cue = cues[i]
            src_has_letters = bool(_LETTER.search(cue.source_text))
            if src_has_letters and not piece.strip():
                warnings.append("cue %d: empty translation, source text kept" % (i + 1))
                piece = cue.source_text
            elif tgt == "ar" and src != "ar" and src_has_letters and len(cue.source_text) > 12 and not _ARABIC.search(piece):
                warnings.append("cue %d: output has no Arabic letters (possibly untranslated)" % (i + 1))
            piece, looped = collapse_loops(piece)
            if looped:
                warnings.append("cue %d: repetition loop in the engine output was shortened; check this line" % (i + 1))
            cue.text = piece.strip()
    return warnings
