"""Speaker-aware recognition: transcribe each speaker TURN on its own.

Whisper's word times drift by a second or more, so attaching words to speakers by time gives wrong lines. Instead:
  1. the diarizer says who speaks when (turn edges are measured on the audio);
  2. speech the diarizer missed (very short words like "18") is added from the VAD, speaker unknown;
  3. every clip is cut out and clips are PACKED into <= ~28 s windows separated by digital silence, so one Whisper pass
     covers many clips (cost ~ speech duration, not clip count) and every recognised word falls inside exactly one clip;
  4. consecutive clips of the same speaker become one cue; a speaker change always starts a new cue.
Pure functions (no I/O except numpy slicing): unit-testable without any model."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .diarize.base import Turn

SR = 16000


@dataclass
class SpeakerOptions:
    merge_gap: float = 0.6            # same speaker, silence <= this: one clip
    pad: float = 0.15                 # extra audio around a clip (context for the recogniser)
    min_vad_region: float = 0.25      # VAD speech shorter than this is ignored
    vad_cover: float = 0.5            # a VAD region covered at least this much by a turn is not "missed"
    max_clip: float = 24.0            # a longer turn is cut at its best internal pause
    window: float = 28.0              # Whisper looks at 30 s: stay below it
    lead: float = 0.3                 # silence at the start of a window
    gap: float = 0.8                  # digital silence between clips in a window
    drop_distance: float = 0.25       # a recognised word farther than this from every clip sits in silence: dropped
    cue_gap: float = 2.5              # same speaker, silence <= this: same cue ...
    sentence_gap: float = 0.6         # ... but not across a sentence end followed by a pause this long
    cue_seconds: float = 7.0          # a cue never lasts longer
    cue_chars: int = 84               # ... nor holds more characters


@dataclass
class Clip:
    start: float                      # true speech start (seconds, original timeline)
    end: float
    speaker: Optional[int]            # 0-based; None = speech the diarizer did not attribute
    a: float = 0.0                    # padded extraction bounds
    b: float = 0.0


@dataclass
class Batch:
    clips: List[Clip]
    offsets: List[float]              # where each clip starts inside the packed audio (seconds)
    length: float


@dataclass
class ClipText:
    clip: Clip
    text: str
    confidence: Optional[float] = None
    words: Optional[list] = None


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def _split_clip(c: Clip, vad: Sequence[Tuple[float, float]], max_len: float) -> List[Clip]:
    if c.end - c.start <= max_len:
        return [c]
    inner = [(s, e) for s, e in vad if s < c.end and e > c.start]
    cut = None
    target = c.start + max_len / 2
    for (s0, e0), (s1, e1) in zip(inner, inner[1:]):          # the pause nearest to the middle of the long turn
        mid = (e0 + s1) / 2
        if c.start + 2 < mid < c.end - 2 and (cut is None or abs(mid - target) < abs(cut - target)):
            cut = mid
    if cut is None:
        cut = c.start + max_len                                # no pause at all: hard cut (rare)
    left, right = Clip(c.start, cut, c.speaker), Clip(cut, c.end, c.speaker)
    return _split_clip(left, vad, max_len) + _split_clip(right, vad, max_len)


def make_clips(turns: Sequence[Turn], vad: Sequence[Tuple[float, float]], total: float,
               opt: SpeakerOptions = None) -> List[Clip]:
    opt = opt or SpeakerOptions()
    clips: List[Clip] = []
    for t in sorted(turns, key=lambda t: (t.start, t.end)):
        if clips and clips[-1].speaker == t.speaker and t.start - clips[-1].end <= opt.merge_gap:
            clips[-1].end = max(clips[-1].end, t.end)
        else:
            clips.append(Clip(t.start, t.end, t.speaker))
    for s, e in vad:                                           # speech the diarizer missed
        if e - s < opt.min_vad_region:
            continue
        covered = sum(_overlap(s, e, c.start, c.end) for c in clips)
        if covered < opt.vad_cover * (e - s):
            clips.append(Clip(s, e, None))
    clips.sort(key=lambda c: (c.start, c.end))
    out: List[Clip] = []
    for c in clips:
        out += _split_clip(c, vad, opt.max_clip)
    for i, c in enumerate(out):                                # padded bounds, never overlapping the neighbours
        prev_end = out[i - 1].end if i else 0.0
        next_start = out[i + 1].start if i + 1 < len(out) else total
        c.a = max(0.0, c.start - opt.pad, (prev_end + c.start) / 2 if i else 0.0)
        c.b = min(total, c.end + opt.pad, (c.end + next_start) / 2 if i + 1 < len(out) else total)
        c.a, c.b = min(c.a, c.start), max(c.b, c.end)
    return out


def pack(clips: Sequence[Clip], opt: SpeakerOptions = None) -> List[Batch]:
    opt = opt or SpeakerOptions()
    batches: List[Batch] = []
    cur: List[Clip] = []
    offs: List[float] = []
    pos = opt.lead
    for c in clips:
        dur = c.b - c.a
        if cur and pos + dur > opt.window:
            batches.append(Batch(cur, offs, pos))
            cur, offs, pos = [], [], opt.lead
        cur.append(c)
        offs.append(pos)
        pos += dur + opt.gap
    if cur:
        batches.append(Batch(cur, offs, pos))
    return batches


def batch_audio(samples: np.ndarray, batch: Batch, opt: SpeakerOptions = None) -> np.ndarray:
    opt = opt or SpeakerOptions()
    parts = [np.zeros(int(opt.lead * SR), dtype=np.float32)]
    for c in batch.clips:
        parts.append(samples[int(c.a * SR):int(c.b * SR)].astype(np.float32, copy=False))
        parts.append(np.zeros(int(opt.gap * SR), dtype=np.float32))
    return np.concatenate(parts)


def assign_words(batch: Batch, words: Sequence[dict], opt: SpeakerOptions = None) -> List[List[dict]]:
    """Distribute recognised words (times inside the packed audio) over the batch's clips, in order.
    Words that sit in the silence between clips (hallucinations) are dropped."""
    opt = opt or SpeakerOptions()
    spans = [(o, o + (c.b - c.a)) for c, o in zip(batch.clips, batch.offsets)]
    out: List[List[dict]] = [[] for _ in batch.clips]
    prev = 0
    for w in words:
        mid = (w["s"] + w["e"]) / 2.0
        best, best_d = None, None
        for j in range(prev, len(spans)):
            a, b = spans[j]
            d = 0.0 if a <= mid <= b else min(abs(mid - a), abs(mid - b))
            if best_d is None or d < best_d - 1e-9:
                best, best_d = j, d
            if a - mid > (best_d or 0.0):
                break
        if best is not None and best_d <= opt.drop_distance:
            out[best].append(w)
            prev = best
    return out


def _ends_sentence(text: str) -> bool:
    return text.rstrip().endswith((".", "?", "!", "…", "؟", "。"))


def cues_from_clips(items: Sequence[ClipText], opt: SpeakerOptions = None):
    """Group recognised clips into cues -> list of dicts {start,end,text,speaker(1-based or None),confidence,words}."""
    opt = opt or SpeakerOptions()
    cues: List[dict] = []
    sp_ids: Dict[int, int] = {}
    for it in items:
        text = " ".join(it.text.split())
        if not text:
            continue
        c = it.clip
        spk = None if c.speaker is None else sp_ids.setdefault(c.speaker, len(sp_ids) + 1)
        if cues:
            last = cues[-1]
            gap = c.start - last["end"]
            same = spk is not None and spk == last["speaker"]
            fits = (c.end - last["start"] <= opt.cue_seconds and len(last["text"]) + 1 + len(text) <= opt.cue_chars)
            sentence_break = _ends_sentence(last["text"]) and gap >= opt.sentence_gap
            if same and gap <= opt.cue_gap and fits and not sentence_break:
                last["text"] += " " + text
                last["end"] = c.end
                last["words"] = (last["words"] or []) + (it.words or [])
                if it.confidence is not None:
                    last["conf"].append(it.confidence)
                continue
        cues.append({"start": c.start, "end": c.end, "text": text, "speaker": spk,
                     "conf": [it.confidence] if it.confidence is not None else [], "words": list(it.words or [])})
    for c in cues:
        c["confidence"] = (sum(c["conf"]) / len(c["conf"])) if c["conf"] else None
        del c["conf"]
    return cues
