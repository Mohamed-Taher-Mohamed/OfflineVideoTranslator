"""Subtitle model, parsers (SRT/VTT/ASS/SSA), writer, validator and timing tools."""
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List

from .errors import SubtitleFormatError
from .fsutil import atomic_write_bytes

RLM = "\u200f"


@dataclass
class Cue:
    start: float
    end: float
    text: str
    source_text: str = ""
    orig_start: float = None
    orig_end: float = None
    speaker: str = ""          # reserved for the future dubbing stage
    confidence: float = None   # ASR confidence if known

    def __post_init__(self):
        if self.orig_start is None:
            self.orig_start = self.start
        if self.orig_end is None:
            self.orig_end = self.end

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


# ---------- time ----------
def fmt_ts(sec: float, sep: str = ",") -> str:
    ms = int(round(max(sec, 0.0) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return "%02d:%02d:%02d%s%03d" % (h, m, s, sep, ms)


_TS = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[,.](\d{1,3})")


def parse_ts(s: str) -> float:
    m = _TS.fullmatch(s.strip())
    if not m:
        raise SubtitleFormatError("bad timestamp: %r" % s)
    h, mi, se, ms = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + int(se) + int(ms.ljust(3, "0")) / 1000.0


# ---------- reading ----------
def read_text_guess(path) -> str:
    raw = Path(path).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8")
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    for enc in ("utf-8", "cp1256", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


_TAGS = re.compile(r"<[^>]+>|\{\\[^}]*\}")


def clean_markup(t: str) -> str:
    return _TAGS.sub("", t).replace("\\N", "\n").replace("\\n", "\n").strip()


def parse_srt(text: str) -> List[Cue]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    cues = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [l for l in block.split("\n")]
        ti = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if ti is None:
            continue
        a, b = lines[ti].split("-->")[:2]
        try:
            st, en = parse_ts(a), parse_ts(b.strip().split()[0])
        except (SubtitleFormatError, IndexError):
            continue
        body = clean_markup("\n".join(lines[ti + 1:]))
        cues.append(Cue(st, en, body))
    return cues


def parse_vtt(text: str) -> List[Cue]:
    text = text.replace("\r\n", "\n").lstrip("\ufeff")
    if not text.startswith("WEBVTT"):
        raise SubtitleFormatError("not a WebVTT file")
    body = re.sub(r"^WEBVTT[^\n]*\n", "", text)
    body = re.sub(r"^(NOTE|STYLE)[^\n]*(\n[^\n]+)*", "", body, flags=re.M)
    return parse_srt(body)


def parse_ass(text: str) -> List[Cue]:
    cues, fmt = [], None
    for line in text.replace("\r\n", "\n").split("\n"):
        if line.startswith("Format:") and fmt is None and "Start" in line:
            fmt = [x.strip() for x in line[7:].split(",")]
        elif line.startswith("Dialogue:") and fmt:
            parts = line[9:].split(",", len(fmt) - 1)
            if len(parts) < len(fmt):
                continue
            row = dict(zip(fmt, parts))
            try:
                st, en = parse_ts(row["Start"].replace(".", ",")), parse_ts(row["End"].replace(".", ","))
            except (KeyError, SubtitleFormatError):
                continue
            cues.append(Cue(st, en, clean_markup(row.get("Text", ""))))
    return cues


def load_subtitles(path) -> List[Cue]:
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in (".srt", ".vtt", ".ass", ".ssa"):
        raise SubtitleFormatError("unsupported subtitle format: %s" % ext)
    text = read_text_guess(p)
    if ext == ".vtt":
        cues = parse_vtt(text)
    elif ext in (".ass", ".ssa"):
        cues = parse_ass(text)
    elif ext == ".srt":
        cues = parse_srt(text)
    else:
        raise SubtitleFormatError("unsupported subtitle format: %s" % ext)
    cues = [c for c in cues if c.text.strip()]
    cues.sort(key=lambda c: (c.start, c.end))
    return cues


# ---------- writing ----------
def _rtl(line: str, on: bool) -> str:
    if on and re.search(r"[\u0600-\u06ff]", line):
        return RLM + line + RLM
    return line


def render_srt(cues: List[Cue], rtl_marks: bool = False) -> str:
    out = []
    for i, c in enumerate(cues, 1):
        body = "\n".join(_rtl(l, rtl_marks) for l in c.text.split("\n"))
        out.append("%d\n%s --> %s\n%s\n" % (i, fmt_ts(c.start), fmt_ts(c.end), body))
    return "\n".join(out)


def write_srt(cues: List[Cue], path, bom: bool = False, rtl_marks: bool = False) -> Path:
    data = render_srt(cues, rtl_marks).encode("utf-8")
    if bom:
        data = b"\xef\xbb\xbf" + data
    atomic_write_bytes(path, data)
    return Path(path)


def to_txt(cues: List[Cue], use_source: bool = False) -> str:
    return "\n".join((c.source_text if use_source else c.text).replace("\n", " ") for c in cues) + "\n"


# ---------- validation ----------
def validate_srt(path) -> List[str]:
    """Return a list of problems; empty list means the file is a valid, ordered SRT."""
    problems = []
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return ["file is not valid UTF-8"]
    text = text.replace("\r\n", "\n")
    blocks = [b for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    if not blocks:
        return ["file contains no cues"]
    prev_end, prev_start = -1.0, -1.0
    for n, b in enumerate(blocks, 1):
        lines = b.split("\n")
        if len(lines) < 3:
            problems.append("cue %d: needs index, time line and text" % n)
            continue
        if lines[0].strip() != str(n):
            problems.append("cue %d: index is %r" % (n, lines[0].strip()))
        m = re.fullmatch(r"(\d\d:\d\d:\d\d,\d{3}) --> (\d\d:\d\d:\d\d,\d{3})", lines[1].strip())
        if not m:
            problems.append("cue %d: bad time line %r" % (n, lines[1]))
            continue
        st, en = parse_ts(m.group(1)), parse_ts(m.group(2))
        if en <= st:
            problems.append("cue %d: end <= start" % n)
        if st < prev_start:
            problems.append("cue %d: out of order" % n)
        if st < prev_end - 1e-6:
            problems.append("cue %d: overlaps previous cue" % n)
        if not "".join(lines[2:]).replace(RLM, "").strip():
            problems.append("cue %d: empty text" % n)
        prev_start, prev_end = st, en
    return problems


# ---------- layout ----------
def wrap_text(text: str, max_chars: int) -> List[str]:
    lines = []
    for para in text.split("\n"):
        cur = ""
        for w in para.split():
            if not cur:
                cur = w
            elif len(cur) + 1 + len(w) <= max_chars:
                cur += " " + w
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
    return lines or [""]


def reflow(cues: List[Cue], max_chars: int = 42, max_lines: int = 2) -> List[Cue]:
    """Wrap to max_chars. If a cue needs more than max_lines, split it into several
    consecutive cues with time shared by text length. Text is never dropped."""
    out = []
    for c in cues:
        lines = wrap_text(c.text, max_chars)
        if len(lines) <= max_lines:
            out.append(Cue(c.start, c.end, "\n".join(lines), c.source_text, c.orig_start, c.orig_end, c.speaker, c.confidence))
            continue
        groups = [lines[i:i + max_lines] for i in range(0, len(lines), max_lines)]
        weights = [sum(len(l) for l in g) or 1 for g in groups]
        total, t = float(sum(weights)), c.start
        for g, w in zip(groups, weights):
            dur = (c.end - c.start) * w / total
            out.append(Cue(t, t + dur, "\n".join(g), c.source_text, c.orig_start, c.orig_end, c.speaker, c.confidence))
            t += dur
    return out


def fix_timing(cues: List[Cue], min_dur: float = 0.8, gap: float = 0.001) -> List[Cue]:
    cues = sorted(cues, key=lambda c: (c.start, c.end))
    for i, c in enumerate(cues):
        nxt = cues[i + 1].start if i + 1 < len(cues) else None
        if c.end < c.start + min_dur:
            c.end = c.start + min_dur
        if nxt is not None and c.end > nxt - gap:
            c.end = max(nxt - gap, c.start + 0.04)
            if c.end > nxt - gap:      # next cue starts essentially together: nudge it
                cues[i + 1].start = c.end + gap
                cues[i + 1].end = max(cues[i + 1].end, cues[i + 1].start + 0.04)
    return cues


def shift(cues: List[Cue], delta: float) -> List[Cue]:
    for c in cues:
        c.start = max(0.0, c.start + delta)
        c.end = max(c.start + 0.04, c.end + delta)
    return cues


def reset_timing(cues: List[Cue]) -> List[Cue]:
    for c in cues:
        c.start, c.end = c.orig_start, c.orig_end
    return cues
