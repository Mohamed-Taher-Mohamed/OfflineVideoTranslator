"""Media probing and streaming extraction through ffmpeg/ffprobe (subprocess, no shell).

Design points:
- Paths are passed as argv items, never through a shell -> Arabic names/spaces are safe.
- Audio is extracted in time windows straight to 16 kHz mono WAV; the whole video is
  never loaded into memory and the source file is opened read-only.
- Files are treated as untrusted: `-protocol_whitelist file,pipe` stops a crafted
  playlist/container from making ffmpeg open network URLs.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

from .errors import MediaError, NoAudioError, UnsupportedMediaError

SUPPORTED_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts", ".mp3", ".wav", ".m4a", ".flac", ".ogg"}
TEXT_SUB_CODECS = {"subrip", "srt", "ass", "ssa", "webvtt", "mov_text", "text", "microdvd"}
BITMAP_SUB_CODECS = {"hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub"}

_ISO3_TO_1 = {
    "eng": "en", "ara": "ar", "fra": "fr", "fre": "fr", "deu": "de", "ger": "de", "spa": "es",
    "ita": "it", "por": "pt", "rus": "ru", "jpn": "ja", "kor": "ko", "zho": "zh", "chi": "zh",
    "tur": "tr", "hin": "hi", "nld": "nl", "dut": "nl", "pol": "pl", "ukr": "uk", "fas": "fa",
    "per": "fa", "heb": "he", "ind": "id", "swe": "sv", "vie": "vi", "tha": "th", "ell": "el",
}

_NO_WINDOW = {"creationflags": 0x08000000} if sys.platform == "win32" else {}


def iso3_to_1(code: str) -> str:
    return _ISO3_TO_1.get((code or "").lower(), (code or "").lower() if len(code or "") == 2 else "")


def find_binary(name: str, engines_dir: Optional[Path] = None) -> str:
    env = os.environ.get("OVT_" + name.upper())
    if env and Path(env).exists():
        return env
    exe = name + (".exe" if sys.platform == "win32" else "")
    if engines_dir:
        cand = Path(engines_dir) / exe
        if cand.exists():
            return str(cand)
    found = shutil.which(name)
    if found:
        return found
    raise MediaError("%s not found. Put it in engines/ or set OVT_%s." % (name, name.upper()), binary=name)


@dataclass
class AudioTrack:
    index: int            # position among audio streams (what -map 0:a:N expects)
    stream_index: int
    codec: str
    channels: int
    sample_rate: int
    language: str = ""
    title: str = ""


@dataclass
class SubtitleTrack:
    index: int            # position among subtitle streams
    stream_index: int
    codec: str
    language: str = ""
    title: str = ""
    is_text: bool = True


@dataclass
class MediaInfo:
    path: str
    name: str
    size_bytes: int
    duration: float
    container: str
    has_video: bool
    audio_tracks: List[AudioTrack] = field(default_factory=list)
    subtitle_tracks: List[SubtitleTrack] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


class FFmpeg:
    def __init__(self, engines_dir: Optional[Path] = None):
        self.ffmpeg = find_binary("ffmpeg", engines_dir)
        self.ffprobe = find_binary("ffprobe", engines_dir)

    def _run(self, args, timeout=None):
        return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
                              stdin=subprocess.DEVNULL, **_NO_WINDOW)

    def probe(self, path) -> MediaInfo:
        p = Path(path)
        if not p.is_file():
            raise MediaError("File not found: %s" % p, path=str(p))
        if not os.access(p, os.R_OK):
            raise MediaError("File is not readable: %s" % p, path=str(p))
        if p.stat().st_size == 0:
            raise UnsupportedMediaError("File is empty: %s" % p, path=str(p))
        r = self._run([self.ffprobe, "-v", "error", "-protocol_whitelist", "file,pipe", "-print_format", "json",
                       "-show_format", "-show_streams", str(p)], timeout=120)
        if r.returncode != 0:
            msg = r.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["unknown error"]
            raise UnsupportedMediaError("Cannot read this file (corrupt or unsupported): %s" % msg[0], path=str(p))
        try:
            d = json.loads(r.stdout.decode("utf-8", "replace"))
        except ValueError as e:
            raise UnsupportedMediaError("ffprobe returned invalid data", path=str(p)) from e
        fmt = d.get("format", {})
        info = MediaInfo(path=str(p), name=p.name, size_bytes=p.stat().st_size,
                         duration=float(fmt.get("duration") or 0.0),
                         container=fmt.get("format_name", ""), has_video=False)
        ai = si = 0
        for s in d.get("streams", []):
            tags = {k.lower(): v for k, v in (s.get("tags") or {}).items()}
            ct = s.get("codec_type")
            if ct == "video" and not (s.get("disposition") or {}).get("attached_pic"):
                info.has_video = True
            elif ct == "audio":
                info.audio_tracks.append(AudioTrack(
                    index=ai, stream_index=s["index"], codec=s.get("codec_name", ""),
                    channels=int(s.get("channels") or 0), sample_rate=int(s.get("sample_rate") or 0),
                    language=tags.get("language", ""), title=tags.get("title", "")))
                ai += 1
            elif ct == "subtitle":
                codec = s.get("codec_name", "")
                info.subtitle_tracks.append(SubtitleTrack(
                    index=si, stream_index=s["index"], codec=codec,
                    language=tags.get("language", ""), title=tags.get("title", ""),
                    is_text=codec not in BITMAP_SUB_CODECS))
                si += 1
        if info.duration <= 0:
            # some containers only expose duration per stream
            for s in d.get("streams", []):
                try:
                    info.duration = max(info.duration, float(s.get("duration") or 0))
                except ValueError:
                    pass
        if not info.audio_tracks:
            raise NoAudioError("This file has no audio track.", path=str(p))
        return info

    def extract_wav(self, path, audio_index: int, start: float, duration: float, out_wav,
                    timeout: Optional[float] = None) -> Path:
        """Decode one window of one audio track to 16 kHz mono PCM WAV."""
        out = Path(out_wav)
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [self.ffmpeg, "-nostdin", "-v", "error", "-protocol_whitelist", "file,pipe",
               "-ss", "%.3f" % start, "-i", str(path), "-map", "0:a:%d" % audio_index,
               "-t", "%.3f" % duration, "-vn", "-sn", "-dn", "-ac", "1", "-ar", "16000",
               "-c:a", "pcm_s16le", "-y", str(out)]
        r = self._run(cmd, timeout=timeout)
        if r.returncode != 0 or not out.exists():
            err = r.stderr.decode("utf-8", "replace").strip()[-300:]
            if out.exists():
                out.unlink()
            raise MediaError("Audio extraction failed: %s" % err, path=str(path), start=start)
        return out

    def find_silences(self, path, audio_index: int, start: float, dur: float, noise_db: float = -35.0,
                      min_len: float = 0.25):
        """Silent intervals (absolute seconds) inside one window of one audio track."""
        cmd = [self.ffmpeg, "-nostdin", "-hide_banner", "-protocol_whitelist", "file,pipe", "-ss", "%.3f" % max(start, 0.0),
               "-i", str(path), "-map", "0:a:%d" % audio_index, "-t", "%.3f" % dur, "-vn", "-sn",
               "-af", "silencedetect=noise=%ddB:d=%.2f" % (noise_db, min_len), "-f", "null", "-"]
        r = self._run(cmd, timeout=120)
        text = r.stderr.decode("utf-8", "replace")
        out, open_at = [], None
        for line in text.splitlines():
            m = re.search(r"silence_start:\s*(-?[\d.]+)", line)
            if m:
                open_at = max(float(m.group(1)), 0.0)
            m = re.search(r"silence_end:\s*(-?[\d.]+)", line)
            if m and open_at is not None:
                out.append((start + open_at, start + float(m.group(1))))
                open_at = None
        if open_at is not None:
            out.append((start + open_at, start + dur))
        return out

    def plan_chunks(self, path, audio_index: int, duration: float, chunk_s: float, snap: bool = True,
                    window: float = 20.0, start: float = 0.0, end: Optional[float] = None):
        """[(start, end)] covering the file. Boundaries move to the middle of the longest silence within
        +-window seconds of the nominal cut, so sentences are rarely cut in half. Falls back to fixed cuts."""
        if end is None or end <= 0:
            end = duration
        if end <= 0:
            return [(float(start), float(start) + float(chunk_s))]
        bounds = [float(start)]
        while True:
            nominal = bounds[-1] + chunk_s
            if nominal >= end - 1.0:
                break
            cut = nominal
            if snap:
                lo, hi = max(nominal - window, bounds[-1] + 0.3 * chunk_s), min(nominal + window, end - 1.0)
                if hi > lo:
                    try:
                        gaps = [(a, b) for a, b in self.find_silences(path, audio_index, lo, hi - lo)
                                if b - a >= 0.25]
                    except Exception:
                        gaps = []
                    if gaps:
                        a, b = max(gaps, key=lambda g: g[1] - g[0])
                        cut = (a + b) / 2.0
            bounds.append(cut)
        bounds.append(float(end))
        return list(zip(bounds[:-1], bounds[1:]))

    def mean_volume_db(self, wav) -> Optional[float]:
        r = self._run([self.ffmpeg, "-nostdin", "-hide_banner", "-i", str(wav), "-af", "volumedetect",
                       "-f", "null", "-"], timeout=120)
        m = re.search(r"mean_volume:\s*(-?[\d.]+) dB", r.stderr.decode("utf-8", "replace"))
        return float(m.group(1)) if m else None

    def extract_subtitle(self, path, sub_index: int, out_srt) -> Path:
        out = Path(out_srt)
        cmd = [self.ffmpeg, "-nostdin", "-v", "error", "-protocol_whitelist", "file,pipe", "-i", str(path),
               "-map", "0:s:%d" % sub_index, "-c:s", "srt", "-f", "srt", "-y", str(out)]
        r = self._run(cmd, timeout=300)
        if r.returncode != 0 or not out.exists():
            raise UnsupportedMediaError("Cannot extract this subtitle track (image-based tracks "
                                        "such as PGS/DVD need OCR, which is not included): %s"
                                        % r.stderr.decode("utf-8", "replace").strip()[-200:])
        return out
