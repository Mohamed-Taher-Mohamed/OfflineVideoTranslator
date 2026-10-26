"""Export a NEW video with the Arabic subtitle track added (soft subtitles, stream copy: no re-encoding,
no quality loss, fast). The original file is never touched. Burned-in (hard) subtitles are NOT implemented."""
import subprocess
from pathlib import Path

from .errors import MediaError, OVTError
from .fsutil import ensure_free_space, unique_path
from .media import FFmpeg, _NO_WINDOW


def export_with_subtitles(ff: FFmpeg, video, srt, out_dir=None, lang="ara", title="Arabic", overwrite=False) -> Path:
    video, srt = Path(video), Path(srt)
    if not srt.is_file():
        raise MediaError("Subtitle file not found: %s" % srt)
    info = ff.probe(video)
    ext = video.suffix.lower()
    sub_codec = "mov_text" if ext in (".mp4", ".m4v", ".mov") else "srt"
    if ext in (".avi", ".webm"):
        ext, sub_codec = ".mkv", "srt"                      # these containers cannot hold SRT; use MKV
    out = Path(out_dir or video.parent) / (video.stem + ".ar" + ext)
    if out.resolve() == video.resolve():
        raise OVTError("Refusing to overwrite the original video")
    if not overwrite:
        out = unique_path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ensure_free_space(out.parent, int(video.stat().st_size * 1.05) + 50 * 2**20, "the new video file")
    n_old = len(info.subtitle_tracks)
    tmp = out.with_name(out.name + ".part" + ext)
    cmd = [ff.ffmpeg, "-nostdin", "-v", "error", "-protocol_whitelist", "file,pipe", "-i", str(video), "-i", str(srt),
           "-map", "0", "-map", "1:0", "-c", "copy", "-c:s:%d" % n_old, sub_codec,
           "-metadata:s:s:%d" % n_old, "language=" + lang, "-metadata:s:s:%d" % n_old, "title=" + title,
           "-disposition:s:%d" % n_old, "default", "-y", str(tmp)]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, **_NO_WINDOW)
    if r.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise MediaError("Export failed: " + r.stderr.decode("utf-8", "replace")[-300:])
    try:                                                    # verify before declaring success
        new = ff.probe(tmp)
        if len(new.subtitle_tracks) != n_old + 1 or abs(new.duration - info.duration) > max(1.0, info.duration * 0.01):
            raise MediaError("Exported file failed verification (tracks/duration mismatch)")
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(out)
    return out
