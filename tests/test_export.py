import hashlib

import pytest

from ovt import subtitles as sub
from ovt.errors import MediaError
from ovt.export import export_with_subtitles


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_export_mkv_adds_arabic_track_keeps_everything_else(ff, media, tmp_path):
    srt = sub.write_srt([sub.Cue(1, 3, "مرحبا بالعالم"), sub.Cue(4, 6, "سطر ثان")], tmp_path / "x.ar.srt")
    before = sha(media["multi"])
    out = export_with_subtitles(ff, media["multi"], srt, tmp_path / "نتيجة")
    assert out.name == "فيديو اختبار 1.ar.mkv" and sha(media["multi"]) == before
    info = ff.probe(out)
    assert len(info.audio_tracks) == 2 and len(info.subtitle_tracks) == 2 and info.has_video
    assert info.subtitle_tracks[-1].language == "ara"
    back = ff.extract_subtitle(out, 1, tmp_path / "back.srt")
    assert [c.text for c in sub.load_subtitles(back)] == ["مرحبا بالعالم", "سطر ثان"]


def test_export_mp4_uses_mov_text_and_never_overwrites(ff, media, tmp_path):
    import subprocess
    mp4 = tmp_path / "a.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=160x120:r=5:d=4", "-f", "lavfi", "-i",
                    "sine=f=300:d=4", "-c:v", "mpeg4", "-c:a", "aac", str(mp4)], check=True)
    srt = sub.write_srt([sub.Cue(1, 2, "اختبار")], tmp_path / "a.ar.srt")
    o1 = export_with_subtitles(ff, mp4, srt)
    o2 = export_with_subtitles(ff, mp4, srt)
    assert o1.name == "a.ar.mp4" and o2.name == "a.ar.1.mp4" and ff.probe(o1).subtitle_tracks[0].codec == "mov_text"


def test_export_errors_leave_no_partial_files(ff, media, tmp_path):
    with pytest.raises(MediaError):
        export_with_subtitles(ff, media["multi"], tmp_path / "missing.srt", tmp_path / "o")
    bad = tmp_path / "bad.srt"; bad.write_bytes(b"\x00\x00")
    try:
        export_with_subtitles(ff, media["multi"], bad, tmp_path / "o2")
    except Exception:
        pass
    assert not list((tmp_path / "o2").glob("*.part*")) if (tmp_path / "o2").exists() else True
