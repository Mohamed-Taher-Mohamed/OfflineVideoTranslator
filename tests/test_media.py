import wave

import pytest

from ovt.errors import MediaError, NoAudioError, UnsupportedMediaError
from ovt import subtitles as sub


def test_probe_arabic_path_with_spaces_two_tracks_and_sub(ff, media):
    info = ff.probe(media["multi"])
    assert info.name == "فيديو اختبار 1.mkv"
    assert abs(info.duration - 12) < 0.6
    assert info.has_video
    assert [t.index for t in info.audio_tracks] == [0, 1]
    assert [t.language for t in info.audio_tracks] == ["eng", "fra"]
    assert len(info.subtitle_tracks) == 1 and info.subtitle_tracks[0].is_text


def test_extract_wav_format_window_and_tracks_differ(ff, media, tmp_path):
    a = ff.extract_wav(media["multi"], 0, 2.0, 4.0, tmp_path / "ص1.wav")
    b = ff.extract_wav(media["multi"], 1, 2.0, 4.0, tmp_path / "t2.wav")
    with wave.open(str(a)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        assert abs(w.getnframes() / 16000 - 4.0) < 0.1
    assert a.read_bytes()[44:] != b.read_bytes()[44:]          # 440 Hz vs 880 Hz tracks
    assert -40 < ff.mean_volume_db(a) < 0                       # audible tone


def test_original_never_modified(ff, media, tmp_path):
    import hashlib
    h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    before = h(media["multi"])
    ff.probe(media["multi"]); ff.extract_wav(media["multi"], 0, 0, 3, tmp_path / "x.wav")
    assert h(media["multi"]) == before


def test_silent_file_has_very_low_volume(ff, media, tmp_path):
    w = ff.extract_wav(media["silent"], 0, 0, 5, tmp_path / "s.wav")
    v = ff.mean_volume_db(w)
    assert v is None or v < -80


def test_errors_are_clear(ff, media, tmp_path):
    with pytest.raises(UnsupportedMediaError):
        ff.probe(media["corrupt"])
    with pytest.raises(UnsupportedMediaError):
        ff.probe(media["empty"])
    with pytest.raises(NoAudioError):
        ff.probe(media["noaudio"])
    with pytest.raises(MediaError):
        ff.probe(tmp_path / "missing.mp4")
    with pytest.raises(MediaError):
        ff.extract_wav(media["multi"], 7, 0, 2, tmp_path / "bad.wav")   # no such track


def test_extract_embedded_subtitle(ff, media, tmp_path):
    f = ff.extract_subtitle(media["multi"], 0, tmp_path / "e.srt")
    cues = sub.load_subtitles(f)
    assert [c.text for c in cues] == ["Hello world.", "Second line here."]
    assert abs(cues[0].start - 1.0) < 0.05   # mkv muxing shifts ~23 ms


@pytest.mark.slow
def test_three_hour_file_streams_without_loading_into_memory(ff, tmp_path):
    import resource, subprocess, time
    big = tmp_path / "long 3h.mkv"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x48:r=1:d=10800",
                    "-f", "lavfi", "-i", "sine=f=300:d=10800", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", "-b:a", "16k", str(big)], check=True)
    info = ff.probe(big)
    assert abs(info.duration - 10800) < 5
    t = time.time()
    w = ff.extract_wav(big, 0, 9000, 300, tmp_path / "w.wav")
    elapsed = time.time() - t
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print("3h file: %.1f MB on disk, window extract %.1fs, our RSS %.0f MB" % (big.stat().st_size / 1e6, elapsed, rss_mb))
    assert w.stat().st_size > 9_000_000
    assert rss_mb < 400
