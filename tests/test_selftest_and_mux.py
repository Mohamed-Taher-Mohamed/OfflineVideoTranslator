import subprocess

from conftest import FakeMT
from ovt import subtitles as sub
from ovt.selftest import run_mt_selftest


def test_selftest_reports_unsupported_honestly():
    r = run_mt_selftest(FakeMT(pairs=()), "en")
    assert r["supported"] is False and r["passed"] is False and r["reason"]


def test_selftest_detects_untranslated_output():
    class Echo(FakeMT):
        def translate(self, texts, s, t):
            return list(texts)
    assert run_mt_selftest(Echo(), "en")["passed"] is False
    assert run_mt_selftest(FakeMT(), "en")["passed"] is True


def test_generated_arabic_srt_is_readable_by_ffmpeg_after_muxing(media, tmp_path):
    """Real check of 'opens in a compatible player': mux the SRT into a video with ffmpeg,
    read the subtitle stream back, compare text and timing."""
    cues = [sub.Cue(1.0, 3.0, "مرحبا بالعالم\nهذا سطر ثان"), sub.Cue(4.0, 6.0, "جملة ثالثة، مع علامات؟")]
    srt = sub.write_srt(cues, tmp_path / "فيلم تجريبي.ar.srt")
    out = tmp_path / "muxed.mkv"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(media["multi"]), "-i", str(srt), "-map", "0:v", "-map", "0:a:0",
                    "-map", "1", "-c", "copy", "-c:s", "srt", "-metadata:s:s:0", "language=ara", str(out)], check=True)
    back = tmp_path / "back.srt"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(out), "-map", "0:s:0", str(back)], check=True)
    got = sub.load_subtitles(back)
    assert [c.text for c in got] == [c.text for c in cues]
    assert all(abs(g.start - c.start) < 0.06 for g, c in zip(got, cues))
