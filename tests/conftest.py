import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ovt import netguard  # noqa: E402
from ovt.asr.base import ASRBackend, Segment  # noqa: E402
from ovt.config import Paths  # noqa: E402
from ovt.media import FFmpeg  # noqa: E402
from ovt.translate.base import Support, Translator  # noqa: E402


def sh(*a):
    r = subprocess.run(list(a), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if r.returncode != 0:                      # show ffmpeg's own message, not just an exit code
        raise RuntimeError("command failed (%s): %s\n%s" % (r.returncode, " ".join(map(str, a))[:200],
                                                           r.stderr.decode("utf-8", "replace")[-600:]))


@pytest.fixture(scope="session")
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp("media")
    out = {}
    srt = d / "en.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:03,000\nHello world.\n\n2\n00:00:04,000 --> 00:00:06,000\nSecond line here.\n", encoding="utf-8")
    # Arabic name + spaces, 2 audio tracks (440 Hz / 880 Hz), embedded text subtitle, 12 s
    ar = d / "مجلد عربي" / "فيديو اختبار 1.mkv"
    ar.parent.mkdir()
    sh("ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=160x120:r=5:d=12",
       "-f", "lavfi", "-i", "sine=f=440:d=12", "-f", "lavfi", "-i", "sine=f=880:d=12", "-i", str(srt),
       "-map", "0", "-map", "1", "-map", "2", "-map", "3", "-c:v", "mpeg4", "-c:a", "aac", "-c:s", "srt",
       "-metadata:s:a:0", "language=eng", "-metadata:s:a:1", "language=fra", "-metadata:s:s:0", "language=eng", str(ar))
    out["multi"] = ar
    sil = d / "silent.mp4"
    sh("ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=160x120:r=5:d=6", "-f", "lavfi",
       "-i", "anullsrc=r=16000:cl=mono", "-t", "6", "-c:v", "mpeg4", "-c:a", "aac", str(sil))
    out["silent"] = sil
    noaud = d / "noaudio.mp4"
    sh("ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=160x120:r=5:d=3", "-c:v", "mpeg4", str(noaud))
    out["noaudio"] = noaud
    bad = d / "corrupt.mp4"
    bad.write_bytes(b"\x00\x01notavideo" * 500)
    out["corrupt"] = bad
    empty = d / "empty.mp4"
    empty.write_bytes(b"")
    out["empty"] = empty
    out["srt"] = srt
    return out


@pytest.fixture
def paths(tmp_path):
    return Paths(tmp_path / "home").ensure()


@pytest.fixture
def ff():
    return FFmpeg()


@pytest.fixture(autouse=True)
def _restore_network():
    yield
    netguard.disable_offline()


# ---- SIMULATED backends: used ONLY to test job logic. They are not real ASR/MT. ----
class FakeASR(ASRBackend):
    name = "FAKE"
    calls = 0

    def __init__(self, lang="en", speech=True):
        self.lang, self.speech = lang, speech

    def transcribe(self, wav, language=None):
        FakeASR.calls += 1
        if not self.speech:
            return self.lang, 0.5, []
        # one segment at 1.0-2.5 s of every window, text carries a counter so order is checkable
        return language or self.lang, 0.99, [Segment(1.0, 2.5, "This is sentence number %d." % FakeASR.calls, 0.9)]


class FakeMT(Translator):
    name = "FAKE"
    calls = 0

    def __init__(self, pairs=(("en", "ar"),), drop_glossary=False):
        self.pairs, self.drop = set(pairs), drop_glossary

    def supports(self, s, t):
        return Support((s, t) in self.pairs, "direct", "pair not installed")

    def translate(self, texts, s, t):
        FakeMT.calls += 1
        return ["ترجمة: " + x for x in texts]
