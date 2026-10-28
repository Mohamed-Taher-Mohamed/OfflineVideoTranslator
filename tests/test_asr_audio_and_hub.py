"""Regressions for two real failures seen on the first Windows run."""
import os
import subprocess
import wave

import numpy as np
import pytest

from ovt import netguard
from ovt.asr.audio_io import read_wav_float32
from ovt.errors import MediaError


def make_wav(path, seconds=2, rate=16000, channels=1):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=f=300:d=%d" % seconds,
                    "-ar", str(rate), "-ac", str(channels), str(path)], check=True)
    return path


def test_wav_reader_gives_normalised_float32(tmp_path):
    a = read_wav_float32(make_wav(tmp_path / "ملف.wav"))
    assert a.dtype == np.float32 and a.shape == (32000,) and 0.05 < float(np.abs(a).max()) <= 1.0


def test_wav_reader_rejects_wrong_format(tmp_path):
    with pytest.raises(MediaError):
        read_wav_float32(make_wav(tmp_path / "st.wav", channels=2))
    with pytest.raises(MediaError):
        read_wav_float32(make_wav(tmp_path / "r.wav", rate=44100))


def test_backend_hands_faster_whisper_an_array_not_a_path(tmp_path, monkeypatch):
    """faster-whisper's own PyAV decoder (broken with av>=17/19) must never be reached."""
    import faster_whisper
    seen = {}

    class FakeModel:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, audio, **k):
            seen["audio"] = audio
            info = type("I", (), {"language": "en", "language_probability": 0.9})()
            seg = type("S", (), {"start": 0.0, "end": 1.0, "text": " hello ", "avg_logprob": -0.1, "no_speech_prob": 0.01})()
            return iter([seg]), info
    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)
    from ovt.asr.faster_whisper_backend import FasterWhisperBackend
    lang, prob, segs = FasterWhisperBackend(str(tmp_path)).transcribe(make_wav(tmp_path / "a.wav"))
    assert isinstance(seen["audio"], np.ndarray) and seen["audio"].dtype == np.float32
    assert lang == "en" and segs[0].text == "hello"


def test_hub_offline_flags_follow_the_guard_and_online_window(tmp_path, monkeypatch):
    """Real failure: the download button failed with 'outgoing traffic has been disabled' because
    HF_HUB_OFFLINE stayed 1 inside the approved online window."""
    import huggingface_hub.constants as C
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    netguard.enable_offline(tmp_path / "n.log")
    assert os.environ["HF_HUB_OFFLINE"] == "1" and C.HF_HUB_OFFLINE is True
    with netguard.temporarily_online("test"):
        assert "HF_HUB_OFFLINE" not in os.environ or os.environ["HF_HUB_OFFLINE"] != "1"
        assert C.HF_HUB_OFFLINE is False
    assert os.environ["HF_HUB_OFFLINE"] == "1" and C.HF_HUB_OFFLINE is True       # restored afterwards
    netguard.disable_offline()                                                      # user turned offline mode off
    assert C.HF_HUB_OFFLINE is False and os.environ.get("HF_HUB_OFFLINE") != "1"


def test_backend_streams_segments_to_callback_and_uses_configured_beam(tmp_path, monkeypatch):
    import faster_whisper
    seen = {}

    class FakeModel:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, audio, **k):
            seen.update(k)
            info = type("I", (), {"language": "en", "language_probability": 0.9})()
            segs = [type("S", (), {"start": float(i), "end": i + 0.5, "text": " t%d " % i, "avg_logprob": -0.2, "no_speech_prob": 0.0})()
                    for i in range(3)]
            return (s for s in segs), info                     # a generator, like the real library
    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)
    from ovt.asr.faster_whisper_backend import FasterWhisperBackend
    got = []
    _, _, segs = FasterWhisperBackend(str(tmp_path), beam_size=1).transcribe(make_wav(tmp_path / "b.wav"), on_segment=got.append)
    assert seen["beam_size"] == 1 and [s.text for s in got] == ["t0", "t1", "t2"] and len(segs) == 3
