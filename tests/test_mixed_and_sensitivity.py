"""Background / mixed-language speech: what is configurable and verifiable without a real model.
These tests prove the options REACH the recogniser and the pipeline; they cannot prove how much extra speech
a real Whisper model recovers (that needs a real model + your video)."""
import pytest

from conftest import FakeASR, FakeMT
from ovt.pipeline import JobOptions, Pipeline
from test_llm_backend import FakeLLM, server  # noqa: F401  (server fixture)
from ovt.translate.llm_backend import LLMTranslator, build_system_prompt


@pytest.fixture(autouse=True)
def reset():
    FakeASR.calls = 0
    FakeASR.languages = []
    FakeMT.calls = 0


def opts(media, out, **kw):
    d = dict(input_path=str(media["multi"]), out_dir=str(out), chunk_seconds=5, asr_model_id="m", snap_to_silence=False,
             overwrite=True)
    d.update(kw)
    return JobOptions(**d)


def run(paths, ff, o, mt=None):
    return Pipeline(paths, ff, lambda: FakeASR(), lambda: mt or FakeMT(), None).run(o)


def test_backend_options_reach_faster_whisper(tmp_path, monkeypatch):
    import faster_whisper
    from ovt.asr.faster_whisper_backend import FasterWhisperBackend
    from test_asr_audio_and_hub import make_wav
    seen = []

    class FakeModel:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, audio, **k):
            seen.append(k)
            return iter([]), type("I", (), {"language": "en", "language_probability": 0.9})()
    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)
    wav = make_wav(tmp_path / "a.wav")
    FasterWhisperBackend(str(tmp_path)).transcribe(wav)
    FasterWhisperBackend(str(tmp_path), sensitivity="high").transcribe(wav)
    FasterWhisperBackend(str(tmp_path), multilingual=True).transcribe(wav)
    FasterWhisperBackend(str(tmp_path), multilingual=True).transcribe(wav, language="en")
    normal, high, mixed, mixed_forced = seen
    assert "vad_parameters" not in normal and "multilingual" not in normal and normal["vad_filter"] is True
    assert high["vad_parameters"]["threshold"] < 0.5 and high["no_speech_threshold"] > 0.6 and high["log_prob_threshold"] < -1.0
    assert mixed["multilingual"] is True and mixed["language"] is None
    assert "multilingual" not in mixed_forced and mixed_forced["language"] == "en"     # an explicit language wins


def test_normal_mode_fixes_the_language_after_the_first_chunk_mixed_mode_does_not(paths, ff, media, tmp_path):
    run(paths, ff, opts(media, tmp_path / "o"))
    assert FakeASR.languages == [None, "en", "en"]              # detected once, then forced on every other chunk
    FakeASR.languages = []
    run(paths, ff, opts(media, tmp_path / "o", mixed_languages=True))
    assert FakeASR.languages == [None, None, None]              # every chunk (window) decides for itself


def test_changing_sensitivity_or_mixed_mode_redoes_recognition(paths, ff, media, tmp_path):
    run(paths, ff, opts(media, tmp_path / "o"))
    n = FakeASR.calls
    assert run(paths, ff, opts(media, tmp_path / "o")).chunks_resumed == 3 and FakeASR.calls == n      # same -> reused
    assert run(paths, ff, opts(media, tmp_path / "o", speech_sensitivity="high")).chunks_resumed == 0
    assert run(paths, ff, opts(media, tmp_path / "o", mixed_languages=True)).chunks_resumed == 0


def test_mixed_languages_with_a_non_mixed_engine_warns_and_uses_dominant_language(paths, ff, media, tmp_path):
    r = run(paths, ff, opts(media, tmp_path / "o", mixed_languages=True))
    assert any("Mixed languages" in w and "LLM" in w for w in r.warnings) and r.source_language == "en"


def test_mixed_languages_with_llm_sends_a_language_agnostic_prompt(paths, ff, media, tmp_path, server):
    FakeLLM.requests = []
    r = run(paths, ff, opts(media, tmp_path / "o", mixed_languages=True, translator_tag="llm:test"),
            mt=LLMTranslator(server, "qwen2.5:7b"))
    assert not any("Mixed languages" in w for w in r.warnings)
    system = FakeLLM.requests[0]["body"]["messages"][0]["content"]
    assert "DIFFER from line to line" in system and "already Arabic" in system
    assert "numbered lines (English)" in build_system_prompt("en") and "numbered lines (English)" not in build_system_prompt("mixed")
    assert "ع:" in open(r.outputs["srt"], encoding="utf-8").read()


def test_settings_page_saves_the_new_options(win):
    from ovt.config import Settings
    sp = win.pages["nav_settings"]
    sp.cb_sens.setCurrentIndex(sp.cb_sens.findData("high")); sp.cb_mixed.setChecked(True)
    sp.save()
    s = Settings.load(win.ctx.paths)
    assert s.speech_sensitivity == "high" and s.mixed_languages is True
    home = win.pages["nav_home"]
    o = home.build_options(type("I", (), {"path": "x", "audio_track": 0, "src_lang": "", "source_mode": "asr",
                                          "subtitle_track": 0, "subtitle_file": "", "range_start": 0.0,
                                          "range_end": 0.0})())
    assert o.speech_sensitivity == "high" and o.mixed_languages is True


from test_gui import win, qapp  # noqa: E402,F401  (fixtures)
