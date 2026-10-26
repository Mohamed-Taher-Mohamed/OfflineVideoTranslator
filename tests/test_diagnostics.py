from ovt.config import Paths
from ovt.diagnostics import format_report, run_diagnostics
from ovt.models import ModelManager
from test_models import make_argos, make_asr_dir


def status(checks):
    return {n: ok for n, ok, _ in checks}


def test_diag_reports_missing_models_honestly_then_passes(tmp_path):
    paths = Paths(tmp_path / "h").ensure()
    st = status(run_diagnostics(paths))
    assert st["ffmpeg/ffprobe"] and st["offline guard"] and st["arabic paths"] and st["writable folders"]
    assert st["models installed"] is False
    mm = ModelManager(paths)
    mm.import_model("asr", make_asr_dir(tmp_path / "a"), "w")
    mm.import_model("mt", make_argos(tmp_path / "t.argosmodel"), "m")
    checks = run_diagnostics(paths)
    assert status(checks)["models installed"] is True
    assert format_report(checks).splitlines()[-1] == "RESULT: %d passed, 0 failed" % len(checks)


def test_a_broken_component_is_reported_not_crashing(tmp_path, monkeypatch):
    import ovt.media
    monkeypatch.setattr(ovt.media, "find_binary", lambda *a, **k: (_ for _ in ()).throw(ovt.media.MediaError("ffmpeg not found")))
    st = status(run_diagnostics(Paths(tmp_path / "h").ensure()))
    assert st["ffmpeg/ffprobe"] is False and st["offline guard"] is True
