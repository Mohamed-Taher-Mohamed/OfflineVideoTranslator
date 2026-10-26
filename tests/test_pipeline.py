"""Job logic tests. ASR/MT here are SIMULATED (conftest.Fake*): they prove chunking, resume,
fingerprints, output writing and error handling - NOT recognition/translation quality."""
import hashlib
import threading
from pathlib import Path

import pytest

from conftest import FakeASR, FakeMT
from ovt import subtitles as sub
from ovt.errors import (CancelledError, DiskSpaceError, UnsupportedLanguageError, UnsupportedMediaError)
from ovt.pipeline import JobOptions, Pipeline


@pytest.fixture(autouse=True)
def reset():
    FakeASR.calls = 0
    FakeMT.calls = 0


def mk(paths, ff, asr=None, mt=None, progress=None, cancel=None):
    return Pipeline(paths, ff, lambda: asr or FakeASR(), lambda: mt or FakeMT(), None, progress, cancel)


def opts(media, out, **kw):
    d = dict(input_path=str(media["multi"]), out_dir=str(out), chunk_seconds=5, asr_model_id="m1", write_txt=True,
             write_source_srt=True)
    d.update(kw)
    return JobOptions(**d)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_full_job_outputs_valid_files_and_original_untouched(paths, ff, media, tmp_path):
    before = sha(media["multi"])
    r = mk(paths, ff).run(opts(media, tmp_path / "نتائج"))
    assert r.chunks_total == 3 and FakeASR.calls == 3 and r.source_language == "en"
    srt = Path(r.outputs["srt"])
    assert srt.name == "فيديو اختبار 1.ar.srt"
    assert sub.validate_srt(srt) == []
    cues = sub.load_subtitles(srt)
    assert len(cues) == 3
    assert [round(c.start) for c in cues] == [1, 6, 11]           # 1.0 s into each 5-s window
    assert all("ترجمة" in c.text for c in cues)
    assert Path(r.outputs["txt"]).name.endswith(".transcript.txt")
    assert sub.validate_srt(r.outputs["source_srt"]) == []
    assert Path(r.outputs["project"]).exists()
    assert sha(media["multi"]) == before
    assert not list((paths.temp / "jobs").rglob("*.wav"))          # temp audio cleaned


def test_default_output_next_to_video_does_not_touch_it(paths, ff, media, tmp_path):
    import shutil
    d = tmp_path / "نسخة"; d.mkdir()
    v = d / "my movie.mkv"; shutil.copy(media["multi"], v)
    before = sha(v)
    r = mk(paths, ff).run(JobOptions(str(v), chunk_seconds=20, asr_model_id="m"))
    assert Path(r.outputs["srt"]) == d / "my movie.ar.srt"
    assert sha(v) == before


def test_resume_after_cancel_reuses_finished_chunks(paths, ff, media, tmp_path):
    cancel = threading.Event()

    def prog(p):
        if p.stage == "asr" and "Chunk 1/3 done" in p.message:
            cancel.set()
    with pytest.raises(CancelledError):
        mk(paths, ff, progress=prog, cancel=cancel).run(opts(media, tmp_path / "o"))
    assert FakeASR.calls == 1 and not list((tmp_path / "o").glob("*.srt"))
    r = mk(paths, ff).run(opts(media, tmp_path / "o"))
    assert r.chunks_resumed == 1 and FakeASR.calls == 3          # 1 before + 2 after, not 4
    assert len(sub.load_subtitles(r.outputs["srt"])) == 3


def test_translation_only_change_does_not_rerun_asr(paths, ff, media, tmp_path):
    mk(paths, ff).run(opts(media, tmp_path / "o", overwrite=True))
    asr_calls, mt_calls = FakeASR.calls, FakeMT.calls
    g = tmp_path / "g.json"; g.write_text('{"keep":["sentence"]}', encoding="utf-8")
    r = mk(paths, ff).run(opts(media, tmp_path / "o", overwrite=True, glossary_path=str(g)))
    assert FakeASR.calls == asr_calls and r.chunks_resumed == 3
    assert FakeMT.calls > mt_calls                                # translation redone
    assert "sentence" in Path(r.outputs["srt"]).read_text(encoding="utf-8")
    # same settings again: translation cache is reused as well
    mt2 = FakeMT.calls
    mk(paths, ff).run(opts(media, tmp_path / "o", overwrite=True, glossary_path=str(g)))
    assert FakeMT.calls == mt2


def test_changing_asr_model_invalidates_chunks(paths, ff, media, tmp_path):
    mk(paths, ff).run(opts(media, tmp_path / "o", overwrite=True))
    n = FakeASR.calls
    r = mk(paths, ff).run(opts(media, tmp_path / "o", overwrite=True, asr_model_id="another"))
    assert r.chunks_resumed == 0 and FakeASR.calls == n + 3


def test_silent_file_reports_no_speech_and_writes_nothing(paths, ff, media, tmp_path):
    r = mk(paths, ff, asr=FakeASR(speech=False)).run(JobOptions(str(media["silent"]), str(tmp_path / "o"), asr_model_id="m"))
    assert r.no_speech and r.outputs == {} and r.warnings
    assert not list((tmp_path / "o").glob("*"))


def test_unsupported_language_pair_fails_honestly(paths, ff, media, tmp_path):
    with pytest.raises(UnsupportedLanguageError) as e:
        mk(paths, ff, asr=FakeASR(lang="fr")).run(opts(media, tmp_path / "o"))
    assert "fr" in str(e.value) and "fake translation" in str(e.value)
    assert not list((tmp_path / "o").glob("*.srt"))


def test_unknown_source_language_asks_user(paths, ff, media, tmp_path):
    with pytest.raises(UnsupportedLanguageError):
        mk(paths, ff).run(JobOptions(str(media["multi"]), str(tmp_path / "o"), source_mode="file",
                                     subtitle_file=str(media["srt"]), asr_model_id="m"))


def test_embedded_subtitles_mode_skips_asr(paths, ff, media, tmp_path):
    r = mk(paths, ff).run(opts(media, tmp_path / "o", source_mode="embedded", subtitle_track=0))
    assert FakeASR.calls == 0 and r.source_language == "en"
    cues = sub.load_subtitles(r.outputs["srt"])
    assert len(cues) == 2 and abs(cues[0].start - 1.0) < 0.06


def test_external_subtitle_file_mode(paths, ff, media, tmp_path):
    r = mk(paths, ff).run(opts(media, tmp_path / "o", source_mode="file", subtitle_file=str(media["srt"]), source_lang="en"))
    assert FakeASR.calls == 0 and len(sub.load_subtitles(r.outputs["srt"])) == 2


def test_same_language_is_copied_not_translated(paths, ff, media, tmp_path):
    r = mk(paths, ff).run(opts(media, tmp_path / "o", source_lang="ar"))
    assert FakeMT.calls == 0 and any("without translation" in w for w in r.warnings)


def test_bad_inputs_are_clear_errors(paths, ff, media, tmp_path):
    with pytest.raises(UnsupportedMediaError):
        mk(paths, ff).run(opts(media, tmp_path, input_path=str(media["corrupt"])))
    with pytest.raises(UnsupportedMediaError):
        mk(paths, ff).run(opts(media, tmp_path, audio_track=5))


def test_disk_full_is_detected_before_work(paths, ff, media, tmp_path, monkeypatch):
    import collections, shutil
    U = collections.namedtuple("U", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: U(10**9, 10**9 - 1000, 1000))
    with pytest.raises(DiskSpaceError):
        mk(paths, ff).run(opts(media, tmp_path / "o"))
    assert FakeASR.calls == 0


def test_never_overwrites_previous_results_unless_asked(paths, ff, media, tmp_path):
    r1 = mk(paths, ff).run(opts(media, tmp_path / "o"))
    r2 = mk(paths, ff).run(opts(media, tmp_path / "o"))
    assert r1.outputs["srt"] != r2.outputs["srt"] and Path(r1.outputs["srt"]).exists() and ".1." in r2.outputs["srt"]
    r3 = mk(paths, ff).run(opts(media, tmp_path / "o", overwrite=True))
    assert r3.outputs["srt"] == r1.outputs["srt"]


def test_audio_track_selection_changes_job_identity(paths, ff, media, tmp_path):
    a = mk(paths, ff).run(opts(media, tmp_path / "o", audio_track=0))
    b = mk(paths, ff).run(opts(media, tmp_path / "o", audio_track=1))
    assert a.job_id != b.job_id


def test_progress_is_monotonic_and_ends_at_one(paths, ff, media, tmp_path):
    seen = []
    mk(paths, ff, progress=seen.append).run(opts(media, tmp_path / "o"))
    fr = [p.fraction for p in seen]
    assert fr == sorted(fr) and fr[-1] == 1.0 and {"probe", "asr", "translate", "write", "done"} <= {p.stage for p in seen}
