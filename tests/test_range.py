"""Translate only a time range of a video; timestamps stay absolute so the SRT fits the FULL video."""
import json
from pathlib import Path

import pytest

from conftest import FakeASR, FakeMT
from ovt import subtitles as sub
from ovt.errors import OVTError
from ovt.pipeline import JobOptions, Pipeline
from ovt.preview_logic import parse_clock, range_tag


@pytest.fixture(autouse=True)
def reset():
    FakeASR.calls = 0
    FakeASR.languages = []


def opts(media, out, **kw):
    d = dict(input_path=str(media["multi"]), out_dir=str(out), chunk_seconds=5, asr_model_id="m", source_lang="en",
             snap_to_silence=False, overwrite=True)
    d.update(kw)
    return JobOptions(**d)


def run(paths, ff, o):
    return Pipeline(paths, ff, lambda: FakeASR(), lambda: FakeMT(), None).run(o)


def test_parse_clock_and_tag():
    assert parse_clock("90") == 90 and parse_clock("9:15") == 555 and parse_clock("1:02:03") == 3723
    assert parse_clock("0:09:15.5") == 555.5 and parse_clock("") == 0.0 and parse_clock(" 2:00 ") == 120
    for bad in ("abc", "1:2:3:4", "-5", "1::2", "9:xx"):
        with pytest.raises(ValueError):
            parse_clock(bad)
    assert range_tag(555, 735) == ".t0915-1215" and range_tag(0, 0) == "" and range_tag(60, 0) == ".t0100-end"


def test_plan_chunks_cover_exactly_the_range(ff, media):
    plan = ff.plan_chunks(media["multi"], 0, 12.0, 5.0, snap=False, start=2.0, end=11.0)
    assert plan[0][0] == 2.0 and plan[-1][1] == 11.0 and all(a[1] == b[0] for a, b in zip(plan, plan[1:]))
    assert ff.plan_chunks(media["multi"], 0, 12.0, 5.0, snap=False) == [(0.0, 5.0), (5.0, 10.0), (10.0, 12.0)]   # unchanged default
    assert ff.plan_chunks(media["multi"], 0, 12.0, 5.0, snap=False, start=3.0) == [(3.0, 8.0), (8.0, 12.0)]       # open end


def test_pipeline_processes_only_the_range_with_absolute_times_and_tagged_names(paths, ff, media, tmp_path):
    r = run(paths, ff, opts(media, tmp_path / "o", range_start=2.0, range_end=9.0, write_txt=True, write_source_srt=True))
    assert r.chunks_total == 2 and FakeASR.calls == 2                       # [2,7] and [7,9] - not the whole 12 s
    srt = Path(r.outputs["srt"])
    assert srt.name == "فيديو اختبار 1.t0002-0009.ar.srt"
    cues = sub.load_subtitles(srt)
    assert [round(c.start) for c in cues] == [3, 8]                         # window start + 1.0 s, ABSOLUTE video time
    assert all(2.0 <= c.start < 9.0 for c in cues) and sub.validate_srt(srt) == []
    assert all(".t0002-0009" in Path(v).name for k, v in r.outputs.items())  # txt, source srt and project carry the tag too
    proj = json.loads(Path(r.outputs["project"]).read_text(encoding="utf-8"))
    assert proj["range"] == [2.0, 9.0] and proj["source_path"].endswith("فيديو اختبار 1.mkv")


def test_whole_file_names_and_cache_are_unchanged_but_other_ranges_do_not_mix(paths, ff, media, tmp_path):
    full = run(paths, ff, opts(media, tmp_path / "o"))
    assert Path(full.outputs["srt"]).name == "فيديو اختبار 1.ar.srt"
    n = FakeASR.calls
    assert run(paths, ff, opts(media, tmp_path / "o")).chunks_resumed == 3 and FakeASR.calls == n
    part = run(paths, ff, opts(media, tmp_path / "o", range_start=5.0, range_end=10.0))
    assert part.chunks_resumed == 0 and FakeASR.calls == n + 1                # different range -> its own work, no mixing
    again = run(paths, ff, opts(media, tmp_path / "o", range_start=5.0, range_end=10.0))
    assert again.chunks_resumed == 1 and FakeASR.calls == n + 1               # ...and it is resumable


def test_open_ended_and_invalid_ranges(paths, ff, media, tmp_path):
    r = run(paths, ff, opts(media, tmp_path / "o", range_start=8.0))
    assert Path(r.outputs["srt"]).name.endswith(".t0008-end.ar.srt") and r.chunks_total == 1
    with pytest.raises(OVTError) as e:
        run(paths, ff, opts(media, tmp_path / "o", range_start=11.9))
    assert "after the end" in str(e.value) or "at or after" in str(e.value)
    with pytest.raises(OVTError):
        run(paths, ff, opts(media, tmp_path / "o", range_start=6.0, range_end=6.5))
    assert run(paths, ff, opts(media, tmp_path / "o", range_start=2.0, range_end=999)).chunks_total >= 2          # end is clamped


def test_range_filters_existing_subtitle_files(paths, ff, media, tmp_path):
    srt = tmp_path / "en.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:03,000\nA.\n\n2\n00:00:05,000 --> 00:00:07,000\nB.\n\n3\n00:00:09,000 --> 00:00:11,000\nC.\n", encoding="utf-8")
    o = opts(media, tmp_path / "o", source_mode="file", subtitle_file=str(srt), range_start=4.0, range_end=8.0)
    cues = sub.load_subtitles(run(paths, ff, o).outputs["srt"])
    assert len(cues) == 1 and abs(cues[0].start - 5.0) < 0.01


def test_audio_only_files_are_valid_inputs_and_ranges_work_on_them(paths, ff, tmp_path):
    import subprocess
    for name, args in (("صوت تجريبي.wav", []), ("clip.m4a", ["-c:a", "aac"]), ("clip.mp3", ["-c:a", "libmp3lame"])):
        f = tmp_path / name
        r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=f=300:d=12", *args, str(f)],
                           capture_output=True)
        if r.returncode != 0:                      # e.g. no mp3 encoder in this ffmpeg build: skip that format only
            continue
        info = ff.probe(f)
        assert not info.has_video and len(info.audio_tracks) == 1
        o = JobOptions(input_path=str(f), out_dir=str(tmp_path / "o"), chunk_seconds=5, asr_model_id="m", source_lang="en",
                       snap_to_silence=False, overwrite=True, range_start=2.0, range_end=9.0)
        res = Pipeline(paths, ff, lambda: FakeASR(), lambda: FakeMT(), None).run(o)
        assert Path(res.outputs["srt"]).name == f.stem + ".t0002-0009.ar.srt" and res.cue_count == 2
