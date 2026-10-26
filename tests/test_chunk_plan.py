import subprocess

import pytest

from conftest import FakeASR, FakeMT
from ovt.pipeline import JobOptions, Pipeline


@pytest.fixture(scope="module")
def gapped(tmp_path_factory):
    """12 s: loud noise-like tone, with real silences at 6.1-6.9 s (near the 5 s cut) and a short 8.0-8.1 blip."""
    d = tmp_path_factory.mktemp("gap")
    f = d / "gap.mkv"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=f=440:d=12", "-af",
                    "volume=enable='between(t,6.1,6.9)':volume=0,volume=enable='between(t,8.0,8.1)':volume=0",
                    "-c:a", "aac", str(f)], check=True)
    return f


def test_silence_detection_finds_the_gap(ff, gapped):
    gaps = ff.find_silences(gapped, 0, 0, 12)
    assert any(abs(a - 6.1) < 0.15 and abs(b - 6.9) < 0.15 for a, b in gaps)


def test_boundary_snaps_to_silence_and_covers_whole_file(ff, gapped):
    plan = ff.plan_chunks(gapped, 0, 12.0, 5.0)
    assert abs(plan[0][1] - 6.5) < 0.2                                 # nominal 5.0 moved into the 6.1-6.9 gap
    assert plan[0][0] == 0.0 and plan[-1][1] == 12.0
    assert all(abs(a[1] - b[0]) < 1e-9 for a, b in zip(plan, plan[1:])) and all(e > s for s, e in plan)
    fixed = ff.plan_chunks(gapped, 0, 12.0, 5.0, snap=False)
    assert fixed == [(0.0, 5.0), (5.0, 10.0), (10.0, 12.0)]


def test_no_silence_keeps_nominal_cuts(ff, media):
    plan = ff.plan_chunks(media["multi"], 0, 12.0, 5.0)                # constant tone: nothing to snap to
    assert [round(e, 1) for _, e in plan] == [5.0, 10.0, 12.0]


def test_pipeline_uses_plan_and_resume_keeps_same_borders(paths, ff, gapped, tmp_path):
    FakeASR.calls = 0
    o = JobOptions(str(gapped), str(tmp_path / "o"), chunk_seconds=5, asr_model_id="m", source_lang="en", overwrite=True)
    r = Pipeline(paths, ff, lambda: FakeASR(), lambda: FakeMT(), None).run(o)
    assert r.chunks_total == 2 and FakeASR.calls == 2        # 0-6.5 and 6.5-12 (a 3rd cut would leave <1 s)
    import json
    plan = json.loads(next((paths.temp / "jobs").rglob("plan.json")).read_text())["plan"]
    assert abs(plan[0][1] - 6.5) < 0.2
    r2 = Pipeline(paths, ff, lambda: FakeASR(), lambda: FakeMT(), None).run(o)
    assert r2.chunks_resumed == 2 and FakeASR.calls == 2              # identical plan -> everything reused
    o.snap_to_silence = False                                         # different plan -> redone, not mixed
    r3 = Pipeline(paths, ff, lambda: FakeASR(), lambda: FakeMT(), None).run(o)
    assert r3.chunks_resumed == 0
