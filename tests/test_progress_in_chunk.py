"""Regression for 'progress stuck at 6% forever': one long chunk used to report nothing until it finished,
and cancel was only checked between chunks."""
import threading
import time

import pytest

from conftest import FakeMT
from ovt.asr.base import ASRBackend, Segment
from ovt.errors import CancelledError
from ovt.pipeline import JobOptions, Pipeline


class ManySegmentsASR(ASRBackend):
    """Simulated slow recogniser: 10 segments inside ONE chunk, produced one by one."""
    produced = 0

    def transcribe(self, wav, language=None, on_segment=None):
        out = []
        for i in range(10):
            time.sleep(0.03)
            seg = Segment(i * 1.0 + 0.2, i * 1.0 + 0.9, "sentence %d" % i, 0.9)
            out.append(seg)
            ManySegmentsASR.produced += 1
            if on_segment:
                on_segment(seg)
        return "en", 0.99, out


def opts(media, out, **kw):
    d = dict(input_path=str(media["multi"]), out_dir=str(out), chunk_seconds=3600, asr_model_id="m", source_lang="en",
             snap_to_silence=False)
    d.update(kw)
    return JobOptions(**d)


def test_progress_moves_inside_a_single_chunk_and_reports_speed_and_text(paths, ff, media, tmp_path):
    ManySegmentsASR.produced = 0
    seen = []
    r = Pipeline(paths, ff, lambda: ManySegmentsASR(), lambda: FakeMT(), None, seen.append).run(opts(media, tmp_path / "o"))
    assert r.chunks_total == 1
    asr = [p for p in seen if p.stage == "asr" and "Recognising" in p.message]
    fracs = [p.fraction for p in asr]
    assert len(asr) >= 10 and fracs == sorted(fracs) and len(set(round(f, 3) for f in fracs)) >= 8   # many distinct steps
    assert 0.0 < fracs[0] < fracs[-1] <= 0.7
    assert all(p.speed and p.speed > 0 for p in asr[1:])
    assert any("sentence 3" in p.detail for p in asr)                                          # live text per segment


def test_cancel_stops_in_the_middle_of_a_chunk(paths, ff, media, tmp_path):
    ManySegmentsASR.produced = 0
    cancel = threading.Event()
    count = {"n": 0}

    def prog(p):
        if p.stage == "asr" and "Recognising" in p.message:
            count["n"] += 1
            if count["n"] == 4:
                cancel.set()
    with pytest.raises(CancelledError):
        Pipeline(paths, ff, lambda: ManySegmentsASR(), lambda: FakeMT(), None, prog, cancel).run(opts(media, tmp_path / "o"))
    assert ManySegmentsASR.produced < 10                      # stopped before the chunk finished
    ManySegmentsASR.produced = 0                              # nothing half-saved: a re-run simply redoes the chunk
    r = Pipeline(paths, ff, lambda: ManySegmentsASR(), lambda: FakeMT(), None).run(opts(media, tmp_path / "o"))
    assert ManySegmentsASR.produced == 10 and r.chunks_resumed == 0 and r.cue_count > 0
