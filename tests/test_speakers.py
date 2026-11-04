"""Speaker-aware recognition. Turn / VAD data below were MEASURED on a real 26 s two-person clip
(sherpa-onnx titanet-small, threshold 0.8; Silero VAD without padding). The recogniser is scripted (no real Whisper)."""
import io
import json
import os
import tarfile
from pathlib import Path

import numpy as np
import pytest

from conftest import FakeMT
from ovt import subtitles as sub
from ovt.asr.base import ASRBackend, Segment
from ovt.diarize.base import Diarizer, Turn
from ovt.errors import ModelIntegrityError, OVTError
from ovt.pathutil import ascii_path
from ovt.pipeline import JobOptions, Pipeline
from ovt.speakers import (ClipText, SpeakerOptions, assign_words, batch_audio, cues_from_clips, make_clips, pack, SR)

REAL_TURNS = [Turn(a, b, s) for a, b, s in [(0.7, 1.4, 0), (2.8, 3.4, 1), (4.0, 4.7, 0), (6.4, 7.6, 0), (9.1, 9.9, 1),
                                           (10.6, 11.5, 0), (12.8, 14.2, 1), (16.0, 16.7, 0), (17.9, 18.8, 0), (19.1, 21.0, 0),
                                           (21.5, 22.2, 1)]]
REAL_VAD = [(0.83, 1.41), (2.14, 2.50), (2.82, 3.46), (4.06, 4.74), (6.53, 7.58), (9.22, 9.98), (10.69, 11.58),
            (12.86, 14.46), (16.03, 16.74), (18.08, 18.88), (19.20, 21.06), (21.60, 22.14)]


# ---------------- pure functions ----------------
def test_clips_keep_acoustic_edges_and_recover_speech_the_diarizer_missed():
    clips = make_clips(REAL_TURNS, REAL_VAD, 26.05)
    assert len(clips) == 11                                              # 10 merged turns + "18" from the VAD
    missed = [c for c in clips if c.speaker is None]
    assert len(missed) == 1 and abs(missed[0].start - 2.14) < 1e-6 and abs(missed[0].end - 2.50) < 1e-6
    assert [c.speaker for c in clips if c.speaker is not None][:3] == [0, 1, 0]
    for c in clips:
        assert c.a <= c.start < c.end <= c.b
    assert all(x.b <= y.a + 1e-9 for x, y in zip(clips, clips[1:]))      # padding never overlaps the neighbour
    assert any(abs(c.start - 17.9) < 1e-6 and abs(c.end - 21.0) < 1e-6 for c in clips)   # 0.3 s apart, same speaker -> merged


def test_a_long_turn_is_cut_at_its_best_pause_not_in_the_middle_of_a_word():
    vad = [(0, 17.5), (18.7, 40)]                                         # one pause at 17.5-18.7 inside a 40 s turn
    clips = make_clips([Turn(0, 40, 0)], vad, 40.0, SpeakerOptions(max_clip=24.0))
    assert len(clips) == 2 and abs(clips[0].end - 18.1) < 1e-6 and abs(clips[1].start - 18.1) < 1e-6
    assert all(c.end - c.start <= 24.0 for c in clips)
    hard = make_clips([Turn(0, 60, 0)], [(0, 60)], 60.0, SpeakerOptions(max_clip=24.0))     # no pause anywhere
    assert len(hard) >= 3 and all(c.end - c.start <= 24.0 + 1e-6 for c in hard)


def test_packing_respects_the_window_and_builds_real_audio_with_silent_gaps():
    turns = [Turn(i * 6.0, i * 6.0 + 5.0, i % 2) for i in range(10)]
    clips = make_clips(turns, [], 60.0)
    batches = pack(clips)
    assert len(batches) >= 2 and all(b.length <= 28.0 + 1e-9 for b in batches)
    assert sum(len(b.clips) for b in batches) == 10
    samples = (np.arange(60 * SR) % 100 + 1).astype(np.float32) / 200.0           # never zero: easy to tell speech from gaps
    b = batches[0]
    audio = batch_audio(samples, b)
    assert abs(len(audio) / SR - b.length) < 0.05
    off = int(b.offsets[0] * SR)
    c = b.clips[0]
    assert np.array_equal(audio[off:off + 100], samples[int(c.a * SR):int(c.a * SR) + 100])      # the clip itself, untouched
    gap_start = off + int((c.b - c.a) * SR)
    assert not audio[gap_start:gap_start + int(0.7 * SR)].any()                                 # digital silence between clips
    assert not audio[:int(0.25 * SR)].any()


def test_words_go_to_the_clip_they_fall_in_and_gap_hallucinations_are_dropped():
    clips = make_clips([Turn(0, 2, 0), Turn(3, 5, 1)], [], 6.0)
    b = pack(clips)[0]
    o0, o1 = b.offsets
    w = lambda t, txt: {"s": t, "e": t + 0.2, "w": " " + txt, "p": 0.9}
    mid_gap = o0 + (clips[0].b - clips[0].a) + 0.4 - 0.1                          # word centre exactly in the middle of the gap
    words = [w(o0 + 0.3, "hello"), w(o0 + 1.0, "world"), w(mid_gap, "Thanks"), w(o1 + 0.2, "bye")]
    got = assign_words(b, words)
    assert [[x["w"].strip() for x in g] for g in got] == [["hello", "world"], ["bye"]]


def test_cues_follow_speaker_turns_not_whisper_segments():
    c = lambda a, b, spk, txt: ClipText(make_clips([Turn(a, b, spk)], [], 99)[0] if spk is not None else
                                        make_clips([], [(a, b)], 99)[0], txt, 0.9)
    items = [c(0.7, 1.4, 0, "how old are you"), c(2.1, 2.5, None, "18"), c(2.8, 3.4, 1, "me too"),
             c(4.0, 4.7, 0, "how"), c(6.4, 7.6, 0, "old are we really 17"), c(9.1, 9.9, 1, "me too")]
    cues = cues_from_clips(items)
    assert [(x["text"], x["speaker"]) for x in cues] == [("how old are you", 1), ("18", None), ("me too", 2),
                                                          ("how old are we really 17", 1), ("me too", 2)]
    assert abs(cues[3]["start"] - 4.0) < 1e-9 and abs(cues[3]["end"] - 7.6) < 1e-9          # same speaker, 1.7 s pause: ONE cue
    far = cues_from_clips([c(0, 1, 0, "a"), c(5, 6, 0, "b")])                                 # 4 s pause: separate
    assert len(far) == 2
    stop = cues_from_clips([c(0, 1, 0, "Stop."), c(1.8, 2.5, 0, "go")])                       # sentence end + 0.8 s pause
    assert len(stop) == 2
    long = cues_from_clips([c(i * 2, i * 2 + 1.5, 0, "word%d" % i) for i in range(8)])        # would exceed 7 s
    assert len(long) >= 2 and all(x["end"] - x["start"] <= 7.0 for x in long)
    assert cues_from_clips([c(0, 1, 0, "  "), c(2, 3, 1, "ok")])[0]["text"] == "ok"           # empty recognitions vanish


def test_ascii_path_helper(tmp_path):
    plain = tmp_path / "model.onnx"; plain.write_bytes(b"abc")
    assert ascii_path(plain) == str(plain)
    arabic = tmp_path / "برنامج الترجمة" / "نموذج.onnx"
    arabic.parent.mkdir(); arabic.write_bytes(b"0123456789")
    p = ascii_path(arabic)
    assert p.isascii() and Path(p).read_bytes() == b"0123456789" and ascii_path(arabic) == p     # copied once, then cached


# ---------------- pipeline ----------------
class FakeDiarizer(Diarizer):
    calls = 0

    def __init__(self, turns):
        self.turns = turns

    def diarize(self, wav_path, progress=None):
        FakeDiarizer.calls += 1
        if progress:
            progress(1, 2); progress(2, 2)
        return list(self.turns)


class ScriptedASR(ASRBackend):
    """Plays the role of Whisper on the PACKED audio: returns ONE segment holding every clip's words at the times
    where the packed audio really has them, plus a hallucinated word in a silent gap."""
    calls = 0
    vad_filter = True

    def __init__(self, batches, texts):
        self.batches, self.texts, self.n = batches, texts, 0

    def transcribe(self, wav, language=None, on_segment=None):
        ScriptedASR.calls += 1
        b = self.batches[self.n]; self.n += 1
        words = []
        for clip, off, text in zip(b.clips, b.offsets, self.texts[self.n - 1]):
            toks = text.split()
            dur = clip.b - clip.a
            for i, t in enumerate(toks):
                words.append({"s": off + 0.1 + i * (dur - 0.2) / len(toks), "e": off + 0.1 + (i + 0.8) * (dur - 0.2) / len(toks),
                              "w": " " + t, "p": 0.9})
        last = b.offsets[-1] + (b.clips[-1].b - b.clips[-1].a)
        words.append({"s": last + 0.45, "e": last + 0.6, "w": " Thanks", "p": 0.4})              # hallucination in the gap
        seg = Segment(words[0]["s"], words[-1]["e"], " ".join(w["w"].strip() for w in words), 0.8, 0.0, words)
        if on_segment:
            on_segment(seg)
        return "en", 0.99, [seg]


TURNS = [Turn(0.7, 1.4, 0), Turn(2.8, 3.4, 1), Turn(4.0, 4.7, 0), Turn(6.4, 7.6, 0), Turn(9.1, 9.9, 1)]
VAD = [(0.83, 1.41), (2.14, 2.50), (2.82, 3.46), (4.06, 4.74), (6.53, 7.58), (9.22, 9.98)]
TEXTS = [["how old are you", "18", "me too", "how", "old are we really 17", "me too"]]


@pytest.fixture
def setup(paths, ff, media, tmp_path, monkeypatch):
    import ovt.vad
    monkeypatch.setattr(ovt.vad, "speech_regions", lambda samples, ms=300: list(VAD))
    total = ff.probe(media["multi"]).duration
    batches = pack(make_clips(TURNS, VAD, total))
    FakeDiarizer.calls = ScriptedASR.calls = 0

    def run(**kw):
        asr = ScriptedASR(batches, TEXTS)
        pl = Pipeline(paths, ff, lambda: asr, lambda: FakeMT(), None, diarizer_factory=lambda: FakeDiarizer(TURNS))
        o = JobOptions(str(media["multi"]), str(tmp_path / "o"), chunk_seconds=5, asr_model_id="m", source_lang="en",
                       detect_speakers=True, spk_model_id="s", overwrite=True, **{"min_cue_seconds": 0.1, **kw})
        return pl.run(o), asr
    return run


def test_pipeline_one_cue_per_speaker_turn_with_acoustic_timing(setup):
    r, asr = setup()
    assert asr.vad_filter is False                                          # speech already isolated: VAD switched off
    proj = json.loads(Path(r.outputs["project"]).read_text(encoding="utf-8"))
    got = [(c["source_text"], c["speaker"], round(c["start"], 1), round(c["end"], 1)) for c in proj["cues"]]
    assert got == [("how old are you", "1", 0.7, 1.4), ("18", "", 2.1, 2.5), ("me too", "2", 2.8, 3.4),
                   ("how old are we really 17", "1", 4.0, 7.6), ("me too", "2", 9.1, 9.9)]
    assert "Thanks" not in " ".join(c["source_text"] for c in proj["cues"])    # the gap hallucination is gone
    assert sub.validate_srt(r.outputs["srt"]) == [] and r.cue_count == 5 and r.source_language == "en"
    assert ScriptedASR.calls == 1                                           # ONE recogniser pass for the whole clip set


def test_speaker_results_are_cached_and_each_option_redoes_only_what_it_changes(setup):
    r1, _ = setup()
    d, a = FakeDiarizer.calls, ScriptedASR.calls
    r2, _ = setup()
    assert (FakeDiarizer.calls, ScriptedASR.calls) == (d, a) and r2.chunks_resumed == 1          # everything reused
    r3, _ = setup(speaker_labels="label")
    assert (FakeDiarizer.calls, ScriptedASR.calls) == (d, a)                                      # labels: output only
    srt = Path(r3.outputs["srt"]).read_text(encoding="utf-8-sig")
    assert srt.count("المتحدث 1:") == 2 and srt.count("المتحدث 2:") == 2 and "18" in srt
    r4, _ = setup(num_speakers=3)                                                                 # new diarization, same turns
    assert FakeDiarizer.calls == d + 1 and ScriptedASR.calls == a                                 # recognition still reused
    assert r4.cue_count == 5


def test_speaker_mode_needs_a_diarizer(paths, ff, media, tmp_path):
    o = JobOptions(str(media["multi"]), str(tmp_path / "o"), asr_model_id="m", detect_speakers=True)
    with pytest.raises(OVTError) as e:
        Pipeline(paths, ff, lambda: None, lambda: FakeMT(), None).run(o)
    assert "not available" in str(e.value)


def test_translation_units_never_cross_speakers():
    from ovt.translate.service import build_units
    from ovt.subtitles import Cue
    mk = lambda a, b, spk, t: Cue(a, b, "", source_text=t, speaker=spk)
    cues = [mk(0, 1, "1", "how"), mk(2.5, 3.5, "1", "old are we really"), mk(4.0, 5.0, "2", "me too"), mk(5.1, 6, "2", "actually")]
    assert build_units(cues) == [[0, 1], [2, 3]]                                # same speaker joins over a 1.5 s pause; speakers never mix
    plain = [mk(0, 1, "", "how"), mk(2.5, 3.5, "", "old")]
    assert build_units(plain) == [[0], [1]]                                     # no speakers known: the old 0.3 s rule


# ---------------- models + download ----------------
def make_spk_dir(d):
    d.mkdir(parents=True, exist_ok=True)
    (d / "segmentation.onnx").write_bytes(b"seg"); (d / "embedding.onnx").write_bytes(b"emb")
    return d


def test_speaker_package_import_validates_and_lists(paths, tmp_path):
    from ovt.models import ModelManager
    mm = ModelManager(paths)
    m = mm.import_model("spk", make_spk_dir(tmp_path / "pk"), "spk-test")
    assert set(m["files"]) >= {"segmentation.onnx", "embedding.onnx"} and [x["id"] for x in mm.list("spk")] == ["spk-test"]
    assert mm.verify("spk", "spk-test") == [] and "spk-test" in [x["id"] for x in mm.list()]
    bad = tmp_path / "bad"; bad.mkdir(); (bad / "embedding.onnx").write_bytes(b"x")
    with pytest.raises(ModelIntegrityError) as e:
        mm.import_model("spk", bad, "spk-bad")
    assert "segmentation.onnx" in str(e.value)


def test_download_speaker_package_from_a_local_server(paths, tmp_path, monkeypatch):
    import http.server, threading
    from ovt import downloader, netguard
    from ovt.models import ModelManager
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tf:
        for name, data in (("pyannote/model.int8.onnx", b"int8"), ("pyannote/model.onnx", b"SEGMENTATION"), ("../evil.txt", b"x")):
            ti = tarfile.TarInfo(name); ti.size = len(data); tf.addfile(ti, io.BytesIO(data))
    files = {"/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2": buf.getvalue(),
             "/speaker-recongition-models/nemo_en_titanet_small.onnx": b"EMBEDDING"}

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def do_GET(self):
            body = files.get(self.path)
            if body is None:
                self.send_error(404); return
            self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(downloader, "SPK_BASE_URL", "http://127.0.0.1:%d" % srv.server_address[1])
    netguard.enable_offline(paths.logs / "n.log")
    mm = ModelManager(paths)
    m = downloader.download_speaker_package(paths, mm, "titanet-small")
    d = Path(m["path"])
    assert (d / "segmentation.onnx").read_bytes() == b"SEGMENTATION" and (d / "embedding.onnx").read_bytes() == b"EMBEDDING"
    assert m["id"] == "spk-titanet-small" and not (paths.temp / "spk-titanet-small").exists() and netguard.is_offline()
    assert not (paths.models.parent / "evil.txt").exists() and not (tmp_path / "evil.txt").exists()
    files.pop("/speaker-recongition-models/nemo_en_titanet_small.onnx")
    with pytest.raises(OVTError):
        downloader.download_speaker_package(paths, mm, "titanet-small")                       # 404 -> clear error
    with pytest.raises(OVTError):
        downloader.download_speaker_package(paths, mm, "no-such-model")
    srv.shutdown()


# ---------------- GUI ----------------
from test_gui import win, qapp, wait_for  # noqa: E402,F401  (fixtures)


def test_settings_and_preflight_for_speakers(win, tmp_path):
    from ovt.config import Settings
    sp = win.pages["nav_settings"]
    assert sp.cb_spkmodel.count() == 0
    sp.cb_spk.setChecked(True); sp.cb_nspk.setCurrentIndex(sp.cb_nspk.findData(2))
    sp.cb_spklabel.setCurrentIndex(sp.cb_spklabel.findData("label"))
    sp.save()
    s = Settings.load(win.ctx.paths)
    assert s.detect_speakers and s.num_speakers == 2 and s.speaker_labels == "label"
    home = win.pages["nav_home"]
    item = type("I", (), {"path": "x", "audio_track": 0, "src_lang": "", "source_mode": "asr", "subtitle_track": 0,
                          "subtitle_file": "", "range_start": 0.0, "range_end": 0.0,
                          "info": type("M", (), {"duration": 26.0})()})()
    assert home.preflight(item) == home.ctx.t("no_spk_model")                   # feature on, model missing: clear message
    win.ctx.models.import_model("spk", make_spk_dir(tmp_path / "pk"), "spk-x")
    win.ctx.settings.spk_model_id = "spk-x"
    assert home.preflight(item) == ""
    o = home.build_options(item)
    assert o.detect_speakers and o.num_speakers == 2 and o.spk_model_id == "spk-x" and o.speaker_labels == "label"


# ---------------- REAL models (optional): OVT_SPK_DIR=<dir with the two models>  OVT_SPK_AUDIO=<16 kHz wav> ----------------
@pytest.mark.skipif(not (os.environ.get("OVT_SPK_DIR") and os.environ.get("OVT_SPK_AUDIO")),
                    reason="set OVT_SPK_DIR and OVT_SPK_AUDIO to run against real sherpa-onnx models")
def test_real_diarizer_on_a_real_two_person_clip():
    from ovt.diarize.sherpa_backend import SherpaDiarizer
    d = Path(os.environ["OVT_SPK_DIR"])
    seg = next(d.rglob("model.onnx"))
    turns = SherpaDiarizer(seg, d / "titanet_small.onnx", 0, 0.8).diarize(os.environ["OVT_SPK_AUDIO"])
    assert len({t.speaker for t in turns}) == 2 and turns[0].speaker == 0
    true_bursts = [(0.83, 1.41), (2.82, 3.46), (4.06, 4.74), (6.53, 7.58), (9.22, 9.98), (10.69, 11.58), (12.86, 14.46),
                   (16.03, 16.74), (18.08, 18.88), (19.20, 21.06), (21.60, 22.14)]           # ("18" is too short for the model)
    for a, b in true_bursts:
        cover = sum(max(0, min(b, t.end) - max(a, t.start)) for t in turns)
        assert cover >= 0.6 * (b - a), "speech burst %.2f-%.2f not covered by a speaker turn" % (a, b)


def test_default_minimum_cue_length_still_never_overlaps_the_next_speaker(setup):
    """With the normal 0.8 s minimum, short cues are lengthened only up to the next cue's start."""
    r, _ = setup(min_cue_seconds=0.8)
    cues = sub.load_subtitles(r.outputs["srt"])
    assert all(a.end <= b.start for a, b in zip(cues, cues[1:])) and all(c.end - c.start >= 0.3 for c in cues)
    assert cues[0].end - cues[0].start >= 0.79                                  # 0.7 s turn lengthened to the minimum
