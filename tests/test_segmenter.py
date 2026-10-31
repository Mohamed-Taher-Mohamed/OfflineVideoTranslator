"""One line per utterance: splitting recogniser segments at the pauses between WORDS.
The pause lengths below are REAL measurements from a user's clip (Silero VAD without padding)."""
import pytest

from conftest import FakeASR, FakeMT
from ovt import subtitles as sub
from ovt.asr.base import Segment
from ovt.pipeline import JobOptions, Pipeline
from ovt.segmenter import SplitOptions, resegment


def words(*spans):
    """spans: (start, end, 'text') -> word dicts the way the backend stores them."""
    return [{"s": a, "e": b, "w": (" " if i else " ") + t, "p": 0.9} for i, (a, b, t) in enumerate(spans)]


def seg(ws, text=None):
    return Segment(ws[0]["s"], ws[-1]["e"], text or " ".join(w["w"].strip() for w in ws), 0.8, 0.0, ws)


def test_pause_between_words_starts_a_new_line():
    ws = words((0.0, 0.4, "how"), (0.4, 0.7, "old"), (0.7, 1.0, "are"), (1.0, 1.3, "you"),      # one utterance
               (2.1, 2.5, "eighteen"),                                                              # 0.8 s pause
               (3.2, 3.5, "me"), (3.5, 3.8, "too"))                                                 # 0.7 s pause
    cues = resegment([seg(ws)])
    assert [c.text for c in cues] == ["how old are you", "eighteen", "me too"]
    assert [round(c.start, 2) for c in cues] == [0.0, 2.1, 3.2]
    assert cues[0].words and cues[0].words[0]["w"].strip() == "how"                      # word timing is kept on the cue


def test_continuous_speech_and_short_hesitation_stay_together():
    ws = words((0, 0.3, "well"), (0.35, 0.6, "I"), (0.95, 1.3, "think"), (1.3, 1.6, "so"))   # 0.35 s hesitation < 0.40
    assert [c.text for c in resegment([seg(ws)])] == ["well I think so"]


def test_real_pauses_measured_on_a_real_clip_give_one_line_per_utterance():
    regions = [(0.83, 1.41), (2.14, 3.46), (4.06, 4.74), (6.53, 7.58), (9.22, 9.98),
               (10.69, 11.58), (12.86, 14.46), (16.03, 16.74), (18.08, 21.06), (21.60, 22.14)]     # pauses 0.54 - 1.79 s
    ws = []
    for k, (a, b) in enumerate(regions):
        n = 3
        for i in range(n):
            ws.append({"s": a + i * (b - a) / n, "e": a + (i + 1) * (b - a) / n - 0.01, "w": " w%d_%d" % (k, i), "p": 0.9})
    cues = resegment([Segment(0.83, 22.14, "x", 0.8, 0.0, ws)])        # ONE 21-second recogniser segment
    assert len(cues) == 10
    assert [round(c.start, 2) for c in cues] == [r[0] for r in regions]
    assert all(c.end - c.start < 4 for c in cues)


def test_sentence_end_splits_on_a_shorter_pause():
    ws = words((0, 0.5, "Stop."), (0.8, 1.2, "Go"), (1.2, 1.6, "now"))                              # 0.3 s after a full stop
    assert [c.text for c in resegment([seg(ws)])] == ["Stop.", "Go now"]
    ws = words((0, 0.5, "Stop"), (0.8, 1.2, "Go"))                                                    # same gap, no punctuation
    assert [c.text for c in resegment([seg(ws)])] == ["Stop Go"]


def test_long_runs_are_cut_at_the_biggest_internal_pause_never_dropping_a_word():
    ws = [{"s": i * 0.5, "e": i * 0.5 + 0.45, "w": " w%d" % i, "p": 0.9} for i in range(30)]       # 15 s, tiny gaps
    ws[17]["s"] += 0.30; ws[17]["e"] += 0.30                                                          # the biggest gap of the run
    cues = resegment([seg(ws)], SplitOptions(max_seconds=6.5))
    assert len(cues) >= 3 and all(c.end - c.start <= 7.0 for c in cues)
    assert " ".join(c.text for c in cues).split() == ["w%d" % i for i in range(30)]               # every word kept, in order
    long_text = [{"s": i * 0.1, "e": i * 0.1 + 0.09, "w": " word%02d" % i, "p": 0.9} for i in range(40)]
    cues = resegment([seg(long_text)], SplitOptions(max_chars=84))
    assert all(len(c.text) <= 90 for c in cues) and len(cues) >= 2


def test_segments_without_word_timing_are_untouched_and_cues_never_overlap():
    plain = Segment(0.0, 9.0, "no words here", 0.7, 0.0, None)
    assert resegment([plain])[0] is plain
    ws = words((0, 0.5, "a"), (0.9, 1.3, "b"), (1.4, 1.7, "c"))
    cues = resegment([seg(ws)], SplitOptions(pause=0.2))
    assert all(a.end <= b.start for a, b in zip(cues, cues[1:])) and all(c.end > c.start for c in cues)
    assert abs(cues[-1].end - 1.85) < 1e-9                                                          # last cue keeps a 0.15 s tail


def test_shift_moves_segment_and_word_times_together():
    s = seg(words((1.0, 1.5, "a"), (2.0, 2.5, "b")))
    s.shift(10.0)
    assert (s.start, s.end) == (11.0, 12.5) and [w["s"] for w in s.words] == [11.0, 12.0]


# ---------------- pipeline integration ----------------
class WordASR(FakeASR):
    """Every window holds ONE recogniser segment with three utterances separated by pauses (relative times)."""

    def transcribe(self, wav, language=None, on_segment=None):
        FakeASR.calls += 1
        ws = words((0.2, 0.6, "how"), (0.6, 1.0, "old"), (1.9, 2.3, "eighteen"), (3.1, 3.4, "me"), (3.4, 3.8, "too"))
        s = Segment(0.2, 3.8, "how old eighteen me too", 0.8, 0.0, ws)
        if on_segment:
            on_segment(s)
        return "en", 0.99, [s]


def run(paths, ff, media, tmp_path, **kw):
    o = JobOptions(str(media["multi"]), str(tmp_path / "o"), chunk_seconds=5, asr_model_id="m", source_lang="en",
                   snap_to_silence=False, overwrite=True, **kw)
    return Pipeline(paths, ff, lambda: WordASR(), lambda: FakeMT(), None).run(o)


def test_pipeline_gives_one_timed_line_per_utterance_with_absolute_times(paths, ff, media, tmp_path):
    FakeASR.calls = 0
    r = run(paths, ff, media, tmp_path)                                     # 12 s video, 5 s windows -> starts 0, 5, 10
    cues = sub.load_subtitles(r.outputs["srt"])
    srcs = sub.load_subtitles(r.outputs["srt"])
    assert r.cue_count >= 9 and sub.validate_srt(r.outputs["srt"]) == []
    starts = [round(c.start, 1) for c in cues]
    assert starts[:3] == [0.2, 1.9, 3.1] and 5.2 in starts and 6.9 in starts and 10.2 in starts   # window offset applied to words
    import json
    proj = json.loads(open(r.outputs["project"], encoding="utf-8").read())
    assert [c["source_text"] for c in proj["cues"][:3]] == ["how old", "eighteen", "me too"]


def test_turning_the_option_off_keeps_whisper_segments(paths, ff, media, tmp_path):
    FakeASR.calls = 0
    r = run(paths, ff, media, tmp_path, smart_split=False)
    assert r.cue_count == 3                                                  # one cue per window, as before


def test_changing_the_pause_only_redoes_the_cutting_not_the_recognition(paths, ff, media, tmp_path):
    FakeASR.calls = 0
    a = run(paths, ff, media, tmp_path)
    n = FakeASR.calls
    b = run(paths, ff, media, tmp_path, split_pause=1.2)                     # only the 0.8 s pause (1.0 -> 1.9) is below this
    assert FakeASR.calls == n and b.chunks_resumed == 3                      # saved chunks (with words) were reused
    assert b.cue_count < a.cue_count
    c = run(paths, ff, media, tmp_path, split_pause=0.4)
    assert FakeASR.calls == n and c.cue_count == a.cue_count                 # back to the first result, still no recognition


def test_settings_page_saves_split_options(win):
    from ovt.config import Settings
    sp = win.pages["nav_settings"]
    sp.cb_split.setChecked(False); sp.sp_pause.setValue(0.6)
    sp.save()
    s = Settings.load(win.ctx.paths)
    assert s.smart_split is False and abs(s.split_pause - 0.6) < 1e-9
    o = win.pages["nav_home"].build_options(type("I", (), {"path": "x", "audio_track": 0, "src_lang": "", "source_mode": "asr",
                                                           "subtitle_track": 0, "subtitle_file": "", "range_start": 0.0,
                                                           "range_end": 0.0})())
    assert o.smart_split is False and abs(o.split_pause - 0.6) < 1e-9


from test_gui import win, qapp  # noqa: E402,F401  (fixtures)
