import pytest
from ovt import subtitles as s


def cues():
    return [s.Cue(1.0, 2.0, "مرحبا بالعالم"), s.Cue(2.5, 4.0, "سطر ثان"), s.Cue(5.0, 6.5, "ثالث")]


def test_timestamps_roundtrip():
    for v in (0, 0.001, 61.5, 3661.999, 86399.5):
        assert abs(s.parse_ts(s.fmt_ts(v)) - v) < 0.0006
    assert s.fmt_ts(3661.5) == "01:01:01,500"


def test_write_valid_srt_utf8_arabic(tmp_path):
    p = s.write_srt(cues(), tmp_path / "ملف عربي.ar.srt")
    assert s.validate_srt(p) == []
    assert "مرحبا بالعالم" in p.read_text(encoding="utf-8")
    back = s.load_subtitles(p)
    assert [c.text for c in back] == [c.text for c in cues()]


def test_bom_option_and_rtl_marks(tmp_path):
    p = s.write_srt(cues(), tmp_path / "a.srt", bom=True, rtl_marks=True)
    raw = p.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert s.RLM in raw.decode("utf-8-sig")
    assert s.validate_srt(p) == []


def test_validator_catches_problems(tmp_path):
    bad = tmp_path / "bad.srt"
    bad.write_text("1\n00:00:05,000 --> 00:00:04,000\nx\n\n3\n00:00:01,000 --> 00:00:02,000\ny\n", encoding="utf-8")
    probs = " | ".join(s.validate_srt(bad))
    assert "end <= start" in probs and "index" in probs and "out of order" in probs
    notutf = tmp_path / "l1.srt"
    notutf.write_bytes("1\n00:00:01,000 --> 00:00:02,000\nمرحبا\n".encode("cp1256"))
    assert s.validate_srt(notutf) == ["file is not valid UTF-8"]
    e = tmp_path / "e.srt"
    e.write_text("", encoding="utf-8")
    assert s.validate_srt(e)


def test_reflow_never_drops_text():
    long = " ".join("كلمة%d" % i for i in range(60))
    out = s.reflow([s.Cue(0, 12, long)], max_chars=30, max_lines=2)
    assert len(out) > 1
    assert all(len(l) <= 30 for c in out for l in c.text.split("\n"))
    assert all(len(c.text.split("\n")) <= 2 for c in out)
    assert " ".join(c.text.replace("\n", " ") for c in out).split() == long.split()
    assert out[0].start == 0 and abs(out[-1].end - 12) < 1e-9
    assert all(a.end <= b.start + 1e-9 for a, b in zip(out, out[1:]))


def test_fix_timing_removes_overlap_and_min_duration():
    c = [s.Cue(0, 5, "a"), s.Cue(1, 2, "b"), s.Cue(1.0, 1.1, "c")]
    out = s.fix_timing(c, min_dur=0.5)
    for a, b in zip(out, out[1:]):
        assert a.end <= b.start
    assert all(x.end > x.start for x in out)


def test_shift_and_reset():
    c = cues()
    s.shift(c, 2.0)
    assert c[0].start == 3.0
    s.shift(c, -10.0)
    assert c[0].start == 0.0 and c[0].end > 0
    s.reset_timing(c)
    assert (c[0].start, c[0].end) == (1.0, 2.0)


def test_parse_vtt_ass_and_legacy_encoding(tmp_path):
    vtt = tmp_path / "a.vtt"
    vtt.write_text("WEBVTT\n\n00:01.000 --> 00:02.500\n<i>Hi</i> there\n", encoding="utf-8")
    assert s.load_subtitles(vtt)[0].text == "Hi there"
    ass = tmp_path / "a.ass"
    ass.write_text("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
                   "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{\\i1}Hello, world{\\i0}\\Nline2\n", encoding="utf-8")
    c = s.load_subtitles(ass)
    assert c[0].text == "Hello, world\nline2" and c[0].end == 2.0
    legacy = tmp_path / "ar.srt"
    legacy.write_bytes("1\n00:00:01,000 --> 00:00:02,000\nمرحبا بكم\n".encode("cp1256"))
    assert s.load_subtitles(legacy)[0].text == "مرحبا بكم"
    with pytest.raises(s.SubtitleFormatError):
        s.load_subtitles(tmp_path / "x.doc")
