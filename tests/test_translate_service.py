import pytest

from conftest import FakeMT
from ovt.glossary import Glossary
from ovt.subtitles import Cue
from ovt.translate import service as svc
from ovt.translate.base import Support, Translator


def mk(*texts, gap=0.2):
    out, t = [], 0.0
    for x in texts:
        out.append(Cue(t, t + 1.5, "", source_text=x)); t += 1.5 + gap
    return out


def test_units_merge_unfinished_sentences_only():
    c = mk("I think that we should", "go home now.", "Then eat.")
    assert svc.build_units(c) == [[0, 1], [2]]
    assert svc.build_units(c, context_merge=False) == [[0], [1], [2]]
    far = mk("Hello and", "world."); far[1].start += 5; far[1].end += 5
    assert svc.build_units(far) == [[0], [1]]


def test_split_over_cues_keeps_every_word():
    t = "واحد اثنان ثلاثة اربعة خمسة ستة سبعة"
    for w in ([5, 5], [1, 9], [3, 3, 3], [1, 1, 1, 1, 1, 1, 1, 1]):
        parts = svc.split_over_cues(t, w)
        assert len(parts) == len(w)
        assert " ".join(parts).split() == t.split()


def test_all_cues_get_text_and_count_preserved():
    c = mk("Hello there.", "How are you today?", "Fine thanks.")
    warn = svc.translate_cues(FakeMT(), c, "en", "ar")
    assert all(x.text.startswith("ترجمة:") or x.text for x in c)
    assert warn == [] or all("untranslated" not in w for w in warn)
    assert len([x for x in c if x.text]) == 3


def test_glossary_keep_and_map_roundtrip():
    g = Glossary(["Hogwarts"], {"Dr.": "د."})
    c = mk("Dr. Smith went to Hogwarts.")
    svc.translate_cues(FakeMT(), c, "en", "ar", g)
    assert "Hogwarts" in c[0].text and "د." in c[0].text and "XQ" not in c[0].text


def test_engine_that_mangles_placeholder_is_reported_not_hidden():
    class Mangler(FakeMT):
        def translate(self, texts, s, t):
            return [x.replace("XQ1Z", "") for x in texts]
    c = mk("Welcome to Hogwarts.")
    warn = svc.translate_cues(Mangler(), c, "en", "ar", Glossary(["Hogwarts"]))
    assert any("glossary term lost" in w for w in warn)


def test_wrong_result_count_is_an_error():
    class Bad(FakeMT):
        def translate(self, texts, s, t):
            return texts[:-1]
    with pytest.raises(RuntimeError):
        svc.translate_cues(Bad(), mk("a b c.", "d e f."), "en", "ar", batch_size=5, context_merge=False)


def test_untranslated_and_empty_output_flagged():
    class Echo(FakeMT):
        def translate(self, texts, s, t):
            return list(texts)
    c = mk("This sentence stayed in English.")
    assert any("no Arabic letters" in w for w in svc.translate_cues(Echo(), c, "en", "ar"))

    class Empty(FakeMT):
        def translate(self, texts, s, t):
            return [""] * len(texts)
    c = mk("Something real to say here.")
    warn = svc.translate_cues(Empty(), c, "en", "ar")
    assert any("empty translation" in w for w in warn) and c[0].text == "Something real to say here."


def test_cache_skips_finished_units():
    FakeMT.calls = 0
    c = mk("One.", "Two.", "Three.")
    cache = {}
    svc.translate_cues(FakeMT(), c, "en", "ar", cache=cache, batch_size=2)
    first = FakeMT.calls
    assert first == 2 and len(cache) == 3
    svc.translate_cues(FakeMT(), mk("One.", "Two.", "Three."), "en", "ar", cache=cache, batch_size=2)
    assert FakeMT.calls == first


def test_repetition_loops_are_collapsed_and_reported_but_normal_text_is_kept():
    from ovt.translate.service import collapse_loops
    assert collapse_loops("هذا هذا هذا هذا هذا هذا جيد")[0] == "هذا جيد"
    assert collapse_loops("pppppppppppppppppp")[0] == "p"
    for ok in ("very very good", "لا لا لا", "No, no, no, no!", "aaa bbb", "كلمة مكررة مرتين مرتين"):
        assert collapse_loops(ok) == (ok, False), ok

    class Looper(FakeMT):
        def translate(self, texts, s, t):
            return ["مرحبا " + "كلمة " * 30 for _ in texts]
    c = mk("Something to say.")
    warn = svc.translate_cues(Looper(), c, "en", "ar")
    assert c[0].text == "مرحبا كلمة" and any("repetition loop" in w for w in warn)
