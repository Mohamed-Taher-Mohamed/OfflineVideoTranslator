"""Real CTranslate2 + SentencePiece execution with a tiny random model (gibberish output on purpose)."""
import zipfile

import pytest

from ovt.translate.ct2_backend import CT2Translator, safe_extract, split_sentences
from tinymodel import make_argos_package, make_nllb_dir


def test_argos_package_loads_and_runs_end_to_end(tmp_path):
    pkg = make_argos_package(tmp_path / "en_ar.argosmodel", tmp_path / "w1")
    t = CT2Translator([("argos", pkg)], tmp_path / "ex")
    assert not t.load_errors
    s = t.supports("en", "ar")
    assert s.ok and s.route == "direct"
    out = t.translate(["hello world this is a test.", "how are you today?", ""], "en", "ar")
    assert len(out) == 3 and out[2] == ""
    assert all(isinstance(x, str) for x in out)         # content is gibberish by design


def test_package_extracts_once_and_not_for_other_pairs(tmp_path):
    pkg = make_argos_package(tmp_path / "en_ar.argosmodel", tmp_path / "w1")
    t = CT2Translator([("argos", pkg)], tmp_path / "ex")
    assert not t.supports("fr", "ar").ok and "fr" in t.supports("fr", "ar").reason
    assert not t.supports("ar", "en").ok
    CT2Translator([("argos", pkg)], tmp_path / "ex")          # second load reuses extraction
    assert (tmp_path / "ex" / "en_ar" / ".extracted").exists()


def test_pivot_through_english_is_reported(tmp_path):
    a = make_argos_package(tmp_path / "fr_en.argosmodel", tmp_path / "w1", "fr", "en")
    b = make_argos_package(tmp_path / "en_ar.argosmodel", tmp_path / "w2", "en", "ar")
    t = CT2Translator([("argos", a), ("argos", b)], tmp_path / "ex")
    s = t.supports("fr", "ar")
    assert s.ok and "pivot" in s.route
    assert len(t.translate(["hello world."], "fr", "ar")) == 1


def test_nllb_folder_direct_pair_with_language_tokens(tmp_path):
    d = make_nllb_dir(tmp_path / "nllb")
    t = CT2Translator([("nllb", d)], tmp_path / "ex")
    assert not t.load_errors
    assert t.supports("en", "ar").ok and "NLLB" in t.supports("en", "ar").route
    assert t.supports("fr", "ar").ok
    assert not t.supports("xx", "ar").ok                     # unknown language code
    out = t.translate(["hello world.", "how are you today? fine."], "en", "ar")
    assert len(out) == 2


def test_broken_package_is_reported_not_hidden(tmp_path):
    bad = tmp_path / "bad.argosmodel"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("x/readme.txt", "no metadata")
    t = CT2Translator([("argos", bad)], tmp_path / "ex")
    s = t.supports("en", "ar")
    assert not s.ok and "bad.argosmodel" in s.reason


def test_zip_slip_blocked(tmp_path):
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as z:
        z.writestr("../../escape.txt", "x")
    with pytest.raises(ValueError):
        safe_extract(evil, tmp_path / "dest")
    assert not (tmp_path.parent / "escape.txt").exists()


def test_sentence_splitter():
    assert split_sentences("Hi there. How are you? Fine!") == ["Hi there.", "How are you?", "Fine!"]
    long = "word " * 400
    assert all(len(p) <= 520 for p in split_sentences(long))
    assert " ".join(" ".join(split_sentences(long)).split()) == " ".join(long.split())


def test_runaway_output_is_bounded(tmp_path):
    """The random tiny model loops forever; output must stay proportional to the input."""
    pkg = make_argos_package(tmp_path / "en_ar.argosmodel", tmp_path / "w1")
    t = CT2Translator([("argos", pkg)], tmp_path / "ex")
    short = t.translate(["hello."], "en", "ar")[0]
    sp = t.argos[("en", "ar")].sp
    n_src = len(sp.encode("hello.", out_type=str))
    assert len(sp.encode(short, out_type=str)) <= 3 * n_src + 10 + 2
    assert t.argos[("en", "ar")].decode_limit([[1] * 500]) == 256


def test_sentencepiece_model_loads_even_when_the_library_cannot_open_non_ascii_paths(tmp_path, monkeypatch):
    """Real Windows failure: sentencepiece's C++ opens `model_file` with the ANSI code page, so a path containing
    Arabic letters gave 'NOT_FOUND ... No such file or directory Error #2'. The engine must pass bytes instead."""
    import sentencepiece
    folder = tmp_path / "برنامج الترجمة" / "models"
    folder.mkdir(parents=True)
    pkg = make_argos_package(folder / "en_ar.argosmodel", tmp_path / "w1")      # built BEFORE the simulation is switched on
    nllb_dir = make_nllb_dir(tmp_path / "نموذج عربي")

    class WindowsLikeSP(sentencepiece.SentencePieceProcessor):
        def __init__(self, *a, model_file=None, **k):
            if model_file is not None:
                raise OSError('NOT_FOUND: "%s": No such file or directory Error #2' % model_file)
            super().__init__(*a, **k)
    monkeypatch.setattr(sentencepiece, "SentencePieceProcessor", WindowsLikeSP)
    t = CT2Translator([("argos", pkg)], folder / "mt-extracted")
    assert not t.load_errors, t.load_errors
    assert t.supports("en", "ar").ok and len(t.translate(["hello world."], "en", "ar")) == 1
    nllb = CT2Translator([("nllb", nllb_dir)], folder / "ex2")
    assert not nllb.load_errors, nllb.load_errors


def test_extracted_cache_follows_the_package_when_it_is_replaced(tmp_path):
    """The unzipped copy in mt-extracted/ is a cache. Replacing the package under the same name must NOT keep serving
    the old unzipped model."""
    import shutil
    ex = tmp_path / "ex"
    a = make_argos_package(tmp_path / "pk" / "x.argosmodel", tmp_path / "w1", "en", "ar") if (tmp_path / "pk").mkdir() is None else None
    t1 = CT2Translator([("argos", a)], ex)
    assert t1.supports("en", "ar").ok and not t1.supports("fr", "ar").ok
    b = make_argos_package(tmp_path / "pk2" / "x.argosmodel", tmp_path / "w2", "fr", "ar") if (tmp_path / "pk2").mkdir() is None else None
    shutil.copy(b, a)                                           # same file name, different content
    t2 = CT2Translator([("argos", a)], ex)
    assert t2.supports("fr", "ar").ok and not t2.supports("en", "ar").ok, "stale extraction was reused"


def test_deleting_or_reimporting_a_translation_model_clears_its_extracted_cache(tmp_path):
    from ovt.config import Paths
    from ovt.models import ModelManager
    from test_models import make_argos
    paths = Paths(tmp_path / "h").ensure()
    mm = ModelManager(paths)
    mm.import_model("mt", make_argos(tmp_path / "a.argosmodel"), "argos-en-ar")
    cache = paths.models / "mt-extracted" / "argos-en-ar"
    cache.mkdir(parents=True); (cache / ".extracted").write_text("old")
    mm.import_model("mt", make_argos(tmp_path / "a.argosmodel", "fr", "ar"), "argos-en-ar")     # re-import under the same id
    assert not cache.exists()
    cache.mkdir(parents=True)
    mm.delete("mt", "argos-en-ar")
    assert not cache.exists() and not (paths.models / "mt" / "argos-en-ar").exists()
