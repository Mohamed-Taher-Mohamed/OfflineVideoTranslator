"""DLL helpers for the packaged Windows program. The PE reader is checked against `pefile` on a REAL Windows extension
module: on Windows CI the installed sentencepiece .pyd; elsewhere set OVT_PYD_SAMPLE=<some Windows .pyd/.dll>."""
import glob
import os
import sys

import pytest

from ovt import winlibs


def sample_pe():
    env = os.environ.get("OVT_PYD_SAMPLE")
    if env and os.path.isfile(env):
        return env
    if sys.platform == "win32":
        import sentencepiece
        found = glob.glob(os.path.join(os.path.dirname(sentencepiece.__file__), "_sentencepiece*.pyd"))
        return found[0] if found else None
    return None


needs_pe = pytest.mark.skipif(sample_pe() is None, reason="no Windows PE file available (set OVT_PYD_SAMPLE)")


@needs_pe
def test_pe_import_reader_agrees_with_pefile():
    pefile = pytest.importorskip("pefile")
    path = sample_pe()
    expected = sorted({e.dll.decode().lower() for e in pefile.PE(path).DIRECTORY_ENTRY_IMPORT})
    got = winlibs.pe_imports(path)
    assert sorted(set(got)) == expected and "msvcp140.dll" in got and "vcruntime140.dll" in got


@needs_pe
def test_failure_explanation_names_the_dll_that_cannot_be_loaded():
    exc = ImportError("DLL load failed while importing _sentencepiece: The specified module could not be found.")
    msg = winlibs.explain_dll_failure(sample_pe(), exc, probe=lambda names: {n: n != "msvcp140.dll" for n in names})
    assert "msvcp140.dll" in msg and "cannot be loaded" in msg and "DLL load failed" in msg
    assert "bcrypt.dll" not in msg.split("cannot be loaded:")[1]
    ok = winlibs.explain_dll_failure(sample_pe(), exc, probe=lambda names: {n: True for n in names})
    assert "search-path or version conflict" in ok                               # nothing missing -> says what that implies


def test_a_file_that_is_not_a_pe_never_hides_the_original_error(tmp_path):
    bad = tmp_path / "x.pyd"; bad.write_bytes(b"not a pe file at all" * 10)
    msg = winlibs.explain_dll_failure(bad, ImportError("original problem"))
    assert msg.startswith("original problem") and "could not analyse" in msg


def test_dll_folders_of_the_bundle_are_added_to_the_search_path(tmp_path, monkeypatch):
    (tmp_path / "ctranslate2").mkdir(); (tmp_path / "ctranslate2" / "a.dll").write_bytes(b"x")
    (tmp_path / "onnxruntime" / "capi").mkdir(parents=True); (tmp_path / "onnxruntime" / "capi" / "b.pyd").write_bytes(b"x")
    (tmp_path / "docs").mkdir(); (tmp_path / "docs" / "readme.txt").write_text("x")
    (tmp_path / "a" / "b" / "c").mkdir(parents=True); (tmp_path / "a" / "b" / "c" / "deep.dll").write_bytes(b"x")
    added = []
    monkeypatch.setattr(os, "add_dll_directory", lambda d: added.append(d), raising=False)
    got = winlibs.prepare_dll_search(str(tmp_path))
    rel = sorted(os.path.relpath(d, tmp_path) for d in got)
    assert rel == [".", os.path.join("ctranslate2"), os.path.join("onnxruntime", "capi")]      # no folders without DLLs / too deep
    assert added == got
    monkeypatch.delattr(os, "add_dll_directory", raising=False)
    assert winlibs.prepare_dll_search(str(tmp_path)) == []                                      # platforms without the API: no-op


def test_probe_on_non_windows_never_fails():
    if sys.platform != "win32":
        assert winlibs.probe_dlls(["whatever.dll"]) == {"whatever.dll": True}
