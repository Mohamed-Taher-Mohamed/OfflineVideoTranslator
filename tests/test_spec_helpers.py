"""The packaging helper that makes the frozen Windows program find sentencepiece's private DLLs."""
import glob
import importlib
import os
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packaging"))
from spec_helpers import libs_binaries  # noqa: E402


def make_pkg(site, name, libs=()):
    (site / name).mkdir(parents=True)
    (site / name / "__init__.py").write_text("")
    if libs:
        (site / (name + ".libs")).mkdir()
        for f in libs:
            (site / (name + ".libs") / f).write_bytes(b"dll")


def test_libs_folders_beside_packages_are_collected_to_the_bundle_root(tmp_path, monkeypatch):
    make_pkg(tmp_path, "ovt_fake_a", ["msvcp140-abc.dll", "vcomp140-def.dll"])
    make_pkg(tmp_path, "ovt_fake_b")                                  # no .libs folder
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    got = libs_binaries(["ovt_fake_a", "ovt_fake_b", "ovt_not_installed_at_all"])
    assert sorted(os.path.basename(src) for src, _ in got) == ["msvcp140-abc.dll", "vcomp140-def.dll"]
    assert {dest for _, dest in got} == {"ovt_fake_a.libs"}           # same folder name: the package looks for exactly this


def sentencepiece_site(tmp_path):
    """Windows: the installed package. Elsewhere: unpack the wheel named in OVT_SP_WHEEL (a win_amd64 sentencepiece wheel)."""
    if sys.platform == "win32":
        return None
    wheel = os.environ.get("OVT_SP_WHEEL")
    if not wheel or not os.path.isfile(wheel):
        return False
    zipfile.ZipFile(wheel).extractall(tmp_path)
    return tmp_path


def test_every_private_dll_that_sentencepiece_imports_is_collected(tmp_path, monkeypatch):
    from ovt.winlibs import pe_imports
    site = sentencepiece_site(tmp_path)
    if site is False:
        pytest.skip("set OVT_SP_WHEEL=<sentencepiece win_amd64 wheel> (or run on Windows)")
    if site is not None:
        monkeypatch.syspath_prepend(str(site))
        importlib.invalidate_caches()
        pyd = glob.glob(str(site / "sentencepiece" / "_sentencepiece*.pyd"))[0]
    else:
        import sentencepiece
        pyd = glob.glob(os.path.join(os.path.dirname(sentencepiece.__file__), "_sentencepiece*.pyd"))[0]
    shipped = {os.path.basename(src).lower() for src, dest in libs_binaries(["sentencepiece"]) if dest == "sentencepiece.libs"}
    system = {"kernel32.dll", "bcrypt.dll", "dbghelp.dll", "python312.dll", "python311.dll", "python313.dll",
              "vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"}
    private = [n for n in pe_imports(pyd) if not n.startswith("api-ms-win-") and n not in system]
    assert all(n in shipped for n in private), "imported but not collected: %s" % [n for n in private if n not in shipped]
    if private:
        assert shipped                                                  # a build with private DLLs must ship them


def test_the_spec_uses_the_helper():
    spec = (ROOT / "packaging" / "ovt.spec").read_text(encoding="utf-8")
    import ast
    ast.parse(spec)
    assert "libs_binaries" in spec and "SPECPATH" in spec and "sentencepiece" in spec
