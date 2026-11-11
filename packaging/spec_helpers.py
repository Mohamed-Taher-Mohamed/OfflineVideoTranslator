"""Helpers used by ovt.spec (kept in a normal module so they can be unit-tested).

Windows wheels built with *delvewheel* (sentencepiece 0.2.3, ...) keep their private runtime DLLs in a folder NEXT TO the
package, `<package>.libs/`, and the package's __init__ adds `<package dir>/../<package>.libs` to the DLL search path.
PyInstaller only collects what is INSIDE the package, so without this the frozen program fails with
'DLL load failed while importing _sentencepiece'. Each such folder must be copied to the bundle root with the same name."""
import importlib.util
from pathlib import Path
from typing import Iterable, List, Tuple


def libs_binaries(packages: Iterable[str]) -> List[Tuple[str, str]]:
    """[(source file, destination folder)] for every `<package>.libs/` sitting beside an installed package."""
    out: List[Tuple[str, str]] = []
    for name in packages:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            continue
        if spec is None or not spec.origin or spec.origin in ("built-in", "frozen"):
            continue
        origin = Path(spec.origin)
        site = origin.parent.parent if spec.submodule_search_locations else origin.parent   # package: .../site/pkg/__init__.py
        libs = site / (name + ".libs")
        if libs.is_dir():
            for f in sorted(libs.iterdir()):
                if f.is_file():
                    out.append((str(f), libs.name))
    return out
