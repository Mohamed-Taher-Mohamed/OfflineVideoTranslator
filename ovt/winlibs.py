"""Windows DLL helpers for the packaged (PyInstaller) program.

* pe_imports(): which DLLs a .pyd/.dll needs (pure-Python PE import-table reader, no dependency).
* probe_dlls(): can each of them be loaded from THIS process (same search rules as the real import)?
* explain_dll_failure(): turns Python's useless "DLL load failed ... module could not be found" into the DLL's NAME.
* prepare_dll_search(): in the frozen app, make every folder of the bundle that holds DLLs searchable, so a package
  that resets the search path cannot hide the DLLs of its neighbours."""
import os
import struct
import sys
from typing import Callable, Dict, List, Optional


def pe_imports(path) -> List[str]:
    d = open(path, "rb").read()
    pe = struct.unpack_from("<I", d, 0x3C)[0]
    if d[pe:pe + 4] != b"PE\0\0":
        raise ValueError("not a PE file")
    n_sec, = struct.unpack_from("<H", d, pe + 6)
    opt_size, = struct.unpack_from("<H", d, pe + 20)
    opt = pe + 24
    magic, = struct.unpack_from("<H", d, opt)
    dd = opt + (112 if magic == 0x20B else 96)               # data directories (PE32+ / PE32)
    rva, size = struct.unpack_from("<II", d, dd + 8)         # entry 1 = import table
    if not rva:
        return []
    secs = []
    for i in range(n_sec):
        o = opt + opt_size + 40 * i
        vs, va, rs, rp = struct.unpack_from("<IIII", d, o + 8)
        secs.append((va, max(vs, rs), rp))

    def off(r):
        for va, ln, rp in secs:
            if va <= r < va + ln:
                return rp + (r - va)
        raise ValueError("bad RVA")
    names, p = [], off(rva)
    while True:
        orig, _, _, name_rva, first = struct.unpack_from("<IIIII", d, p)
        if not (orig or name_rva or first):
            break
        s = off(name_rva)
        names.append(d[s:d.index(b"\0", s)].decode("ascii", "replace").lower())
        p += 20
    return names


def probe_dlls(names) -> Dict[str, bool]:
    """name -> loadable? (Windows only; elsewhere everything counts as loadable)."""
    if sys.platform != "win32":
        return {n: True for n in names}
    import ctypes
    out = {}
    for n in names:
        try:
            ctypes.WinDLL(n)
            out[n] = True
        except OSError:
            out[n] = False
    return out


def explain_dll_failure(pyd_path, exc, probe: Callable = probe_dlls) -> str:
    """'<original error>; cannot be loaded: msvcp140.dll, ...' (names the DLL(s) the import table asks for and the process cannot load)."""
    try:
        names = [n for n in pe_imports(pyd_path) if not n.startswith("api-ms-win-")]
        bad = [n for n, ok in probe(names).items() if not ok]
    except Exception as e:                                   # the diagnosis must never hide the original error
        return "%s (could not analyse %s: %s)" % (exc, os.path.basename(str(pyd_path)), e)
    if bad:
        return "%s; DLL(s) it needs that cannot be loaded: %s" % (exc, ", ".join(bad))
    return "%s; all %d imported DLLs load fine on their own (so the cause is a search-path or version conflict)" % (exc, len(names))


def prepare_dll_search(base: Optional[str] = None) -> List[str]:
    """Add the bundle root and every sub-folder (2 levels) that contains .dll files to the DLL search path."""
    add = getattr(os, "add_dll_directory", None)
    base = base or getattr(sys, "_MEIPASS", None)
    if add is None or not base or not os.path.isdir(base):
        return []
    dirs = [base]
    for root, subdirs, files in os.walk(base):
        depth = os.path.relpath(root, base).count(os.sep) + (0 if root == base else 1)
        if depth >= 2:
            subdirs[:] = []
        if root != base and any(f.lower().endswith((".dll", ".pyd")) for f in files):
            dirs.append(root)
    added = []
    for d in dirs:
        try:
            add(d)
            added.append(d)
        except OSError:
            pass
    return added
