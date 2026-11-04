"""Native libraries (SentencePiece, sherpa-onnx, ...) often open files with the ANSI code page on Windows, so a path with
Arabic letters fails with 'No such file or directory'. ascii_path() hands them a plain-ASCII copy when needed."""
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

from .errors import OVTError


def _is_ascii(p) -> bool:
    try:
        str(p).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _cache_roots():
    yield Path(tempfile.gettempdir())
    for env in ("PUBLIC", "ProgramData", "ALLUSERSPROFILE"):
        v = os.environ.get(env)
        if v:
            yield Path(v)
    if sys.platform != "win32":
        yield Path("/tmp")


def ascii_path(path) -> str:
    """Return `path` itself when it is pure ASCII, else a cached ASCII-path copy of the FILE."""
    p = Path(path)
    if _is_ascii(p):
        return str(p)
    st = p.stat()
    key = hashlib.sha1(("%s|%d|%d" % (p.resolve(), st.st_size, st.st_mtime_ns)).encode("utf-8")).hexdigest()[:16]
    for root in _cache_roots():
        dest = root / "ovt_ascii" / key / ("model" + (p.suffix if _is_ascii(p.suffix) else ""))
        if not _is_ascii(dest):
            continue
        try:
            if not dest.exists() or dest.stat().st_size != st.st_size:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, dest)
            return str(dest)
        except OSError:
            continue
    raise OVTError("Cannot read '%s': its path has non-English characters and no ASCII-only temporary folder is "
                   "available. Move the program folder to a path like C:\\OVT." % p)
