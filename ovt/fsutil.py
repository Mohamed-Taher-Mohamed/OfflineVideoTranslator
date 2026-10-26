"""Small filesystem helpers: atomic writes, disk checks, safe names."""
import json
import os
import shutil
import tempfile
from pathlib import Path

from .errors import DiskSpaceError


def atomic_write_bytes(path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except OSError as e:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        if getattr(e, "errno", None) == 28:  # ENOSPC
            raise DiskSpaceError("Disk is full while writing %s" % path, path=str(path)) from e
        raise


def atomic_write_text(path, text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2))


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def ensure_free_space(path, needed_bytes: int, what: str = "operation") -> None:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    free = shutil.disk_usage(str(p)).free
    if free < needed_bytes:
        raise DiskSpaceError(
            "Not enough free disk space for %s: need %d MB, have %d MB"
            % (what, needed_bytes // 2**20, free // 2**20),
            needed=needed_bytes, free=free)


def unique_path(path) -> Path:
    path = Path(path)
    if not path.exists():
        return path
    for i in range(1, 1000):
        cand = path.with_name("%s.%d%s" % (path.stem, i, path.suffix))
        if not cand.exists():
            return cand
    raise OSError("cannot find a free file name for %s" % path)
