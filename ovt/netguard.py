"""Offline enforcement + network-attempt audit log.

enable_offline() patches the socket layer of THIS process: any non-loopback
connection attempt raises OfflineViolation and is written to logs/network.log.
Child processes (ffmpeg) are separate programs; they are only ever given local
file paths (see media.py) and the packaged ffmpeg is invoked with
`-protocol_whitelist file,pipe` so it cannot open network URLs from a hostile file.
"""
import datetime
import ipaddress
import os
import socket
import threading

from .errors import OfflineViolation

_lock = threading.Lock()
_state = {"enabled": False, "log": None, "orig": None, "attempts": []}


def _is_local(host) -> bool:
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "ignore")
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _record(kind, target):
    line = "%s BLOCKED %s %r" % (datetime.datetime.now().isoformat(timespec="seconds"), kind, target)
    with _lock:
        _state["attempts"].append(line)
        if _state["log"]:
            try:
                os.makedirs(os.path.dirname(_state["log"]), exist_ok=True)
                with open(_state["log"], "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                pass


def enable_offline(log_path=None) -> None:
    if _state["enabled"]:
        return
    _state["log"] = str(log_path) if log_path else None
    orig = {
        "connect": socket.socket.connect,
        "connect_ex": socket.socket.connect_ex,
        "getaddrinfo": socket.getaddrinfo,
        "create_connection": socket.create_connection,
    }
    _state["orig"] = orig

    def guard_connect(self, address, *a, **k):
        host = address[0] if isinstance(address, tuple) else address
        if self.family == getattr(socket, "AF_UNIX", -1) or _is_local(host):
            return orig["connect"](self, address, *a, **k)
        _record("connect", address)
        raise OfflineViolation("Offline mode: network connection blocked (%s)" % (host,))

    def guard_connect_ex(self, address, *a, **k):
        host = address[0] if isinstance(address, tuple) else address
        if self.family == getattr(socket, "AF_UNIX", -1) or _is_local(host):
            return orig["connect_ex"](self, address, *a, **k)
        _record("connect_ex", address)
        raise OfflineViolation("Offline mode: network connection blocked (%s)" % (host,))

    def guard_gai(host, *a, **k):
        if _is_local(host):
            return orig["getaddrinfo"](host, *a, **k)
        _record("dns", host)
        raise OfflineViolation("Offline mode: DNS lookup blocked (%s)" % (host,))

    def guard_cc(address, *a, **k):
        if _is_local(address[0]):
            return orig["create_connection"](address, *a, **k)
        _record("create_connection", address)
        raise OfflineViolation("Offline mode: network connection blocked (%s)" % (address[0],))

    socket.socket.connect = guard_connect
    socket.socket.connect_ex = guard_connect_ex
    socket.getaddrinfo = guard_gai
    socket.create_connection = guard_cc
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[k] = "1"
    _state["enabled"] = True


def disable_offline() -> None:
    """Only for explicit model-download commands and tests."""
    if not _state["enabled"]:
        return
    o = _state["orig"]
    socket.socket.connect = o["connect"]
    socket.socket.connect_ex = o["connect_ex"]
    socket.getaddrinfo = o["getaddrinfo"]
    socket.create_connection = o["create_connection"]
    _state["enabled"] = False


def is_offline() -> bool:
    return _state["enabled"]


def attempts():
    return list(_state["attempts"])


class temporarily_online:
    """Explicit, user-approved window of network access (model downloads only).
    Offline mode is restored in `finally` even if the download fails. The window is written to the network log."""

    def __init__(self, reason: str):
        self.reason, self.was = reason, False

    def __enter__(self):
        self.was = _state["enabled"]
        if self.was:
            _log_event("ALLOWED (user-approved window) " + self.reason)
            disable_offline()
        return self

    def __exit__(self, *exc):
        if self.was:
            enable_offline(_state["log"])
            _log_event("RESTORED offline mode")
        return False


def _log_event(line: str):
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    with _lock:
        _state["attempts"].append("%s %s" % (stamp, line))
        if _state["log"]:
            try:
                with open(_state["log"], "a", encoding="utf-8") as f:
                    f.write("%s %s\n" % (stamp, line))
            except OSError:
                pass
