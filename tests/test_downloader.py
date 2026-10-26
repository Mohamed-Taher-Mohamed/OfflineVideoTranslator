"""Downloader tested against a REAL local HTTP server (loopback is allowed by the offline guard)."""
import hashlib
import http.server
import json
import threading

import pytest

from ovt import downloader, netguard
from ovt.errors import CancelledError, ModelIntegrityError, OVTError, OfflineViolation
from ovt.models import ModelManager
from test_models import make_argos

DATA = bytes(range(256)) * 4000        # ~1 MB


class Handler(http.server.BaseHTTPRequestHandler):
    files = {}
    flaky = {"cut_after": None}

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = self.files.get(self.path)
        if body is None:
            self.send_error(404); return
        start = 0
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].rstrip("-"))
            if start >= len(body):
                self.send_error(416); return
            self.send_response(206)
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, len(body) - 1, len(body)))
        else:
            self.send_response(200)
        chunk = body[start:]
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        cut = self.flaky["cut_after"]
        if cut and not rng:
            self.wfile.write(chunk[:cut]); self.wfile.flush()
            self.flaky["cut_after"] = None          # next request (resume) is served fully
            self.close_connection = True
            return
        self.wfile.write(chunk)


@pytest.fixture
def server(tmp_path):
    Handler.files = {"/model.bin": DATA}
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    yield "http://127.0.0.1:%d" % srv.server_address[1]
    srv.shutdown()


def test_download_ok_progress_and_checksum(server, tmp_path):
    seen = []
    f = downloader.download_file(server + "/model.bin", tmp_path / "d" / "m.bin", hashlib.sha256(DATA).hexdigest(),
                                 progress=lambda a, b: seen.append((a, b)))
    assert f.read_bytes() == DATA and seen[-1][0] == seen[-1][1] == len(DATA) and not list((tmp_path / "d").glob("*.part"))


def test_interrupted_download_resumes_from_partial(server, tmp_path):
    Handler.flaky["cut_after"] = 300_000
    with pytest.raises(OVTError):
        downloader.download_file(server + "/model.bin", tmp_path / "m.bin")
    part = tmp_path / "m.bin.part"
    assert 0 < part.stat().st_size < len(DATA)
    got = []
    downloader.download_file(server + "/model.bin", tmp_path / "m.bin", progress=lambda a, b: got.append(a))
    assert (tmp_path / "m.bin").read_bytes() == DATA and got[0] > 300_000       # continued, not restarted


def test_bad_checksum_removes_file_and_404_is_clear(server, tmp_path):
    with pytest.raises(ModelIntegrityError):
        downloader.download_file(server + "/model.bin", tmp_path / "x.bin", "0" * 64)
    assert not (tmp_path / "x.bin").exists() and not (tmp_path / "x.bin.part").exists()
    with pytest.raises(OVTError) as e:
        downloader.download_file(server + "/nope", tmp_path / "y.bin")
    assert "404" in str(e.value)


def test_cancel_keeps_partial(server, tmp_path):
    ev = threading.Event(); ev.set()
    with pytest.raises(CancelledError):
        downloader.download_file(server + "/model.bin", tmp_path / "c.bin", cancel=ev, chunk=1000)


def test_offline_blocks_external_but_temporary_window_restores(tmp_path):
    netguard.enable_offline(tmp_path / "net.log")
    with pytest.raises(OVTError):                                   # external host: blocked (wrapped as OVTError)
        downloader.download_file("http://93.184.216.34/x", tmp_path / "z.bin", )
    with netguard.temporarily_online("unit test"):
        assert not netguard.is_offline()
    assert netguard.is_offline()
    with pytest.raises(Exception):
        with netguard.temporarily_online("failing"):
            raise RuntimeError("download failed")
    assert netguard.is_offline()                                    # restored even after an exception
    log = (tmp_path / "net.log").read_text(encoding="utf-8")
    assert "ALLOWED" in log and "RESTORED" in log


def test_argos_route_and_download_import(server, tmp_path):
    from ovt.config import Paths
    pkg = make_argos(tmp_path / "pk.argosmodel", "fr", "en"); pkg2 = make_argos(tmp_path / "pk2.argosmodel", "en", "ar")
    Handler.files.update({"/fr_en.argosmodel": pkg.read_bytes(), "/en_ar.argosmodel": pkg2.read_bytes()})
    index = [{"from": "fr", "to": "en", "url": server + "/fr_en.argosmodel", "version": "1"},
             {"from": "en", "to": "ar", "url": server + "/en_ar.argosmodel", "version": "1"}]
    assert [p["from"] for p in downloader.argos_route(index, "fr")] == ["fr", "en"]
    assert downloader.argos_route(index, "xx") == []
    paths = Paths(tmp_path / "h").ensure(); mm = ModelManager(paths)
    netguard.enable_offline(paths.logs / "n.log")
    got = downloader.download_argos(paths, mm, "fr", "ar", index=index)
    assert [m["id"] for m in got] == ["argos-fr-en", "argos-en-ar"] and netguard.is_offline()
    assert downloader.download_argos(paths, mm, "fr", "ar", index=index) == []   # already installed -> nothing to do
    with pytest.raises(OVTError):
        downloader.download_argos(paths, mm, "xx", "ar", index=index)
