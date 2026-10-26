import socket
import threading

import pytest

from ovt import netguard
from ovt.errors import OfflineViolation


def test_external_connection_and_dns_are_blocked_and_logged(tmp_path):
    log = tmp_path / "logs" / "network.log"
    netguard.enable_offline(log)
    with pytest.raises(OfflineViolation):
        socket.create_connection(("93.184.216.34", 80), timeout=2)
    with pytest.raises(OfflineViolation):
        socket.getaddrinfo("example.com", 80)
    s = socket.socket()
    with pytest.raises(OfflineViolation):
        s.connect(("8.8.8.8", 53))
    s.close()
    text = log.read_text(encoding="utf-8")
    assert "93.184.216.34" in text and "example.com" in text and "8.8.8.8" in text


def test_loopback_still_works():
    netguard.enable_offline()
    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)
    port = srv.getsockname()[1]
    t = threading.Thread(target=lambda: srv.accept()[0].close()); t.start()
    c = socket.create_connection(("127.0.0.1", port), timeout=2)
    c.close(); t.join(); srv.close()


def test_library_http_client_is_blocked():
    import urllib.request
    netguard.enable_offline()
    with pytest.raises(Exception) as ei:
        urllib.request.urlopen("http://example.com", timeout=2)
    assert "Offline mode" in str(ei.value) or isinstance(ei.value.__cause__, OfflineViolation) or "blocked" in str(ei.value).lower()


def test_hub_env_vars_set():
    import os
    netguard.enable_offline()
    assert os.environ["HF_HUB_OFFLINE"] == "1"
