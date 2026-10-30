"""LLM translator tested against a REAL local HTTP server that imitates an OpenAI-compatible API
(Ollama / LM Studio / llama-server). It proves protocol, prompting, parsing, retries and privacy rules,
NOT translation quality (that depends on the model you run)."""
import http.server
import json
import re
import threading

import pytest

from conftest import FakeASR
from ovt import netguard
from ovt.errors import OVTError
from ovt.pipeline import JobOptions, Pipeline
from ovt.subtitles import Cue
from ovt.translate.llm_backend import LLMTranslator, build_system_prompt, engine_tag, is_loopback_url, parse_numbered
from ovt.translate.service import translate_cues


class FakeLLM(http.server.BaseHTTPRequestHandler):
    requests = []
    behaviour = {"skip_first_n3": False, "think": False, "fail": False, "models": ["qwen2.5:7b"]}

    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path == "/v1/models":
            self._send({"data": [{"id": m} for m in self.behaviour["models"]]})
        else:
            self.send_error(404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeLLM.requests.append({"body": body, "auth": self.headers.get("Authorization")})
        if self.behaviour["fail"]:
            self._send({"error": "boom"}, 500); return
        user = body["messages"][1]["content"]
        lines = user.split("Lines to translate:\n", 1)[1].splitlines()
        out = []
        for ln in lines:
            m = re.match(r"(\d+)\.\s(.*)", ln)
            if not m:
                continue
            if self.behaviour["skip_first_n3"] and m.group(1) == "3" and len(lines) > 1:
                continue                                            # model "forgets" line 3 of a batch
            out.append("%s. ع:%s" % (m.group(1), m.group(2)))
        text = "\n".join(out)
        if self.behaviour["think"]:
            text = "<think>let me think...\n1. wrong</think>\n" + text
        self._send({"choices": [{"message": {"content": text}}]})


@pytest.fixture
def server():
    FakeLLM.requests = []
    FakeLLM.behaviour.update({"skip_first_n3": False, "think": False, "fail": False, "models": ["qwen2.5:7b"]})
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield "http://127.0.0.1:%d/v1" % srv.server_address[1]
    srv.shutdown()


def test_prompt_and_parser_basics():
    assert "Modern Standard Arabic" in build_system_prompt("en") and "XQ1Z" in build_system_prompt("fr")
    assert parse_numbered("1. أ\n2) ب\n   تكملة\n3: \"ج\"", 3) == {1: "أ", 2: "ب تكملة", 3: "ج"}
    assert parse_numbered("<think>1. x</think>\n1. ok", 1) == {1: "ok"}
    assert parse_numbered("7. outside range\n1. in", 2) == {1: "in"}
    assert is_loopback_url("http://127.0.0.1:11434/v1") and is_loopback_url("http://localhost:1234/v1")
    assert not is_loopback_url("https://api.example.com/v1")


def test_translates_in_batches_keeps_order_and_gives_context(server):
    t = LLMTranslator(server, "qwen2.5:7b", lines_per_call=4)
    texts = ["line %d" % i for i in range(1, 11)]
    out = t.translate(texts, "en", "ar")
    assert out == ["ع:line %d" % i for i in range(1, 11)]
    assert len(FakeLLM.requests) == 3                                         # 4 + 4 + 2
    first = FakeLLM.requests[0]["body"]["messages"][1]["content"]
    second = FakeLLM.requests[1]["body"]["messages"][1]["content"]
    assert "Previous lines" not in first and "Previous lines" in second
    assert "line 4 => ع:line 4" in second                                     # earlier lines are passed as context
    assert FakeLLM.requests[0]["body"]["temperature"] == 0.2 and FakeLLM.requests[0]["body"]["stream"] is False


def test_skipped_line_is_requested_again_alone(server):
    FakeLLM.behaviour["skip_first_n3"] = True
    out = LLMTranslator(server, "qwen2.5:7b", lines_per_call=5).translate(["a", "b", "c", "d", "e"], "en", "ar")
    assert out == ["ع:a", "ع:b", "ع:c", "ع:d", "ع:e"]
    assert len(FakeLLM.requests) == 2 and "1. c" in FakeLLM.requests[1]["body"]["messages"][1]["content"]


def test_think_blocks_are_removed_and_api_key_is_sent(server):
    FakeLLM.behaviour["think"] = True
    out = LLMTranslator(server, "qwen2.5:7b", api_key="sk-test").translate(["hello"], "en", "ar")
    assert out == ["ع:hello"] and FakeLLM.requests[0]["auth"] == "Bearer sk-test"


def test_supports_reports_unreachable_and_missing_model_clearly(server):
    ok = LLMTranslator(server, "qwen2.5:7b").supports("en", "ar")
    assert ok.ok and "local" in ok.route
    s = LLMTranslator(server, "llama3:70b").supports("en", "ar")
    assert not s.ok and "qwen2.5:7b" in s.reason                                 # lists what IS available
    dead = LLMTranslator("http://127.0.0.1:9/v1", "x", timeout=2).supports("en", "ar")
    assert not dead.ok and "Ollama" in dead.reason
    assert not LLMTranslator(server, "qwen2.5:7b").supports("en", "fr").ok
    assert LLMTranslator(server, "qwen2.5:7b").check() == "ع:Break a leg out there tonight!"


def test_server_error_is_a_clear_error(server):
    FakeLLM.behaviour["fail"] = True
    with pytest.raises(OVTError) as e:
        LLMTranslator(server, "qwen2.5:7b").translate(["x"], "en", "ar")
    assert "HTTP 500" in str(e.value)


def test_local_server_works_in_offline_mode_but_cloud_needs_consent(server, tmp_path, monkeypatch):
    netguard.enable_offline(tmp_path / "n.log")
    assert LLMTranslator(server, "qwen2.5:7b").translate(["hi"], "en", "ar") == ["ع:hi"]      # loopback is allowed
    with pytest.raises(OVTError) as e:
        LLMTranslator("https://api.example.com/v1", "m").translate(["hi"], "en", "ar")
    assert "cloud option" in str(e.value) and "TEXT" in str(e.value)               # blocked BEFORE any connection

    class R:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"choices": [{"message": {"content": "1. مرحبا"}}]}).encode()
    seen = {}
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=0: (seen.setdefault("online", not netguard.is_offline()), R())[1])
    out = LLMTranslator("https://api.example.com/v1", "m", api_key="k", allow_cloud=True).translate(["hi"], "en", "ar")
    assert out == ["مرحبا"] and seen["online"] is True and netguard.is_offline()    # window opened for the call only
    log = (tmp_path / "n.log").read_text(encoding="utf-8")
    assert "ALLOWED" in log and "api.example.com" in log and "RESTORED" in log


def test_works_inside_the_translation_service_with_glossary_and_units(server):
    from ovt.glossary import Glossary
    cues = [Cue(0, 2, "", source_text="Dr. Snape teaches at"), Cue(2.1, 4, "", source_text="Hogwarts every year.")]
    translate_cues(LLMTranslator(server, "qwen2.5:7b"), cues, "en", "ar", Glossary(["Hogwarts"], {"Dr.": "د."}))
    joined = " ".join(c.text for c in cues)
    assert "Hogwarts" in joined and "د." in joined and "XQ" not in joined


def test_pipeline_switching_engine_retranslates_but_reuses_speech(paths, ff, media, tmp_path, server):
    FakeASR.calls = 0
    mk = lambda tag: Pipeline(paths, ff, lambda: FakeASR(), lambda: LLMTranslator(server, "qwen2.5:7b"), None)
    o = JobOptions(str(media["multi"]), str(tmp_path / "o"), chunk_seconds=3600, asr_model_id="m", overwrite=True,
                   snap_to_silence=False, translator_tag="llm:qwen2.5:7b:v1")
    r = mk(0).run(o)
    n_asr, n_req = FakeASR.calls, len(FakeLLM.requests)
    assert "ع:" in open(r.outputs["srt"], encoding="utf-8").read() and n_req > 0
    mk(0).run(o)                                                                    # same engine -> cached
    assert len(FakeLLM.requests) == n_req
    o.translator_tag = "nmt"                                                         # different engine -> translation redone
    mk(0).run(o)
    assert FakeASR.calls == n_asr and len(FakeLLM.requests) > n_req                  # ...but recognition was reused


def test_engine_tag():
    from ovt.config import Settings
    assert engine_tag(Settings()) == "nmt"
    assert engine_tag(Settings(mt_engine="llm", llm_model="qwen2.5:7b")).startswith("llm:qwen2.5:7b:")
