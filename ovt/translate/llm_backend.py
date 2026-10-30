"""Subtitle translation with a language model behind an OpenAI-compatible HTTP endpoint.

Works with a LOCAL server on this PC (Ollama, LM Studio, llama.cpp `llama-server`, ...: loopback traffic is
allowed even in offline mode, so nothing leaves the machine) and, only if the user opts in, with a cloud API
(key + explicit consent; text only, never audio/video).

Why: classic NMT (Argos) translates sentence by sentence and tends to be literal. An LLM sees neighbouring
lines, resolves pronouns/cut-off sentences, adapts idioms and keeps the register of the speaker.
STATUS: protocol/prompt/parsing/retry logic is tested against a real local HTTP server that imitates the API;
translation QUALITY depends on the model you run and has not been measured here."""
import json
import re
import urllib.error
import urllib.request
from typing import Dict, List, Optional
from urllib.parse import urlparse

from .. import netguard
from ..errors import OVTError
from .base import Support, Translator

PROMPT_VERSION = "v1"
_NAMES = {"en": "English", "fr": "French", "es": "Spanish", "de": "German", "it": "Italian", "pt": "Portuguese",
          "ru": "Russian", "tr": "Turkish", "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "hi": "Hindi",
          "fa": "Persian", "he": "Hebrew", "nl": "Dutch", "pl": "Polish", "uk": "Ukrainian", "id": "Indonesian",
          "ur": "Urdu", "ar": "Arabic"}
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_NUM = re.compile(r"^\s*(\d{1,3})\s*[\.\)\-:：]\s*(.*)$")

PRESETS = {
    "ollama": ("Ollama", "http://127.0.0.1:11434/v1"),
    "lmstudio": ("LM Studio", "http://127.0.0.1:1234/v1"),
    "llamacpp": ("llama.cpp server", "http://127.0.0.1:8080/v1"),
    "custom": ("Custom / cloud", ""),
}


def is_loopback_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1") or host.startswith("127.")


def engine_tag(settings) -> str:
    """Goes into the translation-cache fingerprint: switching engine/model never reuses old translations."""
    if getattr(settings, "mt_engine", "nmt") == "llm":
        return "llm:%s:%s" % (settings.llm_model, PROMPT_VERSION)
    return "nmt"


def build_system_prompt(src: str, tgt: str = "ar") -> str:
    mixed = src in ("mixed", "")
    src_name = ("the source language (it may DIFFER from line to line: English, Spanish, Hindi, ...; a line that is "
                "already Arabic must be kept as it is)") if mixed else _NAMES.get(src, src)
    return (
        "You are an expert audiovisual translator who writes subtitles. Translate the numbered lines (%s) into "
        "Modern Standard Arabic (الفصحى المبسطة).\n"
        "Rules:\n"
        "- Translate the MEANING, naturally and idiomatically. Never translate word by word. Adapt idioms, slang and "
        "humour to natural Arabic equivalents.\n"
        "- Keep the speaker's tone and register. Keep each line short and easy to read as a subtitle.\n"
        "- The lines are consecutive parts of one video. Use the neighbouring lines and the previous context to resolve "
        "pronouns, names and sentences that continue in the next line.\n"
        "- Keep proper names, brand names and numbers. Tokens such as XQ1Z are protected terms: copy them unchanged.\n"
        "- Output exactly one line per input line, with the same number, in the form: <number>. <Arabic text>\n"
        "- Do not merge or split lines. No explanations, notes or extra text." % src_name)


def parse_numbered(text: str, n: int) -> Dict[int, str]:
    """Parse '1. ...' lines (tolerating wrapped continuation lines, quotes, <think> blocks)."""
    text = _THINK.sub("", text or "").strip()
    out: Dict[int, str] = {}
    cur = None
    for line in text.splitlines():
        m = _NUM.match(line)
        if m and 1 <= int(m.group(1)) <= n:
            cur = int(m.group(1))
            out[cur] = m.group(2).strip()
        elif cur is not None and line.strip():
            out[cur] = (out[cur] + " " + line.strip()).strip()
    for k, v in out.items():
        out[k] = v.strip().strip('"“”«»').strip()
    return out


class LLMTranslator(Translator):
    name = "llm"
    version = PROMPT_VERSION
    handles_mixed = True

    def __init__(self, base_url: str, model: str, api_key: str = "", allow_cloud: bool = False,
                 temperature: float = 0.2, timeout: float = 600.0, lines_per_call: int = 8, context_lines: int = 4):
        self.base = base_url.rstrip("/")
        self.model, self.key, self.allow_cloud = model, api_key, allow_cloud
        self.temperature, self.timeout = temperature, timeout
        self.lines_per_call, self.context_lines = max(1, lines_per_call), max(0, context_lines)
        self.loopback = is_loopback_url(self.base)
        self._history: List[tuple] = []                  # (source, translation) of the last lines, for continuity

    # ---------- transport ----------
    def _request(self, path: str, payload: Optional[dict] = None):
        if not self.loopback and not self.allow_cloud:
            raise OVTError("The AI server '%s' is not on this computer. To use a cloud service enable the cloud "
                           "option in Settings (the subtitle TEXT will be sent to that provider)." % self.base)
        url = self.base + path
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8") if payload is not None else None,
                                     headers=headers, method="POST" if payload is not None else "GET")
        ctx = netguard.temporarily_online("cloud translation request to " + urlparse(url).netloc) \
            if not self.loopback else _Null()
        try:
            with ctx:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            raise OVTError("AI server answered HTTP %d: %s" % (e.code, body)) from e
        except (OSError, ValueError) as e:
            raise OVTError("Cannot reach the AI server at %s (%s). Start Ollama / LM Studio (Start Server) first." % (self.base, e)) from e

    def list_models(self) -> List[str]:
        d = self._request("/models")
        return sorted(m.get("id", "") for m in d.get("data", []) if m.get("id"))

    def check(self) -> str:
        """Used by the Settings 'Test' button: connectivity + the model exists + a real sample translation."""
        models = self.list_models()
        if self.model not in models:
            raise OVTError("Model '%s' is not available on the server. Available: %s"
                           % (self.model, ", ".join(models[:12]) or "(none)"))
        out = self.translate(["Break a leg out there tonight!"], "en", "ar")[0]
        return out

    # ---------- Translator interface ----------
    def supports(self, src: str, tgt: str) -> Support:
        if tgt != "ar":
            return Support(False, reason="the LLM engine is configured for Arabic output only")
        try:
            models = self.list_models()
        except OVTError as e:
            return Support(False, reason=str(e))
        if models and self.model not in models:
            return Support(False, reason="model '%s' not found on the AI server (available: %s)" % (self.model, ", ".join(models[:10])))
        return Support(True, "LLM (%s)" % ("local" if self.loopback else "cloud"))

    def _chat(self, system: str, user: str) -> str:
        d = self._request("/chat/completions", {"model": self.model, "temperature": self.temperature, "stream": False,
                                                 "messages": [{"role": "system", "content": system},
                                                              {"role": "user", "content": user}]})
        try:
            return d["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise OVTError("Unexpected answer from the AI server: %s" % str(d)[:200]) from e

    def _user_prompt(self, lines: List[str]) -> str:
        ctx = ""
        if self._history and self.context_lines:
            ctx = "Previous lines (context only - do NOT translate them again):\n" + "\n".join(
                "- %s => %s" % (s, t) for s, t in self._history[-self.context_lines:]) + "\n\n"
        return ctx + "Lines to translate:\n" + "\n".join("%d. %s" % (i + 1, t) for i, t in enumerate(lines))

    def _translate_batch(self, lines: List[str], system: str) -> List[str]:
        got = parse_numbered(self._chat(system, self._user_prompt(lines)), len(lines))
        res = []
        for i, src in enumerate(lines, 1):
            tr = got.get(i, "")
            if not tr:                                   # the model skipped/merged a line: ask for that one alone
                tr = parse_numbered(self._chat(system, "Lines to translate:\n1. " + src), 1).get(1, "")
            res.append(tr)
        for s, t in zip(lines, res):
            if t:
                self._history.append((s, t))
        del self._history[:-max(self.context_lines, 1) * 2]
        return res

    def translate(self, texts: List[str], src: str, tgt: str) -> List[str]:
        system = build_system_prompt(src, tgt)
        out: List[str] = []
        idx = [i for i, t in enumerate(texts) if t.strip()]
        results = {}
        for b in range(0, len(idx), self.lines_per_call):
            chunk = idx[b:b + self.lines_per_call]
            for i, tr in zip(chunk, self._translate_batch([texts[i].replace("\n", " ") for i in chunk], system)):
                results[i] = tr
        return [results.get(i, texts[i]) for i in range(len(texts))]


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
