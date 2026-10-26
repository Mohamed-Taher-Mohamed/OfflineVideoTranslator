"""User glossary: terms to keep untouched and forced translations.

File format (JSON):  {"keep": ["Hogwarts", "iPhone"], "map": {"Dr.": "د.", "ninja": "نينجا"}}
Mechanism: terms are swapped for opaque placeholders before the MT engine sees the text and
swapped back afterwards. If an engine mangles a placeholder the term is reported as
'lost' so it can be fixed by hand - it is never silently dropped.
"""
import hashlib
import json
import re
from pathlib import Path


class Glossary:
    def __init__(self, keep=None, mapping=None):
        self.keep = [k for k in (keep or []) if k.strip()]
        self.map = {k: v for k, v in (mapping or {}).items() if k.strip()}

    @classmethod
    def load(cls, path):
        if not path:
            return cls()
        d = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return cls(d.get("keep"), d.get("map"))

    def fingerprint(self) -> str:
        blob = json.dumps({"k": sorted(self.keep), "m": sorted(self.map.items())}, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def _terms(self):
        items = [(t, t) for t in self.keep] + list(self.map.items())
        return sorted(items, key=lambda kv: -len(kv[0]))  # longest first

    def protect(self, text: str):
        """Return (text_with_placeholders, {placeholder: replacement})."""
        table = {}
        for n, (src, dst) in enumerate(self._terms()):
            pat = re.compile(r"(?<!\w)" + re.escape(src) + r"(?!\w)", re.IGNORECASE if src.islower() else 0)
            ph = "XQ%dZ" % (n + 1)
            if pat.search(text):
                text = pat.sub(ph, text)
                table[ph] = dst
        return text, table

    @staticmethod
    def restore(text: str, table: dict):
        lost = []
        for ph, dst in table.items():
            if ph in text:
                text = text.replace(ph, dst)
            else:
                lost.append(dst)
        return text, lost
