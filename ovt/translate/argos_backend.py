"""LEGACY (not used by default): needs the heavy argostranslate/stanza/torch stack. Kept only as an alternative; the default is ct2_backend."""
"""Argos Translate backend (OpenNMT/CTranslate2). Packages are installed from LOCAL .argosmodel files.

Status: written against the argostranslate 1.11 source API, NOT executed in the dev sandbox
(its model host is unreachable and its dependencies - stanza/spacy/torch - are very large).
Treat as untested until you run `python -m ovt selftest-mt --src-lang en` on a real machine.
"""
import os
from pathlib import Path
from typing import List

from .base import Support, Translator


class ArgosTranslator(Translator):
    name = "argos"

    def __init__(self, packages_dir, model_files=()):
        os.makedirs(packages_dir, exist_ok=True)
        os.environ["ARGOS_PACKAGES_DIR"] = str(packages_dir)   # must be set BEFORE import
        import argostranslate.package as pkg
        import argostranslate.translate as tr
        self._pkg, self._tr = pkg, tr
        installed = {(p.from_code, p.to_code, str(p.package_version)) for p in pkg.get_installed_packages()}
        for f in model_files:
            pkg.install_from_path(Path(f))     # idempotent for an already installed pair/version
        self.version = "argostranslate"
        self._installed = installed

    def _get(self, src, tgt):
        langs = {l.code: l for l in self._tr.get_installed_languages()}
        if src not in langs or tgt not in langs:
            return None
        return langs[src].get_translation(langs[tgt])

    def supports(self, src: str, tgt: str) -> Support:
        t = self._get(src, tgt)
        if t is None:
            return Support(False, reason="no installed Argos package chain from '%s' to '%s'" % (src, tgt))
        direct = any(p.from_code == src and p.to_code == tgt for p in self._pkg.get_installed_packages())
        return Support(True, "direct" if direct else "pivot via English (lower quality)")

    def translate(self, texts: List[str], src: str, tgt: str) -> List[str]:
        t = self._get(src, tgt)
        if t is None:
            raise RuntimeError("language pair not installed")
        return [t.translate(x) if x.strip() else x for x in texts]
