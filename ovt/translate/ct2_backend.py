"""Lightweight local MT: CTranslate2 + SentencePiece only (no torch / stanza / spacy).

Runs two kinds of locally imported packages:
  * Argos packages (.argosmodel zip = CTranslate2 model + sentencepiece.model + metadata.json)
    - one language pair each; non-English pairs are chained through English.
  * NLLB-200 CTranslate2 folders (model.bin + a SentencePiece model)
    - one model, direct translation between ~200 languages (Arabic = MSA 'arb_Arab').

Everything is loaded from disk; nothing here can reach the network.
STATUS: plumbing is covered by tests with a tiny randomly-weighted CTranslate2 model (proves tokenise ->
CTranslate2 -> detokenise runs end to end). Translation QUALITY with real packages is untested here.
"""
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base import Support, Translator

# ISO-639-1 -> FLORES-200 code used by NLLB
FLORES = {
    "en": "eng_Latn", "ar": "arb_Arab", "fr": "fra_Latn", "es": "spa_Latn", "de": "deu_Latn", "it": "ita_Latn",
    "pt": "por_Latn", "ru": "rus_Cyrl", "tr": "tur_Latn", "ja": "jpn_Jpan", "ko": "kor_Hang", "zh": "zho_Hans",
    "hi": "hin_Deva", "fa": "pes_Arab", "he": "heb_Hebr", "nl": "nld_Latn", "pl": "pol_Latn", "uk": "ukr_Cyrl",
    "id": "ind_Latn", "vi": "vie_Latn", "th": "tha_Thai", "sv": "swe_Latn", "el": "ell_Grek", "ur": "urd_Arab",
    "bn": "ben_Beng", "cs": "ces_Latn", "ro": "ron_Latn", "hu": "hun_Latn", "fi": "fin_Latn", "da": "dan_Latn",
    "no": "nob_Latn", "bg": "bul_Cyrl", "sr": "srp_Cyrl", "hr": "hrv_Latn", "sk": "slk_Latn", "ms": "zsm_Latn",
    "sw": "swh_Latn", "ta": "tam_Taml", "te": "tel_Telu", "ca": "cat_Latn", "az": "azj_Latn", "kk": "kaz_Cyrl",
}
_SENT = re.compile(r"(?<=[.!?؟。！？…])\s+")


def split_sentences(text: str, max_len: int = 250) -> List[str]:
    parts = [p for p in _SENT.split(text.strip()) if p]
    out = []
    for p in parts:
        while len(p) > max_len * 2:                      # runaway text without punctuation
            cut = p.rfind(" ", 0, max_len)
            cut = cut if cut > 0 else max_len
            out.append(p[:cut]); p = p[cut:].lstrip()
        out.append(p)
    return out or [text]


def safe_extract(zip_path, dest: Path) -> None:
    dest = Path(dest).resolve()
    with zipfile.ZipFile(zip_path) as z:
        for m in z.infolist():
            target = (dest / m.filename).resolve()
            if dest not in target.parents and target != dest:
                raise ValueError("unsafe path in archive: %s" % m.filename)     # zip-slip
        z.extractall(dest)


class _Runner:
    """One loaded CTranslate2 model + tokenizer."""

    def __init__(self, model_dir: Path, spm_file: Path, device="auto", compute_type="auto", threads=0):
        import ctranslate2
        import sentencepiece as spm
        if compute_type == "auto":
            compute_type = "int8" if device in ("auto", "cpu") else "default"
        self.ct = ctranslate2.Translator(str(model_dir), device=device, compute_type=compute_type, intra_threads=threads)
        # Read the file in Python and hand SentencePiece the bytes: its C++ code opens `model_file` with the ANSI
        # code page on Windows, which fails ("No such file ... Error #2") when the path has Arabic letters.
        self.sp = spm.SentencePieceProcessor(model_proto=Path(spm_file).read_bytes())

    @staticmethod
    def decode_limit(token_lists) -> int:
        """Output budget relative to the longest source sentence: stops runaway repetition loops
        (a subtitle line never legitimately needs 3x the source tokens + 10)."""
        return min(256, 3 * max((len(t) for t in token_lists), default=0) + 10)

    def _run(self, token_lists, target_prefix, beam):
        res = self.ct.translate_batch(
            token_lists, target_prefix=target_prefix, beam_size=beam, length_penalty=0.2,
            replace_unknowns=False, max_batch_size=1024, batch_type="tokens",
            max_decoding_length=self.decode_limit(token_lists))
        return [r.hypotheses[0] for r in res]

    def decode(self, tokens):
        return self.sp.decode_pieces(tokens).replace("\u2581", " ").strip()


class ArgosPackage(_Runner):
    def __init__(self, pkg_dir: Path, **kw):
        meta = json.loads((pkg_dir / "metadata.json").read_text(encoding="utf-8"))
        self.src, self.tgt = meta["from_code"], meta["to_code"]
        self.prefix = meta.get("target_prefix", "")
        spm_file = pkg_dir / "sentencepiece.model"
        if not spm_file.exists():
            raise ValueError("package %s->%s uses a BPE tokenizer (not supported by the light engine yet)" % (self.src, self.tgt))
        super().__init__(pkg_dir / "model", spm_file, **kw)

    def translate(self, texts: List[str], beam=4) -> List[str]:
        flat, owners = [], []
        for i, t in enumerate(texts):
            for s in (split_sentences(t) if t.strip() else []):
                flat.append(self.sp.encode(s, out_type=str)); owners.append(i)
        out = [[] for _ in texts]
        if flat:
            prefix = [[self.prefix]] * len(flat) if self.prefix else None
            hyps = self._run(flat, prefix, beam)
            for o, h in zip(owners, hyps):
                if self.prefix and h and h[0] == self.prefix:
                    h = h[1:]
                out[o].append(self.decode(h))
        return [" ".join(p) for p in out]


class NLLBModel(_Runner):
    def __init__(self, model_dir: Path, **kw):
        spm_file = next((p for p in sorted(model_dir.glob("*.model"))), None)
        if spm_file is None:
            raise ValueError("NLLB folder needs a SentencePiece *.model file next to model.bin")
        super().__init__(model_dir, spm_file, **kw)

    def pairs_ok(self, src, tgt):
        return src in FLORES and tgt in FLORES

    def translate(self, texts: List[str], src: str, tgt: str, beam=4) -> List[str]:
        s_code, t_code = FLORES[src], FLORES[tgt]
        flat, owners = [], []
        for i, t in enumerate(texts):
            for s in (split_sentences(t) if t.strip() else []):
                flat.append([s_code] + self.sp.encode(s, out_type=str) + ["</s>"]); owners.append(i)
        out = [[] for _ in texts]
        if flat:
            hyps = self._run(flat, [[t_code]] * len(flat), beam)
            for o, h in zip(owners, hyps):
                if h and h[0] == t_code:
                    h = h[1:]
                out[o].append(self.decode(h))
        return [" ".join(p) for p in out]


class CT2Translator(Translator):
    name = "ct2"
    version = "ct2-light"

    def __init__(self, packages: List[Tuple[str, Path]], extract_root: Path, device="auto", compute_type="auto"):
        """packages: [(kind, path)] where kind is 'argos' (.argosmodel file) or 'nllb' (folder)."""
        self.kw = dict(device=device, compute_type=compute_type)
        self.argos: Dict[Tuple[str, str], ArgosPackage] = {}
        self.nllb: Optional[NLLBModel] = None
        self.load_errors: List[str] = []
        extract_root = Path(extract_root)
        for kind, path in packages:
            try:
                if kind == "argos":
                    meta_dir = self._extract_argos(Path(path), extract_root)
                    p = ArgosPackage(meta_dir, **self.kw)
                    self.argos[(p.src, p.tgt)] = p
                elif kind == "nllb":
                    self.nllb = NLLBModel(Path(path), **self.kw)
            except Exception as e:      # one bad package must not hide the others; surfaced via supports()
                self.load_errors.append("%s: %s" % (Path(path).name, e))

    @staticmethod
    def _extract_argos(file: Path, root: Path) -> Path:
        target = root / file.stem
        marker = target / ".extracted"
        if not marker.exists():
            target.mkdir(parents=True, exist_ok=True)
            safe_extract(file, target)
            marker.write_text("ok")
        for meta in target.rglob("metadata.json"):
            return meta.parent
        raise ValueError("metadata.json not found after extraction")

    def _route(self, src, tgt) -> Optional[List[Tuple[str, str]]]:
        if self.nllb and self.nllb.pairs_ok(src, tgt):
            return [("nllb", None)]
        if (src, tgt) in self.argos:
            return [("argos", (src, tgt))]
        if (src, "en") in self.argos and ("en", tgt) in self.argos:
            return [("argos", (src, "en")), ("argos", ("en", tgt))]
        return None

    def supports(self, src, tgt) -> Support:
        r = self._route(src, tgt)
        if r is None:
            why = "no installed model covers '%s' -> '%s'" % (src, tgt)
            if self.load_errors:
                why += " (load problems: %s)" % "; ".join(self.load_errors)
            return Support(False, reason=why)
        if r[0][0] == "nllb":
            return Support(True, "direct (NLLB multilingual)")
        return Support(True, "direct" if len(r) == 1 else "pivot via en")

    def translate(self, texts, src, tgt):
        route = self._route(src, tgt)
        if route is None:
            raise RuntimeError("language pair not installed")
        if route[0][0] == "nllb":
            return self.nllb.translate(texts, src, tgt)
        cur = texts
        for _, pair in route:
            cur = self.argos[pair].translate(cur)
        return cur
