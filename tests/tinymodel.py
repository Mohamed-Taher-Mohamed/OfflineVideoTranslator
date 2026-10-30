"""Builds a TINY randomly-weighted CTranslate2 model + SentencePiece model.
It translates to gibberish by design; it exists only to prove the plumbing runs for real:
SentencePiece encode -> CTranslate2 translate_batch -> decode, zip packaging, loading from disk."""
import json
import re
import zipfile
from pathlib import Path

import numpy as np

CORPUS = ("hello world this is a test of the small model . how are you today ? fine thank you . "
          "the meeting starts tomorrow . please close the door . i did not expect to see you here . " * 8)


def train_spm(dirpath: Path, extra=("eng_Latn", "arb_Arab"), vocab=60) -> Path:
    """SentencePiece's C++ trainer cannot open non-ASCII paths on Windows, so train in an ASCII temp folder
    and copy the finished model into `dirpath` (which may contain Arabic letters)."""
    import shutil
    import tempfile
    import sentencepiece as spm
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="spm_"))
    try:
        txt = work / "corpus.txt"
        txt.write_text(CORPUS.replace(" . ", ".\n"), encoding="utf-8")
        spm.SentencePieceTrainer.train(input=str(txt), model_prefix=str(work / "sentencepiece"), vocab_size=vocab,
                                       user_defined_symbols=list(extra), hard_vocab_limit=False, minloglevel=2)
        shutil.copy(work / "sentencepiece.model", dirpath / "sentencepiece.model")
        shutil.copy(work / "sentencepiece.vocab", dirpath / "sentencepiece.vocab")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return dirpath / "sentencepiece.model"


def _set(spec, path, arr):
    obj = spec
    parts = path.split("/")
    for p in parts[:-1]:
        m = re.fullmatch(r"(.*)_(\d+)", p)
        if hasattr(obj, p):
            obj = getattr(obj, p)
        else:
            obj = getattr(obj, m.group(1))[int(m.group(2))]
    setattr(obj, parts[-1], arr)


def build_ct2(model_dir: Path, spm_model: Path, d=16, ffn=32, seed=0):
    import sentencepiece as spm
    from ctranslate2.specs import TransformerSpec
    sp = spm.SentencePieceProcessor(model_proto=Path(spm_model).read_bytes())
    vocab = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    V = len(vocab)
    rng = np.random.default_rng(seed)
    spec = TransformerSpec.from_config(num_layers=1, num_heads=2, pre_norm=True)
    spec.register_source_vocabulary(vocab)
    spec.register_target_vocabulary(vocab)
    for name, val in spec.variables().items():
        if val is not None:
            continue
        if name.endswith("gamma"):
            arr = np.ones(d, np.float32)
        elif name.endswith("beta"):
            arr = np.zeros(d, np.float32)
        elif "embeddings" in name:
            arr = rng.normal(0, 0.5, (V, d))
        elif "projection" in name:
            arr = rng.normal(0, 0.5, (V, d))
        elif "self_attention/linear_0" in name:
            arr = rng.normal(0, 0.3, (3 * d, d))
        elif "attention/linear_1" in name and "self_attention" not in name:
            arr = rng.normal(0, 0.3, (2 * d, d))
        elif "ffn/linear_0" in name:
            arr = rng.normal(0, 0.3, (ffn, d))
        elif "ffn/linear_1" in name:
            arr = rng.normal(0, 0.3, (d, ffn))
        else:
            arr = rng.normal(0, 0.3, (d, d))
        _set(spec, name, arr.astype(np.float32))
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    spec.validate()
    spec.optimize(quantization=None)
    spec.save(str(model_dir))
    return model_dir


def make_argos_package(path: Path, work: Path, src="en", tgt="ar", prefix="") -> Path:
    work.mkdir(parents=True, exist_ok=True)
    spm_model = train_spm(work)
    mdir = work / "model"
    build_ct2(mdir, spm_model)
    top = "translate-%s_%s-1_0" % (src, tgt)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(top + "/metadata.json", json.dumps({"from_code": src, "to_code": tgt, "package_version": "1.0",
                                                       "name": "tiny", "target_prefix": prefix}))
        z.write(spm_model, top + "/sentencepiece.model")
        for f in mdir.iterdir():
            z.write(f, top + "/model/" + f.name)
    return path


def make_nllb_dir(dirpath: Path) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    spm_model = train_spm(dirpath)
    build_ct2(dirpath, spm_model)
    return dirpath
