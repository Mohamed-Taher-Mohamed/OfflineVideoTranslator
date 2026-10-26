"""Model manager: import / verify / list / delete. Downloading is a SEPARATE explicit action.

Layout:  models/asr/<id>/{model.bin,config.json,tokenizer.json,...}  + model.ovt.json
         models/mt/<id>/<file>.argosmodel                          + model.ovt.json
The manifest stores a SHA-256 of every file computed at import time. That detects later
corruption / truncated copies; it does NOT prove the model came from the official source
(compare against the publisher's checksum yourself if you need provenance - pass
expected_sha256 to import_model for that).
"""
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from .errors import ModelIntegrityError, ModelMissingError, OfflineViolation, OVTError
from .fsutil import atomic_write_json, read_json
from . import netguard

MANIFEST = "model.ovt.json"
ASR_REQUIRED = ("model.bin", "config.json", "tokenizer.json")

# Static catalog (no network needed to read it). Sizes are APPROXIMATE, from memory of the
# published repos, and must be re-verified against the real download before being shown as facts.
ASR_CATALOG = {
    "whisper-tiny":     {"repo": "Systran/faster-whisper-tiny",     "approx_mb": 75,   "license": "MIT"},
    "whisper-base":     {"repo": "Systran/faster-whisper-base",     "approx_mb": 145,  "license": "MIT"},
    "whisper-small":    {"repo": "Systran/faster-whisper-small",    "approx_mb": 480,  "license": "MIT"},
    "whisper-medium":   {"repo": "Systran/faster-whisper-medium",   "approx_mb": 1500, "license": "MIT"},
    "whisper-large-v3": {"repo": "Systran/faster-whisper-large-v3", "approx_mb": 3100, "license": "MIT"},
}


def sha256_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _safe_id(model_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", model_id or ""):
        raise OVTError("Invalid model id (use letters, digits, . _ - only): %r" % model_id)
    return model_id


class ModelManager:
    def __init__(self, paths):
        self.paths = paths

    def _dir(self, kind, model_id) -> Path:
        return self.paths.models / kind / _safe_id(model_id)

    def list(self, kind: Optional[str] = None) -> List[Dict]:
        out = []
        for k in ([kind] if kind else ["asr", "mt"]):
            base = self.paths.models / k
            if not base.is_dir():
                continue
            for d in sorted(base.iterdir()):
                m = read_json(d / MANIFEST)
                if m:
                    m["path"] = str(d)
                    out.append(m)
        return out

    def get(self, kind, model_id) -> Dict:
        m = read_json(self._dir(kind, model_id) / MANIFEST)
        if not m:
            raise ModelMissingError(
                "Model '%s' (%s) is not installed. Import it with: ovt models import --kind %s --id %s --path <folder-or-file>"
                % (model_id, kind, kind, model_id), kind=kind, model_id=model_id)
        m["path"] = str(self._dir(kind, model_id))
        return m

    def import_model(self, kind: str, src, model_id: str, name: str = "", version: str = "",
                     languages: Optional[List[str]] = None, license: str = "", source: str = "",
                     expected_sha256: Optional[Dict[str, str]] = None) -> Dict:
        src = Path(src)
        dest = self._dir(kind, model_id)
        staging = dest.with_name(dest.name + ".importing")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        mt_type = ""
        try:
            if kind == "asr":
                if not src.is_dir():
                    raise OVTError("ASR model must be a folder (faster-whisper/CTranslate2 format): %s" % src)
                missing = [f for f in ASR_REQUIRED if not (src / f).is_file()]
                if missing:
                    raise ModelIntegrityError("Not a valid faster-whisper model folder, missing: %s" % ", ".join(missing))
                for f in src.iterdir():
                    if f.is_file():
                        shutil.copy2(f, staging / f.name)
                languages = languages or ["multilingual (verify per language)"]
            elif kind == "mt":
                if src.is_dir():                      # NLLB-style CTranslate2 folder
                    if not (src / "model.bin").is_file() or not list(src.glob("*.model")):
                        raise ModelIntegrityError("Translation folder needs model.bin and a SentencePiece *.model file")
                    for f in src.iterdir():
                        if f.is_file():
                            shutil.copy2(f, staging / f.name)
                    mt_type = "nllb"
                    languages = languages or ["multilingual (NLLB-style; verify per language)"]
                else:                                 # Argos package
                    if not src.is_file() or not zipfile.is_zipfile(src):
                        raise OVTError("MT model must be an Argos .argosmodel file or an NLLB CTranslate2 folder: %s" % src)
                    with zipfile.ZipFile(src) as z:
                        bad = z.testzip()
                        if bad:
                            raise ModelIntegrityError("Corrupt archive member: %s" % bad)
                        meta_name = next((n for n in z.namelist() if n.endswith("metadata.json")), None)
                        if not meta_name:
                            raise ModelIntegrityError("Not an Argos package: metadata.json missing")
                        meta = json.loads(z.read(meta_name).decode("utf-8"))
                    languages = languages or ["%s->%s" % (meta.get("from_code"), meta.get("to_code"))]
                    name = name or meta.get("name", "")
                    version = version or str(meta.get("package_version", ""))
                    shutil.copy2(src, staging / (_safe_id(model_id) + ".argosmodel"))
                    mt_type = "argos"
            else:
                raise OVTError("unknown model kind: %s" % kind)

            files = {}
            total = 0
            for f in sorted(staging.iterdir()):
                files[f.name] = {"sha256": sha256_file(f), "size": f.stat().st_size}
                total += f.stat().st_size
            for fname, want in (expected_sha256 or {}).items():
                if files.get(fname, {}).get("sha256") != want.lower():
                    raise ModelIntegrityError("Checksum mismatch for %s (expected %s)" % (fname, want))
            manifest = {"id": model_id, "kind": kind, "name": name or model_id, "version": version,
                        "languages": languages, "license": license or "UNSPECIFIED - check before distributing",
                        "source": source or "local import", "mt_type": mt_type, "files": files, "size_bytes": total}
            atomic_write_json(staging / MANIFEST, manifest)
            backup = None
            if dest.exists():                       # keep the previous version until the new one is in place
                backup = dest.with_name(dest.name + ".previous")
                if backup.exists():
                    shutil.rmtree(backup)
                dest.rename(backup)
            staging.rename(dest)
            if backup and backup.exists():
                shutil.rmtree(backup)
            return self.get(kind, model_id)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def verify(self, kind, model_id, full: bool = True) -> List[str]:
        m = self.get(kind, model_id)
        problems = []
        for fname, meta in m["files"].items():
            p = Path(m["path"]) / fname
            if not p.is_file():
                problems.append("missing: " + fname)
            elif p.stat().st_size != meta["size"]:
                problems.append("size mismatch: " + fname)
            elif full and sha256_file(p) != meta["sha256"]:
                problems.append("checksum mismatch: " + fname)
        return problems

    def require(self, kind, model_id) -> Dict:
        """Used before running: cheap check (existence + sizes). Full hash via verify()."""
        m = self.get(kind, model_id)
        problems = self.verify(kind, model_id, full=False)
        if problems:
            raise ModelIntegrityError("Model '%s' is damaged (%s). Re-import it." % (model_id, "; ".join(problems)))
        return m

    def delete(self, kind, model_id) -> None:
        d = self._dir(kind, model_id)
        if d.exists():
            shutil.rmtree(d)

    def download_asr(self, key: str, allow_network: bool = False) -> Dict:
        """Explicit user action. Refuses while offline mode is on. NOT exercised in the dev sandbox."""
        if netguard.is_offline() and not allow_network:
            raise OfflineViolation("Offline mode is ON: downloads are blocked. Approve a one-time download window, or import a local copy.")
        if key not in ASR_CATALOG:
            raise OVTError("unknown catalog model: %s (known: %s)" % (key, ", ".join(ASR_CATALOG)))
        from huggingface_hub import snapshot_download  # imported lazily: only the downloader needs it
        info = ASR_CATALOG[key]
        tmp = self.paths.temp / ("dl-" + key)
        with netguard.temporarily_online("download " + info["repo"]):
            snapshot_download(info["repo"], local_dir=str(tmp))
        m = self.import_model("asr", tmp, key, name=key, license=info["license"],
                              source="https://huggingface.co/" + info["repo"])
        shutil.rmtree(tmp, ignore_errors=True)
        return m
