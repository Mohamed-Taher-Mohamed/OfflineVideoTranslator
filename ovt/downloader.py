
"""Explicit model downloads (never called implicitly). Resumable, size-checked, cancellable.
Network access is only opened inside `netguard.temporarily_online(...)`, so offline mode is restored
afterwards no matter what happens."""
import http.client
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional

from . import netguard
from .errors import CancelledError, ModelIntegrityError, OVTError
from .models import sha256_file

ARGOS_INDEX_URL = "https://raw.githubusercontent.com/argosopentech/argospm-index/main/index.json"
UA = {"User-Agent": "OfflineVideoTranslator/0.1"}


def _open(url, start=0, timeout=30):
    req = urllib.request.Request(
        url,
        headers=dict(
            UA,
            **({"Range": "bytes=%d-" % start} if start else {})
        )
    )
    return urllib.request.urlopen(req, timeout=timeout)


def download_file(
    url: str,
    dest,
    expected_sha256: Optional[str] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    cancel: Optional[threading.Event] = None,
    chunk: int = 1 << 20
) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0

    try:
        resp = _open(url, have)

    except urllib.error.HTTPError as e:
        if e.code == 416 and have:
            # Nothing left to fetch: .part already complete
            resp = None
        else:
            raise OVTError(
                "Download failed (HTTP %d) for %s" % (e.code, url)
            ) from e

    except (OSError, http.client.HTTPException) as e:
        # Connection failed before the response was fully opened.
        # Keep the partial file so the download can be resumed.
        part.touch(exist_ok=True)
        raise OVTError(
            "Download incomplete: connection failed (%s). "
            "Run it again to resume from %d bytes."
            % (e, have)
        ) from e

    total = have

    if resp is not None:
        with resp:
            length = int(resp.headers.get("Content-Length") or 0)

            if resp.status == 206:
                total = have + length
            else:
                # Server ignored Range: start over
                have, total = 0, length

            mode = "ab" if resp.status == 206 else "wb"
            done = have

            with open(part, mode) as f:
                while True:
                    if cancel is not None and cancel.is_set():
                        raise CancelledError(
                            "Download cancelled (partial file kept for resume)."
                        )

                    try:
                        b = resp.read(chunk)

                    except (OSError, http.client.HTTPException) as e:
                        raise OVTError(
                            "Download interrupted (%s). "
                            "Run it again to resume from %d bytes."
                            % (e, done)
                        ) from e

                    if not b:
                        break

                    f.write(b)
                    done += len(b)

                    if progress:
                        progress(done, total)

            if total and done != total:
                raise OVTError(
                    "Download incomplete (%d of %d bytes); "
                    "run it again to resume."
                    % (done, total)
                )

    if expected_sha256 and sha256_file(part) != expected_sha256.lower():
        part.unlink(missing_ok=True)
        raise ModelIntegrityError(
            "Checksum mismatch after download; the partial file was removed."
        )

    part.replace(dest)
    return dest


def list_argos_packages(index_url: str = ARGOS_INDEX_URL) -> List[dict]:
    with netguard.temporarily_online("fetch Argos package index"):
        with _open(index_url) as r:
            data = json.loads(r.read().decode("utf-8"))

    return [
        {
            "from": p["from_code"],
            "to": p["to_code"],
            "from_name": p.get("from_name", ""),
            "to_name": p.get("to_name", ""),
            "url": next(
                (l for l in p["links"] if l.startswith("http")),
                ""
            ),
            "version": str(p.get("package_version", ""))
        }
        for p in data
    ]


def argos_route(index: List[dict], src: str, tgt: str = "ar") -> List[dict]:
    """Packages needed for src -> tgt: direct, else via English.
    Empty list = impossible with Argos.
    """
    by = {(p["from"], p["to"]): p for p in index}

    if (src, tgt) in by:
        return [by[(src, tgt)]]

    if (src, "en") in by and ("en", tgt) in by:
        return [by[(src, "en")], by[("en", tgt)]]

    return []


def download_argos(
    paths,
    models,
    src: str,
    tgt: str = "ar",
    index=None,
    progress=None,
    cancel=None
) -> List[dict]:
    """Download + import every package needed for src->tgt.
    Returns the imported manifests.
    """
    index = index if index is not None else list_argos_packages()
    route = argos_route(index, src, tgt)

    if not route:
        raise OVTError(
            "Argos has no package chain from '%s' to '%s'."
            % (src, tgt)
        )

    out = []

    for p in route:
        mid = "argos-%s-%s" % (p["from"], p["to"])

        if any(m["id"] == mid for m in models.list("mt")):
            continue

        tmp = paths.temp / (mid + ".argosmodel")

        with netguard.temporarily_online("download " + p["url"]):
            download_file(p["url"], tmp, progress=progress, cancel=cancel)

        out.append(
            models.import_model(
                "mt",
                tmp,
                mid,
                license="see package metadata (Argos Open Tech)",
                source=p["url"]
            )
        )

        tmp.unlink(missing_ok=True)

    return out


# ---- speaker (diarization) models: pyannote segmentation +
# a speaker-embedding model, from the sherpa-onnx releases ----

SPK_BASE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download"

SPK_SEGMENTATION = (
    "speaker-segmentation-models/"
    "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)

SPK_EMBEDDINGS = {
    "titanet-small": {
        "file": "speaker-recongition-models/nemo_en_titanet_small.onnx",
        "approx_mb": 38,
        "desc": (
            "NeMo TitaNet small (English voices) - "
            "the default, stable on a 2-speaker test clip"
        )
    },
    "campplus-en": {
        "file": (
            "speaker-recongition-models/"
            "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"
        ),
        "approx_mb": 28,
        "desc": (
            "3D-Speaker CAM++ (English voices) - "
            "smaller, less stable in my test"
        )
    },
}  # tag 'speaker-recongition-models' is spelled that way in the upstream release


def _extract_segmentation(archive: Path, dest: Path) -> None:
    """Write model.onnx from the tar.bz2 to `dest`
    WITHOUT extracting anything else (no path traversal possible).
    """
    import tarfile

    with tarfile.open(archive, "r:bz2") as tf:
        member = next(
            (
                m for m in tf.getmembers()
                if m.isfile() and Path(m.name).name == "model.onnx"
            ),
            None
        )

        if member is None:
            raise ModelIntegrityError(
                "The segmentation archive does not contain model.onnx"
            )

        data = tf.extractfile(member).read()

    dest.write_bytes(data)


def download_speaker_package(
    paths,
    models,
    key: str = "titanet-small",
    progress=None,
    cancel=None
) -> dict:
    if key not in SPK_EMBEDDINGS:
        raise OVTError(
            "unknown speaker model: %s (known: %s)"
            % (key, ", ".join(SPK_EMBEDDINGS))
        )

    import shutil

    tmp = paths.temp / ("spk-" + key)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    try:
        emb = SPK_EMBEDDINGS[key]

        with netguard.temporarily_online(
            "download speaker models (" + key + ")"
        ):
            arc = download_file(
                "%s/%s" % (SPK_BASE_URL, SPK_SEGMENTATION),
                tmp / "seg.tar.bz2",
                progress=progress,
                cancel=cancel
            )

            download_file(
                "%s/%s" % (SPK_BASE_URL, emb["file"]),
                tmp / "embedding.onnx",
                progress=progress,
                cancel=cancel
            )

        _extract_segmentation(arc, tmp / "segmentation.onnx")
        (tmp / "seg.tar.bz2").unlink()

        return models.import_model(
            "spk",
            tmp,
            "spk-" + key,
            license="see sherpa-onnx / model cards",
            source="%s/%s" % (SPK_BASE_URL, emb["file"])
        )

    finally:
        shutil.rmtree(tmp, ignore_errors=True)
