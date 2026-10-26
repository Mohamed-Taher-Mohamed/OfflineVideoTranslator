import json
import zipfile

import pytest

from ovt import netguard
from ovt.errors import ModelIntegrityError, ModelMissingError, OfflineViolation, OVTError
from ovt.models import ModelManager, sha256_file


def make_asr_dir(d, payload=b"weights"):
    d.mkdir(parents=True)
    (d / "model.bin").write_bytes(payload)
    (d / "config.json").write_text("{}")
    (d / "tokenizer.json").write_text("{}")
    return d


def make_argos(p, f="en", t="ar"):
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("translate-%s_%s/metadata.json" % (f, t), json.dumps({"from_code": f, "to_code": t, "package_version": "1.0", "name": "x"}))
        z.writestr("translate-%s_%s/model/model.bin" % (f, t), b"zz")
    return p


def test_import_list_verify_delete(paths, tmp_path):
    mm = ModelManager(paths)
    m = mm.import_model("asr", make_asr_dir(tmp_path / "src"), "my-whisper", license="MIT")
    assert m["size_bytes"] > 0 and set(m["files"]) >= {"model.bin", "config.json", "tokenizer.json"}
    assert [x["id"] for x in mm.list("asr")] == ["my-whisper"]
    assert mm.verify("asr", "my-whisper") == []
    (paths.models / "asr" / "my-whisper" / "model.bin").write_bytes(b"corrupted!")
    assert any("size mismatch" in p or "checksum" in p for p in mm.verify("asr", "my-whisper"))
    with pytest.raises(ModelIntegrityError):
        mm.require("asr", "my-whisper")
    mm.delete("asr", "my-whisper")
    with pytest.raises(ModelMissingError):
        mm.get("asr", "my-whisper")


def test_same_size_corruption_caught_by_full_hash(paths, tmp_path):
    mm = ModelManager(paths)
    mm.import_model("asr", make_asr_dir(tmp_path / "s", b"AAAA"), "m")
    (paths.models / "asr" / "m" / "model.bin").write_bytes(b"BBBB")
    assert mm.verify("asr", "m", full=False) == []          # cheap check can't see it
    assert "checksum mismatch: model.bin" in mm.verify("asr", "m", full=True)


def test_invalid_folder_and_ids_rejected(paths, tmp_path):
    mm = ModelManager(paths)
    bad = tmp_path / "bad"; bad.mkdir(); (bad / "model.bin").write_bytes(b"x")
    with pytest.raises(ModelIntegrityError):
        mm.import_model("asr", bad, "x")
    for evil in ("../escape", "a/b", "", "x" * 100, "sp ace"):
        with pytest.raises(OVTError):
            mm.import_model("asr", make_asr_dir(tmp_path / ("s" + str(abs(hash(evil))))), evil)
    assert not (paths.models.parent / "escape").exists()


def test_failed_reimport_keeps_previous_version(paths, tmp_path):
    mm = ModelManager(paths)
    mm.import_model("asr", make_asr_dir(tmp_path / "v1", b"one"), "m", version="1")
    broken = tmp_path / "v2"; broken.mkdir(); (broken / "model.bin").write_bytes(b"two")
    with pytest.raises(ModelIntegrityError):
        mm.import_model("asr", broken, "m", version="2")
    assert mm.get("asr", "m")["version"] == "1" and mm.verify("asr", "m") == []
    assert not list((paths.models / "asr").glob("*.importing"))


def test_expected_checksum_enforced(paths, tmp_path):
    mm = ModelManager(paths)
    src = make_asr_dir(tmp_path / "s")
    good = sha256_file(src / "model.bin")
    mm.import_model("asr", src, "ok", expected_sha256={"model.bin": good})
    with pytest.raises(ModelIntegrityError):
        mm.import_model("asr", src, "no", expected_sha256={"model.bin": "0" * 64})
    assert not (paths.models / "asr" / "no").exists()


def test_import_argos_package_reads_metadata(paths, tmp_path):
    mm = ModelManager(paths)
    m = mm.import_model("mt", make_argos(tmp_path / "t.argosmodel"), "en-ar")
    assert m["languages"] == ["en->ar"] and m["version"] == "1.0"
    with pytest.raises(OVTError):
        mm.import_model("mt", tmp_path / "notzip.argosmodel", "z")
    (tmp_path / "plain.argosmodel").write_text("not a zip")
    with pytest.raises(OVTError):
        mm.import_model("mt", tmp_path / "plain.argosmodel", "z")


def test_download_refused_while_offline(paths):
    netguard.enable_offline()
    with pytest.raises(OfflineViolation):
        ModelManager(paths).download_asr("whisper-tiny")
