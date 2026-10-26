import json

import pytest

from ovt.cli import main
from test_models import make_asr_dir


def test_cli_probe_and_models_and_reset(tmp_path, media, capsys):
    home = str(tmp_path / "h")
    assert main(["--home", home, "probe", str(media["multi"])]) == 0
    assert json.loads(capsys.readouterr().out)["name"] == "فيديو اختبار 1.mkv"
    assert main(["--home", home, "models", "import", "--kind", "asr", "--id", "w", "--path", str(make_asr_dir(tmp_path / "m"))]) == 0
    assert main(["--home", home, "models", "list"]) == 0 and "w" in capsys.readouterr().out
    assert main(["--home", home, "models", "verify", "--kind", "asr", "--id", "w"]) == 0
    assert main(["--home", home, "reset-settings"]) == 0
    assert main(["--home", home, "models", "list"]) == 0 and "w" in capsys.readouterr().out   # models survive reset


def test_cli_reports_missing_model_clearly(tmp_path, media, capsys):
    rc = main(["--home", str(tmp_path / "h"), "translate", str(media["multi"]), "--asr-model", "nope", "--mt-model", "x"])
    err = capsys.readouterr().err
    assert rc == 1 and "model_missing" in err and "nope" in err


def test_cli_corrupt_file_message(tmp_path, media, capsys):
    rc = main(["--home", str(tmp_path / "h"), "probe", str(media["corrupt"])])
    assert rc == 1 and "media_unsupported" in capsys.readouterr().err


def test_damaged_asr_model_gives_clear_error_and_log(tmp_path, capsys):
    """Uses the REAL faster-whisper/CTranslate2 loader with a junk model.bin (no simulation)."""
    import subprocess
    from test_models import make_asr_dir
    vid = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=f=300:d=3", str(vid)], check=True)
    home = str(tmp_path / "h")
    main(["--home", home, "models", "import", "--kind", "asr", "--id", "junk", "--path", str(make_asr_dir(tmp_path / "m"))])
    capsys.readouterr()
    rc = main(["--home", home, "translate", str(vid), "--asr-model", "junk", "--src-lang", "en"])
    err = capsys.readouterr().err
    assert rc == 1 and "model_integrity" in err and "Traceback" not in err
    assert "Could not load the speech model" in (tmp_path / "h" / "logs" / "errors.log").read_text(encoding="utf-8")
