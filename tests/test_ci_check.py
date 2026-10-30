import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(root):
    return subprocess.run([sys.executable, str(ROOT / "ci" / "check_files.py"), str(root)], capture_output=True)


def test_the_real_project_has_every_listed_file():
    r = run(ROOT)
    assert r.returncode == 0, r.stdout.decode("utf-8", "replace")


def test_missing_files_are_named_clearly(tmp_path):
    (tmp_path / "ci").mkdir()
    (tmp_path / "ci" / "required_files.txt").write_text("ovt/a.py\novt/translate/llm_backend.py\n", encoding="utf-8")
    (tmp_path / "ovt").mkdir(); (tmp_path / "ovt" / "a.py").write_text("x")
    r = run(tmp_path)
    out = r.stdout.decode("utf-8", "replace")
    assert r.returncode == 1 and "ovt/translate/llm_backend.py" in out and "ovt/a.py" not in out.split("MISSING")[1].split("\n\n")[0].replace("ovt/translate", "")
