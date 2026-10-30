"""Fails FAST with a readable message when part of the project was not uploaded (partial uploads are the most
common cause of confusing 'No module named ...' errors). The list is ci/required_files.txt."""
import sys
from pathlib import Path


def main(root: str = ".") -> int:
    root = Path(root)
    lines = (root / "ci" / "required_files.txt").read_text(encoding="utf-8").splitlines()
    missing = [l.strip() for l in lines if l.strip() and not (root / l.strip()).is_file()]
    if missing:
        print("MISSING PROJECT FILES (%d of %d) - upload them and run again:" % (len(missing), len(lines)))
        for m in missing:
            print("  - " + m)
        print("\nملفات ناقصة (%d): ارفعها من الحزمة الأخيرة ثم أعد التشغيل." % len(missing))
        return 1
    print("All %d project files are present." % len(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
