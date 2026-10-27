"""Single entry point for the packaged program.
  OfflineVideoTranslator.exe            -> GUI
  OfflineVideoTranslator.exe --diag     -> system check, saved to logs/diagnostics.txt (works from the windowed exe)
  ovt-cli.exe <command> ...             -> the test CLI with a console (same code)"""
import sys


def run():
    from ovt.cli import force_utf8_console
    force_utf8_console()
    if "--diag" in sys.argv:
        from ovt.config import Paths
        from ovt.diagnostics import format_report, run_diagnostics
        paths = Paths().ensure()
        rep = format_report(run_diagnostics(paths))
        (paths.logs / "diagnostics.txt").write_text(rep, encoding="utf-8")
        try:
            print(rep)
        except Exception:
            pass
        return 0 if "0 failed" in rep.splitlines()[-1] else 4
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        from ovt.cli import main
        return main(sys.argv[2:])
    from ovt.gui.__main__ import main
    return main()


if __name__ == "__main__":
    sys.exit(run())
