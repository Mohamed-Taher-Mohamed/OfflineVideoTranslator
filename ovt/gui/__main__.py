import os
import sys

from PySide6.QtCore import QTimer

from PySide6.QtWidgets import QApplication

from .. import netguard
from ..config import Paths, Settings
from ..media import FFmpeg
from ..models import ModelManager
from ..cli import build_pipeline
from .context import Ctx
from .main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    paths = Paths().ensure()
    settings = Settings.load(paths)
    if settings.offline_mode:
        netguard.enable_offline(paths.logs / "network.log")
    ctx = Ctx(paths, settings, ModelManager(paths),
              lambda opts, prog, cancel: build_pipeline(paths, settings, opts, prog, cancel),
              lambda: FFmpeg(paths.engines))
    w = MainWindow(ctx)
    w.show()
    quit_ms = os.environ.get("OVT_SMOKE_QUIT_MS")       # smoke tests: start the real GUI, then close it by itself
    if quit_ms:
        QTimer.singleShot(int(quit_ms), app.quit)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
