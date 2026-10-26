"""Background threads. The GUI thread never runs ffmpeg/ASR/MT itself."""
import threading

from PySide6.QtCore import QThread, Signal

from ..errors import CancelledError, OVTError


class JobThread(QThread):
    progress = Signal(object)          # ovt.pipeline.Progress
    done = Signal(object)              # ovt.pipeline.JobResult
    failed = Signal(str, str)          # code, message
    cancelled = Signal()

    def __init__(self, options, builder, parent=None):
        super().__init__(parent)
        self.options, self.builder = options, builder
        self.cancel_event = threading.Event()

    def request_cancel(self):
        self.cancel_event.set()

    def run(self):
        try:
            pl = self.builder(self.options, self.progress.emit, self.cancel_event)
            self.done.emit(pl.run(self.options))
        except CancelledError:
            self.cancelled.emit()
        except OVTError as e:
            self.failed.emit(e.code, str(e))
        except Exception as e:            # never let a thread die silently
            import traceback
            self.failed.emit("internal", "%s: %s" % (type(e).__name__, e))
            self.error_trace = traceback.format_exc()


class FuncThread(QThread):
    """Run any callable (model import / verify / download) off the GUI thread."""
    ok = Signal(object)
    err = Signal(str, str)
    tick = Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn

    def run(self):
        try:
            self.ok.emit(self.fn())
        except OVTError as e:
            self.err.emit(e.code, str(e))
        except Exception as e:
            self.err.emit("internal", "%s: %s" % (type(e).__name__, e))
