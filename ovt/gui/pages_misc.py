import socket
import shutil
from pathlib import Path

from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)

from .. import netguard
from ..errors import OfflineViolation


def read_tail(path: Path, limit=200_000) -> str:
    try:
        data = Path(path).read_bytes()[-limit:]
        return data.decode("utf-8", "replace")
    except OSError:
        return ""


class LogsPage(QWidget):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.file = ctx.paths.logs / "errors.log"
        t = ctx.t
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        self.b_ref, self.b_exp, self.b_clr = QPushButton(t("l_refresh")), QPushButton(t("l_export")), QPushButton(t("l_clear"))
        self.b_diag = QPushButton(t("l_diag"))
        for b in (self.b_diag, self.b_ref, self.b_exp, self.b_clr):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        self.view = QPlainTextEdit(); self.view.setReadOnly(True)
        lay.addWidget(self.view, 1)
        self.b_ref.clicked.connect(self.refresh)
        self.b_diag.clicked.connect(self.run_diag)
        self.b_exp.clicked.connect(self.export)
        self.b_clr.clicked.connect(self.clear)
        self.refresh()

    def refresh(self):
        self.view.setPlainText(read_tail(self.file) or self.ctx.t("l_empty"))

    def run_diag(self):
        from ..diagnostics import format_report, run_diagnostics
        rep = format_report(run_diagnostics(self.ctx.paths))
        self.view.setPlainText(rep)
        try:
            (self.ctx.paths.logs / "diagnostics.txt").write_text(rep, encoding="utf-8")
        except OSError:
            pass
        return rep

    def export_to(self, dest):
        shutil.copyfile(self.file, dest) if self.file.exists() else Path(dest).write_text("", encoding="utf-8")

    def export(self):
        f, _ = QFileDialog.getSaveFileName(self, self.ctx.t("l_export"), "ovt-errors.log", "Log (*.log *.txt)")
        if f:
            self.export_to(f)

    def clear(self):
        self.file.unlink(missing_ok=True)
        self.refresh()


class PrivacyPage(QWidget):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        t = ctx.t
        lay = QVBoxLayout(self)
        txt = QLabel(t("p_text")); txt.setWordWrap(True)
        self.lbl_off = QLabel()
        lay.addWidget(txt); lay.addWidget(self.lbl_off)
        row = QHBoxLayout()
        self.b_test = QPushButton(t("p_test")); self.b_jobs = QPushButton(t("p_clear_jobs")); self.b_tmp = QPushButton(t("p_clear_temp"))
        for b in (self.b_test, self.b_jobs, self.b_tmp):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        self.res = QLabel(""); lay.addWidget(self.res)
        hint = QLabel(t("p_jobs_note")); hint.setObjectName("hint"); hint.setWordWrap(True); lay.addWidget(hint)
        lay.addWidget(QLabel(t("p_net_log")))
        self.view = QPlainTextEdit(); self.view.setReadOnly(True); lay.addWidget(self.view, 1)
        self.b_test.clicked.connect(self.test_block)
        self.b_jobs.clicked.connect(lambda: self.wipe(ctx.paths.temp / "jobs"))
        self.b_tmp.clicked.connect(lambda: self.wipe(ctx.paths.temp))
        self.refresh()

    def refresh(self):
        off = netguard.is_offline()
        self.lbl_off.setText(self.ctx.t("offline_on") if off else self.ctx.t("offline_off"))
        self.lbl_off.setObjectName("offlineOn" if off else "offlineOff")
        self.lbl_off.style().unpolish(self.lbl_off); self.lbl_off.style().polish(self.lbl_off)
        self.view.setPlainText(read_tail(self.ctx.paths.logs / "network.log"))

    def test_block(self) -> bool:
        """Really tries to open an outside connection; returns True when it was blocked."""
        try:
            socket.create_connection(("93.184.216.34", 80), timeout=2).close()
            self.res.setText(self.ctx.t("p_test_open")); blocked = False
        except OfflineViolation:
            self.res.setText(self.ctx.t("p_test_blocked")); blocked = True
        except OSError:
            self.res.setText(self.ctx.t("p_test_blocked") + " (no route)"); blocked = True
        self.refresh()
        return blocked

    def wipe(self, d: Path):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
        self.res.setText(self.ctx.t("p_done"))


class DubbingPage(QWidget):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        h = QLabel(ctx.t("dub_title")); h.setStyleSheet("font-size:18px;font-weight:bold;")
        p = QLabel(ctx.t("dub_text")); p.setWordWrap(True)
        lay.addWidget(h); lay.addWidget(p); lay.addStretch(1)
