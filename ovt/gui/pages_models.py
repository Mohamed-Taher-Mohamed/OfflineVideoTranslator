from PySide6.QtWidgets import (QMenu, QFileDialog, QHBoxLayout, QInputDialog, QLabel, QMessageBox, QPushButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget, QAbstractItemView)

from .. import downloader, netguard
from .i18n import LANG_NAMES
from ..errors import OVTError
from ..models import ASR_CATALOG
from .workers import FuncThread


class ModelsPage(QWidget):
    changed = None

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.job = None
        t = ctx.t
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        self.b = {}
        for k in ("m_import_asr", "m_import_mt_file", "m_import_mt_dir", "m_verify", "m_delete", "m_set_default", "m_download"):
            self.b[k] = QPushButton(t(k))
            row.addWidget(self.b[k])
        row.addStretch(1)
        lay.addLayout(row)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels([t(k) for k in ("m_kind", "m_id", "m_version", "m_size", "m_langs", "m_license", "m_source", "m_default")])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self.table, 1)
        self.status = QLabel(t("m_note"))
        self.status.setObjectName("hint")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.b["m_import_asr"].clicked.connect(self.import_asr)
        self.b["m_import_mt_file"].clicked.connect(self.import_mt_file)
        self.b["m_import_mt_dir"].clicked.connect(self.import_mt_dir)
        self.b["m_verify"].clicked.connect(self.verify)
        self.b["m_delete"].clicked.connect(self.delete)
        self.b["m_set_default"].clicked.connect(self.set_default)
        self.b["m_download"].clicked.connect(self.download)
        self.refresh()

    def selected(self):
        r = self.table.currentRow()
        if r < 0:
            return None
        return self.table.item(r, 0).text(), self.table.item(r, 1).text()

    def refresh(self):
        s = self.ctx.settings
        ms = self.ctx.models.list()
        self.table.setRowCount(len(ms))
        for r, m in enumerate(ms):
            default = "✔" if (m["kind"] == "asr" and m["id"] == s.asr_model_id) or (m["kind"] == "mt" and m["id"] == s.mt_model_id) else ""
            vals = [m["kind"], m["id"], m["version"] or "-", "%.1f MB" % (m["size_bytes"] / 2**20), ", ".join(m["languages"]),
                    m["license"], m["source"], default]
            for c, v in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem(v))
        self.b["m_download"].setEnabled(True)

    # ---- running long operations off the GUI thread ----
    def run_bg(self, fn, on_ok):
        if self.job:
            return
        self.status.setText(self.ctx.t("m_working"))
        self.job = FuncThread(fn)
        self.job.ok.connect(lambda r: (setattr(self, "job", None), on_ok(r)))
        self.job.err.connect(self.on_err)
        self.job.start()

    def on_err(self, code, msg):
        self.job = None
        self.status.setText("[%s] %s" % (code, msg))
        QMessageBox.warning(self, "Error", msg)

    def ask_id(self, default=""):
        mid, ok = QInputDialog.getText(self, self.ctx.t("m_id"), self.ctx.t("m_id_prompt"), text=default)
        if not ok or not mid.strip():
            return None, ""
        lic, _ = QInputDialog.getText(self, self.ctx.t("m_license"), self.ctx.t("m_license_prompt"))
        return mid.strip(), lic

    def do_import(self, kind, path, mid, lic):
        def done(m):
            self.status.setText("%s: %s" % (self.ctx.t("m_import_ok"), m["id"]))
            s = self.ctx.settings
            if kind == "asr" and not s.asr_model_id:
                s.asr_model_id = m["id"]; s.save(self.ctx.paths)
            if kind == "mt" and not s.mt_model_id:
                pass                    # empty mt_model_id = "use all installed", which is the sane default
            self.refresh()
        self.run_bg(lambda: self.ctx.models.import_model(kind, path, mid, license=lic), done)

    def import_asr(self):
        d = QFileDialog.getExistingDirectory(self, self.ctx.t("m_import_asr"))
        if d:
            mid, lic = self.ask_id(d.replace("\\", "/").rstrip("/").split("/")[-1])
            if mid:
                self.do_import("asr", d, mid, lic)

    def import_mt_file(self):
        f, _ = QFileDialog.getOpenFileName(self, self.ctx.t("m_import_mt_file"), "", "Argos (*.argosmodel)")
        if f:
            mid, lic = self.ask_id(f.replace("\\", "/").split("/")[-1].rsplit(".", 1)[0])
            if mid:
                self.do_import("mt", f, mid, lic)

    def import_mt_dir(self):
        d = QFileDialog.getExistingDirectory(self, self.ctx.t("m_import_mt_dir"))
        if d:
            mid, lic = self.ask_id(d.replace("\\", "/").rstrip("/").split("/")[-1])
            if mid:
                self.do_import("mt", d, mid, lic)

    def verify(self):
        sel = self.selected()
        if sel:
            kind, mid = sel
            self.run_bg(lambda: self.ctx.models.verify(kind, mid),
                        lambda pr: self.status.setText("%s: %s" % (mid, self.ctx.t("m_verify_ok") if not pr else "; ".join(pr))))

    def delete(self):
        sel = self.selected()
        if sel and QMessageBox.question(self, sel[1], self.ctx.t("m_confirm_delete")) == QMessageBox.Yes:
            self.ctx.models.delete(*sel)
            s = self.ctx.settings
            if sel[1] in (s.asr_model_id, s.mt_model_id):
                s.asr_model_id = "" if sel[1] == s.asr_model_id else s.asr_model_id
                s.mt_model_id = "" if sel[1] == s.mt_model_id else s.mt_model_id
                s.save(self.ctx.paths)
            self.refresh()

    def set_default(self):
        sel = self.selected()
        if sel:
            s = self.ctx.settings
            if sel[0] == "asr":
                s.asr_model_id = sel[1]
            else:
                s.mt_model_id = sel[1]
            s.save(self.ctx.paths)
            self.refresh()

    # ---- explicit, user-approved downloads ----
    def confirm_online(self, what: str) -> bool:
        return QMessageBox.question(self, self.ctx.t("m_download"), self.ctx.t("m_dl_confirm") % what) == QMessageBox.Yes

    def download(self):
        menu = QMenu(self)
        t = self.ctx.t
        sub = menu.addMenu(t("m_dl_asr"))
        for key, info in ASR_CATALOG.items():
            sub.addAction("%s  (~%d MB)" % (key, info["approx_mb"]), lambda k=key: self.download_asr(k))
        sub2 = menu.addMenu(t("m_dl_mt"))
        idx = 0 if self.ctx.i18n.lang == "ar" else 1
        for code, names in LANG_NAMES.items():
            if code != "ar":
                sub2.addAction("%s (%s)" % (names[idx], code), lambda c=code: self.download_mt(c))
        menu.exec(self.b["m_download"].mapToGlobal(self.b["m_download"].rect().bottomLeft()))

    def download_asr(self, key):
        if self.job or not self.confirm_online("%s (~%d MB)" % (key, ASR_CATALOG[key]["approx_mb"])):
            return
        def ok(m):
            s = self.ctx.settings
            if not s.asr_model_id:
                s.asr_model_id = m["id"]; s.save(self.ctx.paths)
            self.status.setText(self.ctx.t("m_dl_done") + ": " + m["id"]); self.refresh()
        self.run_bg(lambda: self.ctx.models.download_asr(key, allow_network=True), ok)

    def download_mt(self, lang):
        if self.job:
            return
        self.status.setText(self.ctx.t("m_working"))

        def work():
            idx = downloader.list_argos_packages()
            route = downloader.argos_route(idx, lang, "ar")
            if not route:
                raise OVTError("Argos has no translation chain from '%s' to Arabic." % lang)
            return route, idx
        self.job = FuncThread(work)
        self.job.ok.connect(lambda r: self.after_route(lang, *r))
        self.job.err.connect(self.on_err)
        self.job.start()

    def after_route(self, lang, route, idx):
        self.job = None
        names = " + ".join("%s→%s" % (p["from"], p["to"]) for p in route)
        what = names + ("\n" + self.ctx.t("m_dl_pivot") if len(route) > 1 else "")
        if not self.confirm_online(what):
            self.status.setText(""); return
        paths, mm = self.ctx.paths, self.ctx.models

        def work():
            def prog(a, b):
                self.job.tick.emit("%s: %.0f%%" % (names, 100.0 * a / b if b else 0))
            return downloader.download_argos(paths, mm, lang, "ar", index=idx, progress=prog)

        def ok(got):
            self.status.setText(self.ctx.t("m_dl_done") if got else self.ctx.t("m_dl_nothing")); self.refresh()
        self.job = FuncThread(work)
        self.job.tick.connect(self.status.setText)
        self.job.ok.connect(lambda r: (setattr(self, "job", None), ok(r)))
        self.job.err.connect(self.on_err)
        self.job.start()
