from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QDialog, QTableWidget, QTableWidgetItem, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget)

from .. import netguard
from ..translate.llm_backend import LLMTranslator, PRESETS, is_loopback_url
from .workers import FuncThread
import json
from pathlib import Path

from ..config import Settings
from ..fsutil import atomic_write_json


class GlossaryDialog(QDialog):
    """Edits the glossary as a two-column table and stores it as the JSON the engine reads."""

    def __init__(self, ctx, path: str = "", parent=None):
        super().__init__(parent)
        self.ctx = ctx
        t = ctx.t
        self.setWindowTitle(t("gl_title")); self.resize(560, 420)
        lay = QVBoxLayout(self)
        hint = QLabel(t("gl_hint")); hint.setObjectName("hint"); hint.setWordWrap(True); lay.addWidget(hint)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels([t("gl_term"), t("gl_repl")])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 230)
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        self.b_add, self.b_del, self.b_save = QPushButton(t("gl_add")), QPushButton(t("gl_del")), QPushButton(t("gl_save"))
        self.b_save.setObjectName("primary")
        for b in (self.b_add, self.b_del, self.b_save):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        self.b_add.clicked.connect(lambda: self.add_row())
        self.b_del.clicked.connect(self.del_row)
        self.b_save.clicked.connect(self.accept)
        if path and Path(path).is_file():
            self.load(path)
        else:
            self.add_row()

    def add_row(self, term="", repl=""):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(term))
        self.table.setItem(r, 1, QTableWidgetItem(repl))

    def del_row(self):
        r = self.table.currentRow()
        if r >= 0:
            self.table.removeRow(r)

    def load(self, path):
        d = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        for k in d.get("keep", []):
            self.add_row(k, "")
        for k, v in d.get("map", {}).items():
            self.add_row(k, v)
        if not self.table.rowCount():
            self.add_row()

    def to_json(self) -> dict:
        keep, mp = [], {}
        for r in range(self.table.rowCount()):
            term = (self.table.item(r, 0).text() if self.table.item(r, 0) else "").strip()
            repl = (self.table.item(r, 1).text() if self.table.item(r, 1) else "").strip()
            if term:
                if repl:
                    mp[term] = repl
                else:
                    keep.append(term)
        return {"keep": keep, "map": mp}

    def save(self, path) -> Path:
        atomic_write_json(path, self.to_json())
        return Path(path)


class SettingsPage(QWidget):
    applied = Signal(bool)           # True if UI must be rebuilt (language/theme changed)

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        t = ctx.t
        s = ctx.settings
        outer = QVBoxLayout(self)
        inner = QWidget()
        lay = QVBoxLayout(inner)

        g = QGroupBox(t("grp_general")); f = QFormLayout(g)
        self.cb_offline = QCheckBox(t("set_offline")); self.cb_offline.setChecked(s.offline_mode)
        self.cb_ui = QComboBox(); self.cb_ui.addItem("العربية", "ar"); self.cb_ui.addItem("English", "en")
        self.cb_ui.setCurrentIndex(self.cb_ui.findData(s.ui_language))
        self.cb_theme = QComboBox(); self.cb_theme.addItem(t("theme_light"), "light"); self.cb_theme.addItem(t("theme_dark"), "dark")
        self.cb_theme.setCurrentIndex(self.cb_theme.findData(s.theme))
        f.addRow(self.cb_offline); f.addRow(t("set_ui_lang"), self.cb_ui); f.addRow(t("set_theme"), self.cb_theme)
        lay.addWidget(g)

        g = QGroupBox(t("grp_engine")); f = QFormLayout(g)
        self.cb_engine = QComboBox(); self.cb_engine.addItem(t("eng_nmt"), "nmt"); self.cb_engine.addItem(t("eng_llm"), "llm")
        self.cb_engine.setCurrentIndex(self.cb_engine.findData(s.mt_engine))
        self.cb_preset = QComboBox()
        for k, (label, url) in PRESETS.items():
            self.cb_preset.addItem(label, k)
        self.ed_llm_url = QLineEdit(s.llm_base_url)
        self.cb_llm_model = QComboBox(); self.cb_llm_model.setEditable(True); self.cb_llm_model.setCurrentText(s.llm_model)
        self.b_llm_refresh = QPushButton(t("llm_refresh"))
        hm = QHBoxLayout(); hm.addWidget(self.cb_llm_model, 1); hm.addWidget(self.b_llm_refresh)
        self.ed_llm_key = QLineEdit(s.llm_api_key); self.ed_llm_key.setEchoMode(QLineEdit.Password)
        self.cb_cloud = QCheckBox(t("llm_cloud")); self.cb_cloud.setChecked(s.llm_cloud_ok)
        self.b_llm_test = QPushButton(t("llm_test"))
        self.lbl_llm = QLabel(t("llm_hint")); self.lbl_llm.setObjectName("hint"); self.lbl_llm.setWordWrap(True)
        f.addRow(t("set_mt"), self.cb_engine); f.addRow(t("llm_preset"), self.cb_preset); f.addRow(t("llm_url"), self.ed_llm_url)
        f.addRow(t("llm_model"), hm); f.addRow(t("llm_key"), self.ed_llm_key); f.addRow(self.cb_cloud)
        f.addRow(self.b_llm_test); f.addRow(self.lbl_llm)
        self.cb_preset.currentIndexChanged.connect(self.on_preset)
        self.cb_engine.currentIndexChanged.connect(self.update_llm_enabled)
        self.ed_llm_url.textChanged.connect(self.update_llm_enabled)
        self.b_llm_refresh.clicked.connect(self.refresh_llm_models)
        self.b_llm_test.clicked.connect(self.test_llm)
        self.llm_job = None
        self.update_llm_enabled()
        lay.addWidget(g)

        g = QGroupBox(t("grp_models")); f = QFormLayout(g)
        self.cb_asr, self.cb_mt = QComboBox(), QComboBox()
        self.fill_models()
        f.addRow(t("set_asr"), self.cb_asr); f.addRow(t("set_mt"), self.cb_mt)
        lay.addWidget(g)

        g = QGroupBox(t("grp_output")); f = QFormLayout(g)
        self.ed_out = QLineEdit(s.output_dir); self.ed_out.setPlaceholderText(t("set_outdir_same"))
        b = QPushButton(t("set_choose")); b.clicked.connect(self.pick_out)
        h = QHBoxLayout(); h.addWidget(self.ed_out, 1); h.addWidget(b)
        self.cb_txt = QCheckBox(t("set_txt")); self.cb_txt.setChecked(s.write_txt)
        self.cb_src = QCheckBox(t("set_srcsrt")); self.cb_src.setChecked(s.write_source_srt)
        self.ed_gl = QLineEdit(s.glossary_path)
        b2 = QPushButton(t("set_choose")); b2.clicked.connect(self.pick_gl)
        b3 = QPushButton(t("gl_edit")); b3.clicked.connect(self.edit_glossary)
        h2 = QHBoxLayout(); h2.addWidget(self.ed_gl, 1); h2.addWidget(b2); h2.addWidget(b3)
        self.cb_ctx = QCheckBox(t("set_context")); self.cb_ctx.setChecked(s.context_merge)
        f.addRow(t("set_outdir"), h); f.addRow(self.cb_txt); f.addRow(self.cb_src); f.addRow(t("set_glossary"), h2); f.addRow(self.cb_ctx)
        lay.addWidget(g)

        g = QGroupBox(t("grp_sub")); f = QFormLayout(g)
        self.sp_chars = QSpinBox(); self.sp_chars.setRange(15, 100); self.sp_chars.setValue(s.max_chars_per_line)
        self.sp_lines = QSpinBox(); self.sp_lines.setRange(1, 4); self.sp_lines.setValue(s.max_lines)
        self.sp_min = QDoubleSpinBox(); self.sp_min.setRange(0.2, 5); self.sp_min.setSingleStep(0.1); self.sp_min.setValue(s.min_cue_seconds)
        self.cb_split = QCheckBox(t("set_split")); self.cb_split.setChecked(s.smart_split)
        self.sp_pause = QDoubleSpinBox(); self.sp_pause.setRange(0.15, 3.0); self.sp_pause.setSingleStep(0.05); self.sp_pause.setValue(s.split_pause)
        self.cb_bom = QCheckBox(t("set_bom")); self.cb_bom.setChecked(s.utf8_bom)
        self.cb_rtl = QCheckBox(t("set_rtl")); self.cb_rtl.setChecked(s.rtl_marks)
        f.addRow(t("set_chars"), self.sp_chars); f.addRow(t("set_lines"), self.sp_lines); f.addRow(t("set_min_dur"), self.sp_min)
        f.addRow(self.cb_split); f.addRow(t("set_pause"), self.sp_pause); f.addRow(self.cb_bom); f.addRow(self.cb_rtl)
        lay.addWidget(g)

        g = QGroupBox(t("grp_perf")); f = QFormLayout(g)
        self.sp_chunk = QSpinBox(); self.sp_chunk.setRange(30, 1800); self.sp_chunk.setSingleStep(30); self.sp_chunk.setValue(s.chunk_seconds)
        self.cb_dev = QComboBox()
        for d in ("auto", "cpu", "cuda"):
            self.cb_dev.addItem(d, d)
        self.cb_dev.setCurrentIndex(self.cb_dev.findData(s.device))
        self.cb_keep = QCheckBox(t("set_keep_tmp")); self.cb_keep.setChecked(s.keep_temp)
        self.cb_sens = QComboBox(); self.cb_sens.addItem(t("sens_normal"), "normal"); self.cb_sens.addItem(t("sens_high"), "high")
        self.cb_sens.setCurrentIndex(max(0, self.cb_sens.findData(s.speech_sensitivity)))
        self.cb_mixed = QCheckBox(t("set_mixed")); self.cb_mixed.setChecked(s.mixed_languages)
        self.cb_beam = QComboBox()
        for v in (1, 3, 5):
            self.cb_beam.addItem(t("beam_%d" % v), v)
        self.cb_beam.setCurrentIndex(max(0, self.cb_beam.findData(s.beam_size)))
        f.addRow(t("set_beam"), self.cb_beam); f.addRow(t("set_sens"), self.cb_sens); f.addRow(self.cb_mixed); f.addRow(t("set_chunk"), self.sp_chunk); f.addRow(t("set_device"), self.cb_dev); f.addRow(self.cb_keep)
        lay.addWidget(g)

        row = QHBoxLayout()
        self.b_save = QPushButton(t("set_save")); self.b_save.setObjectName("primary")
        self.b_reset = QPushButton(t("set_reset"))
        row.addWidget(self.b_save); row.addWidget(self.b_reset); row.addStretch(1)
        lay.addLayout(row)
        self.msg = QLabel(""); self.msg.setObjectName("hint")
        lay.addWidget(self.msg); lay.addStretch(1)
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setWidget(inner)
        outer.addWidget(sc)
        self.b_save.clicked.connect(self.save)
        self.b_reset.clicked.connect(self.reset)

    # ---- LLM engine helpers ----
    def on_preset(self):
        url = PRESETS[self.cb_preset.currentData()][1]
        if url:
            self.ed_llm_url.setText(url)

    def update_llm_enabled(self):
        on = self.cb_engine.currentData() == "llm"
        cloud = not is_loopback_url(self.ed_llm_url.text().strip() or "http://127.0.0.1")
        for w in (self.cb_preset, self.ed_llm_url, self.cb_llm_model, self.b_llm_refresh, self.b_llm_test):
            w.setEnabled(on)
        self.ed_llm_key.setEnabled(on and cloud)
        self.cb_cloud.setEnabled(on and cloud)

    def make_llm(self) -> LLMTranslator:
        return LLMTranslator(self.ed_llm_url.text().strip(), self.cb_llm_model.currentText().strip(),
                             self.ed_llm_key.text().strip(), self.cb_cloud.isChecked(), timeout=300)

    def _run_llm(self, fn, ok):
        if self.llm_job:
            return
        self.msg.setText(self.ctx.t("llm_testing"))
        self.llm_job = FuncThread(fn)
        self.llm_job.ok.connect(lambda r: (setattr(self, "llm_job", None), ok(r)))
        self.llm_job.err.connect(lambda code, m: (setattr(self, "llm_job", None), self.msg.setText(m)))
        self.llm_job.start()

    def refresh_llm_models(self):
        llm = self.make_llm()

        def done(models):
            cur = self.cb_llm_model.currentText()
            self.cb_llm_model.clear(); self.cb_llm_model.addItems(models); self.cb_llm_model.setCurrentText(cur or (models[0] if models else ""))
            self.msg.setText(", ".join(models[:8]))
        self._run_llm(llm.list_models, done)

    def test_llm(self):
        llm = self.make_llm()
        self._run_llm(llm.check, lambda out: self.msg.setText("%s: %s" % (self.ctx.t("llm_ok"), out)))

    def fill_models(self):
        s, t = self.ctx.settings, self.ctx.t
        self.cb_asr.clear(); self.cb_mt.clear()
        self.cb_asr.addItem("—", "")
        for m in self.ctx.models.list("asr"):
            self.cb_asr.addItem(m["id"], m["id"])
        self.cb_mt.addItem(t("set_all_mt"), "")
        for m in self.ctx.models.list("mt"):
            self.cb_mt.addItem(m["id"], m["id"])
        self.cb_asr.setCurrentIndex(max(0, self.cb_asr.findData(s.asr_model_id)))
        self.cb_mt.setCurrentIndex(max(0, self.cb_mt.findData(s.mt_model_id)))

    def pick_out(self):
        d = QFileDialog.getExistingDirectory(self, self.ctx.t("set_outdir"))
        if d:
            self.ed_out.setText(d)

    def pick_gl(self):
        f, _ = QFileDialog.getOpenFileName(self, self.ctx.t("set_glossary"), "", "JSON (*.json)")
        if f:
            self.ed_gl.setText(f)

    def edit_glossary(self):
        dlg = GlossaryDialog(self.ctx, self.ed_gl.text().strip(), self)
        if dlg.exec() == QDialog.Accepted:
            target = self.ed_gl.text().strip() or str(self.ctx.paths.settings / "glossary.json")
            dlg.save(target)
            self.ed_gl.setText(target)

    def save(self, confirm=True):
        s, paths = self.ctx.settings, self.ctx.paths
        if s.offline_mode and not self.cb_offline.isChecked() and confirm:
            if QMessageBox.question(self, "", self.ctx.t("offline_warn_confirm")) != QMessageBox.Yes:
                self.cb_offline.setChecked(True)
                return
        rebuild = (s.ui_language != self.cb_ui.currentData()) or (s.theme != self.cb_theme.currentData())
        s.offline_mode = self.cb_offline.isChecked()
        s.ui_language, s.theme = self.cb_ui.currentData(), self.cb_theme.currentData()
        s.asr_model_id, s.mt_model_id = self.cb_asr.currentData() or "", self.cb_mt.currentData() or ""
        s.output_dir, s.write_txt, s.write_source_srt = self.ed_out.text().strip(), self.cb_txt.isChecked(), self.cb_src.isChecked()
        s.glossary_path, s.context_merge = self.ed_gl.text().strip(), self.cb_ctx.isChecked()
        s.max_chars_per_line, s.max_lines, s.min_cue_seconds = self.sp_chars.value(), self.sp_lines.value(), self.sp_min.value()
        s.utf8_bom, s.rtl_marks = self.cb_bom.isChecked(), self.cb_rtl.isChecked()
        s.chunk_seconds, s.device, s.keep_temp = self.sp_chunk.value(), self.cb_dev.currentData(), self.cb_keep.isChecked()
        s.beam_size = self.cb_beam.currentData()
        s.smart_split, s.split_pause = self.cb_split.isChecked(), self.sp_pause.value()
        s.speech_sensitivity, s.mixed_languages = self.cb_sens.currentData(), self.cb_mixed.isChecked()
        s.mt_engine, s.llm_base_url = self.cb_engine.currentData(), self.ed_llm_url.text().strip()
        s.llm_model, s.llm_api_key, s.llm_cloud_ok = self.cb_llm_model.currentText().strip(), self.ed_llm_key.text().strip(), self.cb_cloud.isChecked()
        s.save(paths)
        if s.offline_mode:
            netguard.enable_offline(paths.logs / "network.log")
        else:
            netguard.disable_offline()
        self.msg.setText(self.ctx.t("set_saved"))
        self.applied.emit(rebuild)

    def reset(self):
        Settings.reset(self.ctx.paths)
        fresh = Settings()
        for k, v in fresh.__dict__.items():
            setattr(self.ctx.settings, k, v)
        netguard.enable_offline(self.ctx.paths.logs / "network.log")
        self.applied.emit(True)
