import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHeaderView
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QMessageBox, QProgressBar, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget, QPlainTextEdit, QDialog)

from ..errors import OVTError
from ..media import SUPPORTED_EXT
from ..pipeline import JobOptions
from .context import QueueItem
from .i18n import LANG_NAMES
from .workers import JobThread

COLS = ["col_name", "col_size", "col_duration", "col_tracks", "col_lang", "col_status", "col_progress", "col_model"]


LRM = "\u200e"          # keeps Latin/number runs in their natural order inside RTL cells


def ltr(text):
    return LRM + text + LRM


def fmt_size(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return "%.1f %s" % (n, u) if u != "B" else "%d B" % n
        n /= 1024.0


def fmt_dur(s):
    s = int(s)
    return "%d:%02d:%02d" % (s // 3600, s % 3600 // 60, s % 60)


def open_folder(path):
    path = str(path)
    if sys.platform == "win32":
        os.startfile(path)          # noqa
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


class HomePage(QWidget):
    open_in_editor = Signal(str)
    status_message = Signal(str)

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.thread = None
        self.current = None            # QueueItem being processed
        self.stop_reason = ""          # "pause" | "cancel"
        self.setAcceptDrops(True)
        t = ctx.t
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        self.btn = {}
        for key, primary in (("add_videos", False), ("remove", False), ("start_all", True), ("pause", False),
                             ("resume", False), ("cancel", False), ("retry", False), ("open_folder", False),
                             ("report", False), ("edit_result", False), ("export_video", False)):
            b = QPushButton(t(key))
            if primary:
                b.setObjectName("primary")
            self.btn[key] = b
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels([t(c) for c in COLS])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setSectionResizeMode(6, QHeaderView.Fixed)
        self.table.setColumnWidth(6, 150)
        lay.addWidget(self.table, 1)

        self.detail = QGroupBox(t("detail_title"))
        g = QGridLayout(self.detail)
        self.cb_track, self.cb_mode, self.cb_sub, self.cb_lang = QComboBox(), QComboBox(), QComboBox(), QComboBox()
        self.btn_subfile = QPushButton(t("choose_file"))
        self.lbl_subfile = QLabel("")
        self.lbl_subfile.setObjectName("hint")
        for i, (key, w) in enumerate((("audio_track", self.cb_track), ("source_mode", self.cb_mode),
                                      ("sub_track", self.cb_sub), ("src_lang", self.cb_lang))):
            g.addWidget(QLabel(t(key)), i // 2, (i % 2) * 2)
            g.addWidget(w, i // 2, (i % 2) * 2 + 1)
        g.addWidget(self.btn_subfile, 2, 0)
        g.addWidget(self.lbl_subfile, 2, 1, 1, 3)
        g.addWidget(QLabel(t("target_lang")), 3, 0)
        g.addWidget(QLabel(t("arabic")), 3, 1)
        self.cb_mode.addItem(t("mode_asr"), "asr")
        self.cb_mode.addItem(t("mode_embedded"), "embedded")
        self.cb_mode.addItem(t("mode_file"), "file")
        self.cb_lang.addItem(t("auto"), "")
        idx = 0 if ctx.i18n.lang == "ar" else 1
        for code, names in LANG_NAMES.items():
            if code != "ar":
                self.cb_lang.addItem("%s (%s)" % (names[idx], code), code)
        lay.addWidget(self.detail)

        st = QHBoxLayout()
        self.lbl_stage = QLabel("")
        self.lbl_info = QLabel("")
        self.lbl_info.setObjectName("hint")
        self.overall = QProgressBar()
        self.overall.setRange(0, 1000)
        st.addWidget(QLabel(t("overall")))
        st.addWidget(self.overall, 1)
        lay.addLayout(st)
        lay.addWidget(self.lbl_stage)
        lay.addWidget(self.lbl_info)

        self.btn["add_videos"].clicked.connect(self.pick_files)
        self.btn["remove"].clicked.connect(self.remove_selected)
        self.btn["start_all"].clicked.connect(self.start_all)
        self.btn["pause"].clicked.connect(lambda: self.stop_current("pause"))
        self.btn["resume"].clicked.connect(self.resume_selected)
        self.btn["cancel"].clicked.connect(self.cancel_selected)
        self.btn["retry"].clicked.connect(self.retry_selected)
        self.btn["open_folder"].clicked.connect(self.open_results)
        self.btn["report"].clicked.connect(self.show_report)
        self.btn["edit_result"].clicked.connect(self.edit_selected)
        self.btn["export_video"].clicked.connect(self.export_selected)
        self.table.itemSelectionChanged.connect(self.on_select)
        self.cb_track.currentIndexChanged.connect(self.store_detail)
        self.cb_mode.currentIndexChanged.connect(self.store_detail)
        self.cb_sub.currentIndexChanged.connect(self.store_detail)
        self.cb_lang.currentIndexChanged.connect(self.store_detail)
        self.btn_subfile.clicked.connect(self.pick_subfile)
        self.refresh()

    # ---------- queue ----------
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        self.add_paths([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])

    def pick_files(self):
        exts = " ".join("*" + x for x in sorted(SUPPORTED_EXT))
        files, _ = QFileDialog.getOpenFileNames(self, self.ctx.t("add_videos"), "", "Video/Audio (%s);;All (*)" % exts)
        self.add_paths(files)

    def add_paths(self, paths):
        ff = self.ctx.ffmpeg_factory()
        for p in paths:
            if any(os.path.abspath(i.path) == os.path.abspath(p) for i in self.ctx.items):
                self.status_message.emit("%s: %s" % (Path(p).name, self.ctx.t("dup_file")))
                continue
            try:
                info = ff.probe(p)
            except OVTError as e:
                QMessageBox.warning(self, self.ctx.t("unsupported_file"), "%s\n\n%s" % (Path(p).name, e))
                continue
            self.ctx.items.append(QueueItem(os.path.abspath(p), info, audio_track=min(self.ctx.settings.audio_track, len(info.audio_tracks) - 1),
                                            src_lang=self.ctx.settings.source_language))
        self.refresh()

    def remove_selected(self):
        it = self.selected()
        if it and it.status != "running":
            self.ctx.items.remove(it)
            self.refresh()

    def selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        r = rows[0] if rows else self.table.currentRow()
        return self.ctx.items[r] if 0 <= r < len(self.ctx.items) else None

    # ---------- rendering ----------
    def model_label(self):
        s = self.ctx.settings
        return s.asr_model_id or "—"

    def set_cell(self, r, c, text):
        it = self.table.item(r, c)
        if it is None:
            self.table.setItem(r, c, QTableWidgetItem(text))
        elif it.text() != text:
            it.setText(text)

    def refresh(self):
        keep_item = self.selected()
        t = self.ctx.t
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.ctx.items))
        for r, it in enumerate(self.ctx.items):
            vals = [it.info.name, ltr(fmt_size(it.info.size_bytes)), ltr(fmt_dur(it.info.duration)), str(len(it.info.audio_tracks)),
                    it.detected_lang or it.src_lang or "—", t("st_" + it.status) + ((": " + it.error) if it.status == "error" else ""),
                    None, ltr(self.model_label())]
            for c, v in enumerate(vals):
                if v is None:
                    pb = self.table.cellWidget(r, c)
                    if not isinstance(pb, QProgressBar):
                        pb = QProgressBar()
                        pb.setRange(0, 1000)
                        self.table.setCellWidget(r, c, pb)
                    pb.setValue(int(it.progress * 1000))
                else:
                    self.set_cell(r, c, v)
            self.table.item(r, 0).setToolTip(it.path)
        self.table.blockSignals(False)
        if keep_item in self.ctx.items:
            self.table.setCurrentCell(self.ctx.items.index(keep_item), 0)
        self.update_buttons()
        self.on_select()

    def update_progress_row(self, it):
        r = self.ctx.items.index(it)
        pb = self.table.cellWidget(r, 6)
        if isinstance(pb, QProgressBar):
            pb.setValue(int(it.progress * 1000))
        self.set_cell(r, 5, self.ctx.t("st_" + it.status))
        self.set_cell(r, 4, it.detected_lang or it.src_lang or "—")

    def update_buttons(self):
        busy = self.thread is not None
        it = self.selected()
        any_queued = any(i.status == "queued" for i in self.ctx.items)
        b = self.btn
        b["start_all"].setEnabled(not busy and any_queued)
        b["pause"].setEnabled(busy)
        b["resume"].setEnabled(not busy and it is not None and it.status == "paused")
        b["cancel"].setEnabled(it is not None and it.status in ("queued", "running", "paused"))
        b["retry"].setEnabled(not busy and it is not None and it.status in ("error", "cancelled"))
        b["remove"].setEnabled(it is not None and it.status != "running")
        b["open_folder"].setEnabled(it is not None and it.result is not None and bool(it.result.outputs))
        b["report"].setEnabled(it is not None and it.result is not None)
        b["export_video"].setEnabled(not busy and it is not None and it.result is not None and "srt" in it.result.outputs)
        b["edit_result"].setEnabled(it is not None and it.result is not None and "project" in it.result.outputs)

    def on_select(self):
        it = self.selected()
        widgets = (self.cb_track, self.cb_mode, self.cb_sub, self.cb_lang, self.btn_subfile)
        for w in widgets:
            w.blockSignals(True)
        self.cb_track.clear(); self.cb_sub.clear()
        if it:
            for tr in it.info.audio_tracks:
                self.cb_track.addItem("#%d  %s  %dch  %s %s" % (tr.index + 1, tr.codec, tr.channels, tr.language, tr.title), tr.index)
            for st in it.info.subtitle_tracks:
                self.cb_sub.addItem("#%d  %s  %s %s%s" % (st.index + 1, st.codec, st.language, st.title, "" if st.is_text else " (image)"), st.index)
            self.cb_track.setCurrentIndex(max(0, it.audio_track))
            self.cb_mode.setCurrentIndex(self.cb_mode.findData(it.source_mode))
            self.cb_sub.setCurrentIndex(max(0, it.subtitle_track) if self.cb_sub.count() else -1)
            self.cb_lang.setCurrentIndex(max(0, self.cb_lang.findData(it.src_lang)))
            self.lbl_subfile.setText(it.subtitle_file)
        editable = bool(it) and it.status in ("queued", "error", "cancelled", "paused")
        for w in widgets:
            w.setEnabled(editable)
            w.blockSignals(False)
        self.update_buttons()

    def store_detail(self):
        it = self.selected()
        if not it or it.status == "running":
            return
        it.audio_track = max(0, self.cb_track.currentData() or 0)
        it.source_mode = self.cb_mode.currentData()
        it.subtitle_track = max(0, self.cb_sub.currentData() or 0) if self.cb_sub.count() else 0
        it.src_lang = self.cb_lang.currentData() or ""
        it.status = "queued" if it.status in ("error", "cancelled") else it.status
        self.refresh_row_only(it)

    def refresh_row_only(self, it):
        r = self.ctx.items.index(it)
        self.set_cell(r, 4, it.detected_lang or it.src_lang or "—")
        self.set_cell(r, 5, self.ctx.t("st_" + it.status))

    def pick_subfile(self):
        it = self.selected()
        if not it:
            return
        f, _ = QFileDialog.getOpenFileName(self, self.ctx.t("choose_file"), "", "Subtitles (*.srt *.vtt *.ass *.ssa)")
        if f:
            it.subtitle_file = f
            self.lbl_subfile.setText(f)

    # ---------- running ----------
    def build_options(self, it: QueueItem) -> JobOptions:
        s = self.ctx.settings
        return JobOptions(
            input_path=it.path, out_dir=s.output_dir,
            audio_track=it.audio_track, source_lang=it.src_lang, target_lang="ar", asr_model_id=s.asr_model_id,
            mt_model_id=s.mt_model_id, source_mode=it.source_mode, subtitle_track=it.subtitle_track,
            subtitle_file=it.subtitle_file, chunk_seconds=s.chunk_seconds, glossary_path=s.glossary_path,
            context_merge=s.context_merge, write_txt=s.write_txt, write_source_srt=s.write_source_srt,
            keep_temp=s.keep_temp, max_chars=s.max_chars_per_line, max_lines=s.max_lines,
            min_cue_seconds=s.min_cue_seconds, bom=s.utf8_bom, rtl_marks=s.rtl_marks)

    def preflight(self, it):
        s = self.ctx.settings
        if it.source_mode == "asr" and not s.asr_model_id:
            return self.ctx.t("no_asr_model")
        if it.source_mode == "file" and not it.subtitle_file:
            return self.ctx.t("mode_file") + ": " + self.ctx.t("choose_file")
        if not any(True for _ in self.ctx.models.list("mt")):
            return self.ctx.t("no_mt_model")
        return ""

    def start_all(self):
        if self.thread:
            return
        self.run_next()

    def run_next(self):
        nxt = next((i for i in self.ctx.items if i.status == "queued"), None)
        if nxt is None:
            self.thread, self.current = None, None
            self.update_buttons()
            return
        problem = self.preflight(nxt)
        if problem:
            nxt.status, nxt.error = "error", problem
            self.refresh()
            QMessageBox.warning(self, nxt.info.name, problem)
            return self.run_next()
        self.current, self.stop_reason = nxt, ""
        nxt.status, nxt.error, nxt.progress = "running", "", 0.0
        self.thread = JobThread(self.build_options(nxt), self.ctx.builder)
        self.thread.progress.connect(self.on_progress)
        self.thread.done.connect(self.on_done)
        self.thread.failed.connect(self.on_failed)
        self.thread.cancelled.connect(self.on_cancelled)
        self.thread.finished.connect(self.on_thread_finished)
        self.thread.start()
        self.refresh()

    def on_progress(self, p):
        it = self.current
        if not it:
            return
        it.progress = p.fraction
        self.update_progress_row(it)
        self.overall.setValue(int(self.overall_fraction() * 1000))
        chunk = (" — %s %d/%d" % (self.ctx.t("chunk"), p.chunk, p.chunks)) if p.chunks else ""
        self.lbl_stage.setText("%s: %s%s" % (self.ctx.t("stage"), self.ctx.t("stage_" + p.stage), chunk))
        eta = fmt_dur(p.eta) if p.eta is not None else "—"
        self.lbl_info.setText("%s: %s    %s: %s    %s: %s" % (self.ctx.t("elapsed"), ltr(fmt_dur(p.elapsed)), self.ctx.t("eta"), ltr(eta),
                                                              self.ctx.t("model"), ltr(self.model_label())))

    def overall_fraction(self):
        its = self.ctx.items
        return sum(1.0 if i.status in ("done", "nospeech") else i.progress for i in its) / max(len(its), 1)

    def on_done(self, res):
        it = self.current
        it.result, it.progress = res, 1.0
        it.detected_lang = res.source_language
        it.status = "nospeech" if res.no_speech else "done"

    def on_failed(self, code, msg):
        it = self.current
        it.status, it.error = "error", msg
        self.log_error(code, msg)

    def on_cancelled(self):
        it = self.current
        it.status = "paused" if self.stop_reason == "pause" else "cancelled"
        it.message = self.ctx.t("saved_progress_hint")

    def on_thread_finished(self):
        self.thread = None
        self.refresh()
        self.overall.setValue(int(self.overall_fraction() * 1000))
        if self.stop_reason == "pause":
            return
        self.run_next()

    def log_error(self, code, msg):
        import datetime
        try:
            with open(self.ctx.paths.logs / "errors.log", "a", encoding="utf-8") as f:
                f.write("=== %s ===\n[%s] %s\n" % (datetime.datetime.now().isoformat(timespec="seconds"), code, msg))
        except OSError:
            pass

    def stop_current(self, reason):
        if self.thread:
            self.stop_reason = reason
            self.thread.request_cancel()

    def resume_selected(self):
        it = self.selected()
        if it and it.status == "paused" and not self.thread:
            it.status = "queued"
            self.run_next()

    def cancel_selected(self):
        it = self.selected()
        if not it:
            return
        if it.status == "running":
            self.stop_current("cancel")
        elif it.status in ("queued", "paused"):
            it.status = "cancelled"
            self.refresh()

    def retry_selected(self):
        it = self.selected()
        if it and it.status in ("error", "cancelled") and not self.thread:
            it.status, it.error = "queued", ""
            self.run_next()

    # ---------- results ----------
    def open_results(self):
        it = self.selected()
        if it and it.result and it.result.outputs:
            open_folder(Path(next(iter(it.result.outputs.values()))).parent)

    def edit_selected(self):
        it = self.selected()
        if it and it.result and "project" in it.result.outputs:
            self.open_in_editor.emit(it.result.outputs["project"])

    def export_selected(self):
        it = self.selected()
        if not it or not it.result or "srt" not in it.result.outputs or getattr(self, "_export", None):
            return
        from ..export import export_with_subtitles
        from .workers import FuncThread
        ff = self.ctx.ffmpeg_factory()
        self.status_message.emit(self.ctx.t("m_working"))
        self._export = FuncThread(lambda: export_with_subtitles(ff, it.path, it.result.outputs["srt"], self.ctx.settings.output_dir or None))

        def ok(path):
            self._export = None
            it.result.outputs["video"] = str(path)
            self.status_message.emit("%s: %s" % (self.ctx.t("export_ok"), path))

        def err(code, msg):
            self._export = None
            self.log_error(code, msg)
            self.status_message.emit(msg)
            QMessageBox.warning(self, self.ctx.t("export_video"), msg)
        self._export.ok.connect(ok)
        self._export.err.connect(err)
        self._export.start()

    def report_text(self, it):
        t = self.ctx.t
        r = it.result
        lines = [it.info.name, "", "%s: %s" % (t("col_lang"), r.source_language or "—"),
                 "%s: %d" % ("Cues", r.cue_count), "%.1f s" % r.seconds, "", t("outputs") + ":"]
        lines += ["  " + p for p in r.outputs.values()] or ["  " + t("none")]
        lines += ["", t("warnings") + ":"] + (["  - " + w for w in r.warnings] or ["  " + t("none")])
        return "\n".join(lines)

    def show_report(self):
        it = self.selected()
        if not it or not it.result:
            return
        d = QDialog(self)
        d.setWindowTitle(self.ctx.t("job_report_title"))
        d.resize(640, 420)
        v = QVBoxLayout(d)
        te = QPlainTextEdit(self.report_text(it))
        te.setReadOnly(True)
        v.addWidget(te)
        b = QPushButton(self.ctx.t("close"))
        b.clicked.connect(d.accept)
        v.addWidget(b)
        d.exec()
