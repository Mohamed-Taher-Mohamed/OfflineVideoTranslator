import json
from pathlib import Path
from typing import List

from PySide6.QtCore import Qt
import re

from PySide6.QtWidgets import (QStyledItemDelegate, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from .. import subtitles as sub
from ..errors import OVTError
from ..fsutil import atomic_write_text


_AR = re.compile(r"[\u0600-\u06ff]")


class DirectionDelegate(QStyledItemDelegate):
    """Chooses reading direction per cell from its content, so English text keeps its punctuation in the
    right place inside an RTL window (and Arabic stays RTL)."""

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        arabic = bool(_AR.search(option.text))
        option.direction = Qt.RightToLeft if arabic else Qt.LeftToRight
        option.displayAlignment = (Qt.AlignRight if arabic else Qt.AlignLeft) | Qt.AlignVCenter


class EditorPage(QWidget):
    COLS = ["#", "e_col_start", "e_col_end", "e_col_src", "e_col_tr"]

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.cues: List[sub.Cue] = []
        self.path = ""
        t = ctx.t
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        self.b_open, self.b_save, self.b_txt = QPushButton(t("e_open")), QPushButton(t("e_save")), QPushButton(t("e_save_txt"))
        self.sp = QDoubleSpinBox()
        self.sp.setRange(-3600, 3600); self.sp.setDecimals(3); self.sp.setSingleStep(0.1)
        self.b_shift, self.b_reset = QPushButton(t("e_apply_shift")), QPushButton(t("e_reset_time"))
        for w in (self.b_open, self.b_save, self.b_txt, QLabel(t("e_shift")), self.sp, self.b_shift, self.b_reset):
            row.addWidget(w)
        row.addStretch(1)
        lay.addLayout(row)
        self.lbl = QLabel("")
        self.lbl.setObjectName("hint")
        lay.addWidget(self.lbl)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([c if c == "#" else t(c) for c in self.COLS])
        self.table.verticalHeader().setVisible(False)
        self.delegate = DirectionDelegate(self.table)
        self.table.setItemDelegateForColumn(3, self.delegate)
        self.table.setItemDelegateForColumn(4, self.delegate)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(1, 120); self.table.setColumnWidth(2, 120); self.table.setColumnWidth(3, 320)
        lay.addWidget(self.table, 1)
        self._loading = False
        self.b_open.clicked.connect(self.pick)
        self.b_save.clicked.connect(self.save_dialog)
        self.b_txt.clicked.connect(self.export_txt_dialog)
        self.b_shift.clicked.connect(lambda: self.shift(self.sp.value()))
        self.b_reset.clicked.connect(self.reset_timing)
        self.table.itemChanged.connect(self.on_edit)

    # ---- loading ----
    def pick(self):
        f, _ = QFileDialog.getOpenFileName(self, self.ctx.t("e_open"), "", "Project/SRT (*.ovtproject.json *.srt *.vtt *.ass)")
        if f:
            self.load(f)

    def load(self, path):
        p = Path(path)
        if p.name.endswith(".ovtproject.json"):
            d = json.loads(p.read_text(encoding="utf-8"))
            self.cues = [sub.Cue.from_dict(c) for c in d["cues"]]
            self.path = str(p)
        else:
            self.cues = sub.load_subtitles(p)
            for c in self.cues:
                c.source_text = ""
            self.path = str(p)
        self.lbl.setText(self.path)
        self.render()

    def render(self):
        self._loading = True
        self.table.setRowCount(len(self.cues))
        for r, c in enumerate(self.cues):
            vals = [str(r + 1), sub.fmt_ts(c.start), sub.fmt_ts(c.end), c.source_text, c.text]
            for k, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if k in (0, 3):
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(r, k, it)
        self._loading = False

    # ---- editing ----
    def on_edit(self, item):
        if self._loading:
            return
        r, k = item.row(), item.column()
        c = self.cues[r]
        if k in (1, 2):
            try:
                v = sub.parse_ts(item.text())
            except OVTError:
                self.lbl.setText(self.ctx.t("e_bad_time"))
                self.render()
                return
            if k == 1:
                c.start = v
            else:
                c.end = v
        elif k == 4:
            c.text = item.text()

    def shift(self, delta):
        sub.shift(self.cues, delta)
        self.render()

    def reset_timing(self):
        sub.reset_timing(self.cues)
        self.render()

    # ---- saving ----
    def save_srt(self, path) -> List[str]:
        """Write and validate. Returns list of problems (empty = valid)."""
        s = self.ctx.settings
        sub.write_srt(sorted(self.cues, key=lambda c: c.start), path, s.utf8_bom, s.rtl_marks)
        return sub.validate_srt(path)

    def save_dialog(self):
        default = str(Path(self.path).with_suffix("")) + ".edited.srt" if self.path else "subtitles.ar.srt"
        f, _ = QFileDialog.getSaveFileName(self, self.ctx.t("e_save"), default, "SRT (*.srt)")
        if not f:
            return
        probs = self.save_srt(f)
        if probs:
            QMessageBox.warning(self, self.ctx.t("e_invalid"), "\n".join(probs[:8]))
        else:
            self.lbl.setText(self.ctx.t("e_saved") + ": " + f)

    def export_txt_dialog(self):
        f, _ = QFileDialog.getSaveFileName(self, self.ctx.t("e_save_txt"), "transcript.txt", "TXT (*.txt)")
        if f:
            atomic_write_text(f, sub.to_txt(self.cues))
