"""Video preview with synchronised source + translated subtitles and a clickable cue list.
Uses Qt Multimedia (PySide6-Addons). If it is not installed the page degrades to 'open in default player'."""
import json
import os
from pathlib import Path
from typing import List

from PySide6.QtCore import QUrl, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QSlider,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QMessageBox)

from .. import subtitles as sub
from ..errors import OVTError
from ..preview_logic import clock, cue_index_at, neighbour_start

try:
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
    from PySide6.QtMultimediaWidgets import QVideoWidget
    HAVE_MM = True
except Exception:                      # addon missing: the page still works as a subtitle viewer
    HAVE_MM = False

SPEEDS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


class PreviewPage(QWidget):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        t = ctx.t
        self.cues: List[sub.Cue] = []
        self.starts, self.ends = [], []
        self.video_path = ""
        self.subs_path = ""
        self.cur = None
        self._seeking = False
        self._syncing_table = False
        self.pos_ms = 0                      # last known position (updated at once on seek; the player's own lags)

        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.b_video, self.b_subs = QPushButton(t("pv_open_video")), QPushButton(t("pv_open_subs"))
        self.b_ext = QPushButton(t("pv_open_external"))
        for b in (self.b_video, self.b_subs, self.b_ext):
            top.addWidget(b)
        self.lbl_path = QLabel(""); self.lbl_path.setObjectName("hint")
        top.addWidget(self.lbl_path, 1)
        lay.addLayout(top)

        body = QHBoxLayout()
        left = QVBoxLayout()
        if HAVE_MM:
            self.video = QVideoWidget(); self.video.setMinimumSize(420, 240)
            self.player = QMediaPlayer(self)
            # Audio can be switched off (OVT_PREVIEW_AUDIO=0): on machines with a broken audio stack Qt can crash
            # while enumerating devices, which would take the whole program down.
            self.audio = QAudioOutput(self) if os.environ.get("OVT_PREVIEW_AUDIO", "1") != "0" else None
            if self.audio is not None:
                self.player.setAudioOutput(self.audio)
            self.player.setVideoOutput(self.video)
            left.addWidget(self.video, 1)
        else:
            self.player = self.audio = None
            msg = QLabel(t("pv_no_mm")); msg.setWordWrap(True); msg.setAlignment(Qt.AlignCenter)
            left.addWidget(msg, 1)
        self.lbl_tr = QLabel(""); self.lbl_tr.setWordWrap(True); self.lbl_tr.setAlignment(Qt.AlignCenter)
        self.lbl_tr.setMinimumHeight(70)
        self.lbl_tr.setStyleSheet("background:#000; color:#fff; font-size:20px; font-weight:bold; padding:8px; border-radius:6px;")
        self.lbl_src = QLabel(""); self.lbl_src.setWordWrap(True); self.lbl_src.setAlignment(Qt.AlignCenter)
        self.lbl_src.setLayoutDirection(Qt.LeftToRight); self.lbl_src.setObjectName("hint")
        left.addWidget(self.lbl_tr); left.addWidget(self.lbl_src)

        ctl = QHBoxLayout()
        self.b_play, self.b_prev, self.b_next = QPushButton(t("pv_play")), QPushButton(t("pv_prev")), QPushButton(t("pv_next"))
        self.slider = QSlider(Qt.Horizontal); self.slider.setRange(0, 0); self.slider.setLayoutDirection(Qt.LeftToRight)
        self.lbl_time = QLabel("0:00 / 0:00"); self.lbl_time.setLayoutDirection(Qt.LeftToRight)
        self.cb_speed = QComboBox()
        for sp in SPEEDS:
            self.cb_speed.addItem("%gx" % sp, sp)
        self.cb_speed.setCurrentIndex(SPEEDS.index(1.0))
        self.vol = QSlider(Qt.Horizontal); self.vol.setRange(0, 100); self.vol.setValue(80); self.vol.setMaximumWidth(90)
        for w in (self.b_prev, self.b_play, self.b_next):
            ctl.addWidget(w)
        ctl.addWidget(self.slider, 1); ctl.addWidget(self.lbl_time)
        ctl.addWidget(QLabel(t("pv_speed"))); ctl.addWidget(self.cb_speed)
        ctl.addWidget(QLabel(t("pv_volume"))); ctl.addWidget(self.vol)
        left.addLayout(ctl)
        body.addLayout(left, 3)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["", t("pv_src"), t("pv_tr")])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setColumnWidth(0, 52); self.table.setColumnWidth(1, 170)
        self.table.horizontalHeader().setStretchLastSection(True)
        body.addWidget(self.table, 2)
        lay.addLayout(body, 1)
        self.status = QLabel(""); self.status.setObjectName("hint"); lay.addWidget(self.status)

        self.b_video.clicked.connect(self.pick_video)
        self.b_subs.clicked.connect(self.pick_subs)
        self.b_ext.clicked.connect(self.open_external)
        self.b_play.clicked.connect(self.toggle)
        self.b_prev.clicked.connect(lambda: self.step(-1))
        self.b_next.clicked.connect(lambda: self.step(+1))
        self.table.cellClicked.connect(self.on_row_clicked)
        if HAVE_MM:
            self.player.positionChanged.connect(self.on_position)
            self.player.durationChanged.connect(lambda d: self.slider.setRange(0, d))
            self.player.playbackStateChanged.connect(self.on_state)
            self.player.errorOccurred.connect(self.on_error)
            self.slider.sliderPressed.connect(lambda: setattr(self, "_seeking", True))
            self.slider.sliderReleased.connect(self.on_slider_released)
            self.cb_speed.currentIndexChanged.connect(lambda: self.player.setPlaybackRate(self.cb_speed.currentData()))
            if self.audio is not None:
                self.vol.valueChanged.connect(lambda v: self.audio.setVolume(v / 100.0))
                self.audio.setVolume(0.8)
        for w in (self.b_play, self.slider, self.cb_speed):
            w.setEnabled(HAVE_MM)
        self.vol.setEnabled(HAVE_MM and getattr(self, "audio", None) is not None)

    # ---------- loading ----------
    def pick_video(self):
        f, _ = QFileDialog.getOpenFileName(self, self.ctx.t("pv_open_video"), "", "Video (*.mp4 *.mkv *.mov *.avi *.webm *.m4v);;All (*)")
        if f:
            self.load_video(f)

    def pick_subs(self):
        f, _ = QFileDialog.getOpenFileName(self, self.ctx.t("pv_open_subs"), "", "Project/SRT (*.ovtproject.json *.srt *.vtt *.ass)")
        if f:
            self.load_subs(f)

    def open_preview(self, video, subs):
        """Entry point used by the Home page and the main window."""
        self.load_subs(subs, video)
        if video:
            self.load_video(video)

    def load_video(self, path):
        self.video_path = str(path)
        self.lbl_path.setText(self.video_path)
        if HAVE_MM:
            self.player.setSource(QUrl.fromLocalFile(self.video_path))
            self.player.pause()                  # shows the first frame without starting playback

    def load_subs(self, path, video=""):
        p = Path(path)
        try:
            if p.name.endswith(".ovtproject.json"):
                d = json.loads(p.read_text(encoding="utf-8"))
                cues = [sub.Cue.from_dict(c) for c in d["cues"]]
                guess = video or d.get("source_path") or ""
                if not guess or not Path(guess).is_file():
                    cand = p.with_name(d.get("source_file", ""))
                    guess = str(cand) if d.get("source_file") and cand.is_file() else ""
                if guess and not self.video_path:
                    self.video_path = guess
            else:
                cues = sub.load_subtitles(p)
                for c in cues:
                    c.source_text = c.source_text or ""
        except (OVTError, OSError, ValueError, KeyError) as e:
            QMessageBox.warning(self, self.ctx.t("pv_open_subs"), str(e))
            return
        self.subs_path = str(p)
        self.set_cues(cues)
        if self.video_path and (not HAVE_MM or self.player.source().isEmpty()):
            self.load_video(self.video_path)
        elif not self.video_path:
            self.status.setText(self.ctx.t("pv_need_video"))

    def set_cues(self, cues):
        self.cues = sorted(cues, key=lambda c: c.start)
        self.starts = [c.start for c in self.cues]
        self.ends = [c.end for c in self.cues]
        self.cur = None
        self.table.setRowCount(len(self.cues))
        for r, c in enumerate(self.cues):
            for k, v in enumerate((clock(c.start), c.source_text.replace("\n", " "), c.text.replace("\n", " "))):
                self.table.setItem(r, k, QTableWidgetItem(v))
        self.show_cue(None)

    # ---------- playback / sync ----------
    def toggle(self):
        if not HAVE_MM:
            return
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def on_state(self, st):
        self.b_play.setText(self.ctx.t("pv_pause") if HAVE_MM and st == QMediaPlayer.PlayingState else self.ctx.t("pv_play"))

    def on_error(self, err, msg):
        self.status.setText("%s: %s" % (self.ctx.t("pv_error"), msg))

    def seek_ms(self, ms):
        if HAVE_MM:
            self.player.setPosition(int(ms))
        self.on_position(int(ms))

    def on_slider_released(self):
        self._seeking = False
        self.seek_ms(self.slider.value())

    def on_position(self, ms):
        """Called by the player (and directly by tests). Updates slider, clock, subtitle labels and table highlight."""
        self.pos_ms = ms
        sec = ms / 1000.0
        if not self._seeking:
            self.slider.blockSignals(True); self.slider.setValue(int(ms)); self.slider.blockSignals(False)
        total = (self.player.duration() / 1000.0) if HAVE_MM else 0
        self.lbl_time.setText("%s / %s" % (clock(sec), clock(total)))
        idx = cue_index_at(self.starts, self.ends, sec)
        if idx != self.cur:
            self.show_cue(idx)

    def show_cue(self, idx):
        self.cur = idx
        if idx is None:
            self.lbl_tr.setText(""); self.lbl_src.setText("")
            return
        c = self.cues[idx]
        self.lbl_tr.setText(c.text.replace("\n", " "))
        self.lbl_src.setText(c.source_text.replace("\n", " "))
        self._syncing_table = True
        self.table.setCurrentCell(idx, 0)
        self.table.scrollToItem(self.table.item(idx, 0), QAbstractItemView.PositionAtCenter)
        self._syncing_table = False

    def on_row_clicked(self, row, col):
        if not self._syncing_table and 0 <= row < len(self.cues):
            self.seek_ms(self.cues[row].start * 1000)

    def step(self, direction):
        pos = self.pos_ms / 1000.0
        t = neighbour_start(self.starts, pos, direction)
        if t is not None:
            self.seek_ms(t * 1000)

    def open_external(self):
        if self.video_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.video_path))
