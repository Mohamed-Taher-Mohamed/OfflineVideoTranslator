import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QListWidget, QMainWindow, QStackedWidget, QVBoxLayout, QWidget

from .. import netguard
from .context import Ctx
from .i18n import I18n
from .pages_editor import EditorPage
from .pages_home import HomePage
from .pages_misc import DubbingPage, LogsPage, PrivacyPage
from .pages_models import ModelsPage
from .pages_preview import PreviewPage
from .pages_settings import SettingsPage
from .theme import stylesheet

NAV = [("nav_home", HomePage), ("nav_editor", EditorPage), ("nav_preview", PreviewPage), ("nav_models", ModelsPage), ("nav_settings", SettingsPage),
       ("nav_logs", LogsPage), ("nav_privacy", PrivacyPage), ("nav_dubbing", DubbingPage)]


class MainWindow(QMainWindow):
    def __init__(self, ctx: Ctx):
        super().__init__()
        self.ctx = ctx
        self.resize(1180, 760)
        self.build()

    def build(self):
        ctx = self.ctx
        ctx.i18n = I18n(ctx.settings.ui_language)
        app = QApplication.instance()
        app.setLayoutDirection(Qt.RightToLeft if ctx.i18n.rtl else Qt.LeftToRight)
        app.setStyleSheet(stylesheet(ctx.settings.theme))
        self.setWindowTitle(ctx.t("app_title"))
        root = QWidget()
        h = QHBoxLayout(root)
        h.setContentsMargins(0, 0, 0, 0)
        self.nav = QListWidget(); self.nav.setObjectName("nav"); self.nav.setFixedWidth(190)
        self.stack = QStackedWidget()
        self.pages = {}
        for key, cls in NAV:
            self.nav.addItem(ctx.t(key))
            page = cls(ctx)
            self.pages[key] = page
            self.stack.addWidget(page)
        right = QVBoxLayout()
        self.banner = QLabel()
        right.addWidget(self.banner)
        right.addWidget(self.stack, 1)
        self.status = QLabel("")
        self.status.setObjectName("hint")
        right.addWidget(self.status)
        h.addWidget(self.nav)
        h.addLayout(right, 1)
        self.setCentralWidget(root)
        self.nav.currentRowChanged.connect(self.on_nav)
        self.nav.setCurrentRow(0)
        home = self.pages["nav_home"]
        home.open_in_editor.connect(self.open_editor)
        home.open_in_preview.connect(self.open_preview)
        home.status_message.connect(self.status.setText)
        self.pages["nav_settings"].applied.connect(self.on_settings_applied)
        self.update_banner()

    def on_nav(self, row):
        self.stack.setCurrentIndex(row)
        key = NAV[row][0]
        if key == "nav_settings":
            self.pages[key].fill_models()
        elif key == "nav_privacy" or key == "nav_logs":
            self.pages[key].refresh()
        elif key == "nav_models":
            self.pages[key].refresh()
        elif key == "nav_home":
            self.pages[key].refresh()

    def update_banner(self):
        off = netguard.is_offline()
        self.banner.setText(self.ctx.t("offline_on") if off else self.ctx.t("offline_off"))
        self.banner.setObjectName("offlineOn" if off else "offlineOff")
        self.banner.style().unpolish(self.banner); self.banner.style().polish(self.banner)

    def open_preview(self, video, subs):
        self.pages["nav_preview"].open_preview(video, subs)
        self.nav.setCurrentRow([k for k, _ in NAV].index("nav_preview"))

    def open_editor(self, path):
        self.pages["nav_editor"].load(path)
        self.nav.setCurrentRow([k for k, _ in NAV].index("nav_editor"))

    def on_settings_applied(self, rebuild):
        if rebuild:
            busy = self.pages["nav_home"].thread is not None
            if not busy:
                self.build()
                self.nav.setCurrentRow([k for k, _ in NAV].index("nav_settings"))
                return
        self.update_banner()

    def closeEvent(self, e):
        pv = self.pages.get("nav_preview")
        if pv is not None and getattr(pv, "player", None) is not None:
            pv.player.stop()
        home = self.pages["nav_home"]
        if home.thread:
            home.stop_current("pause")
            home.thread.wait(5000)
        e.accept()
