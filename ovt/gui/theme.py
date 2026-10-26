LIGHT = dict(bg="#f4f5f7", panel="#ffffff", text="#1b1f24", sub="#5b6470", accent="#1f6feb", border="#d0d5dc", sel="#dbe8ff", bar="#e6e9ee")
DARK = dict(bg="#14171b", panel="#1c2026", text="#e8eaed", sub="#9aa3ad", accent="#4c9aff", border="#313843", sel="#25405f", bar="#2a3038")


def stylesheet(theme: str) -> str:
    c = DARK if theme == "dark" else LIGHT
    return """
    QWidget {{ background:{bg}; color:{text}; font-size:13px; }}
    QListWidget#nav {{ background:{panel}; border:none; border-right:1px solid {border}; font-size:14px; }}
    QListWidget#nav::item {{ padding:12px 16px; }}
    QListWidget#nav::item:selected {{ background:{sel}; color:{text}; border-left:3px solid {accent}; }}
    QTableWidget, QTextEdit, QPlainTextEdit, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        background:{panel}; border:1px solid {border}; border-radius:5px; padding:3px; selection-background-color:{sel}; selection-color:{text}; }}
    QHeaderView::section {{ background:{bar}; border:none; padding:6px; font-weight:bold; }}
    QPushButton {{ background:{panel}; border:1px solid {border}; border-radius:6px; padding:7px 14px; }}
    QPushButton:hover {{ border-color:{accent}; }}
    QPushButton:disabled {{ color:{sub}; background:{bar}; }}
    QPushButton#primary {{ background:{accent}; color:white; border:none; font-weight:bold; }}
    QPushButton#primary:disabled {{ background:{bar}; color:{sub}; }}
    QGroupBox {{ border:1px solid {border}; border-radius:8px; margin-top:12px; padding-top:10px; }}
    QGroupBox::title {{ subcontrol-origin:margin; left:12px; padding:0 6px; color:{sub}; }}
    QProgressBar {{ border:1px solid {border}; border-radius:5px; text-align:center; background:{bar}; height:16px; }}
    QProgressBar::chunk {{ background:{accent}; border-radius:4px; }}
    QLabel#offlineOn {{ color:#1a7f37; font-weight:bold; }}
    QLabel#offlineOff {{ color:#cf222e; font-weight:bold; }}
    QLabel#hint {{ color:{sub}; }}
    """.format(**c)
