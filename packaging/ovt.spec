# PyInstaller spec: one folder containing the GUI exe + a console CLI exe sharing the same libraries.
# Build:  pyinstaller --noconfirm --clean packaging/ovt.spec
import sys
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs

datas, binaries, hidden = [], [], []
for pkg in ("faster_whisper", "ctranslate2", "onnxruntime", "av", "tokenizers", "sentencepiece"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hidden += h
hidden += ["ovt.gui.main_window", "ovt.gui.pages_preview", "ovt.cli", "ovt.diagnostics",
           "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets"]

excludes = ["torch", "torchaudio", "torchvision", "tensorflow", "stanza", "spacy", "argostranslate",
            "matplotlib", "pandas", "scipy", "IPython", "notebook", "tkinter", "PySide6.QtWebEngineCore",
            "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore", "PySide6.QtQuick", "PySide6.QtQml"]

common = dict(pathex=[".."], binaries=binaries, datas=datas, hiddenimports=hidden, excludes=excludes, noarchive=False)
a_gui = Analysis(["../OfflineVideoTranslator.py"], **common)
a_cli = Analysis(["cli_entry.py"], **common)
MERGE((a_gui, "OfflineVideoTranslator", "OfflineVideoTranslator"), (a_cli, "ovt-cli", "ovt-cli"))

pyz_gui, pyz_cli = PYZ(a_gui.pure), PYZ(a_cli.pure)
exe_gui = EXE(pyz_gui, a_gui.scripts, [], exclude_binaries=True, name="OfflineVideoTranslator", console=False)
exe_cli = EXE(pyz_cli, a_cli.scripts, [], exclude_binaries=True, name="ovt-cli", console=True)
coll = COLLECT(exe_gui, a_gui.binaries, a_gui.datas, exe_cli, a_cli.binaries, a_cli.datas,
               strip=False, upx=False, name="OfflineVideoTranslator")
