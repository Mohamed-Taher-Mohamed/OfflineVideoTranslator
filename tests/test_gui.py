"""GUI tests run headless (QT_QPA_PLATFORM=offscreen). Backends are the SIMULATED Fake* ones, so these
prove the GUI wiring (threads, queue, pause/resume, pages), not recognition/translation quality."""
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from conftest import FakeASR, FakeMT
from ovt import netguard, subtitles as sub
from ovt.config import Paths, Settings
from ovt.media import FFmpeg
from ovt.models import ModelManager
from ovt.pipeline import Pipeline
from ovt.gui.context import Ctx
from ovt.gui.main_window import MainWindow
from test_models import make_asr_dir, make_argos


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    for w in app.topLevelWidgets():          # explicit teardown: avoids a Qt crash at interpreter exit
        w.close(); w.deleteLater()
    app.processEvents()


class SlowASR(FakeASR):
    def transcribe(self, wav, language=None):
        time.sleep(0.4)
        return super().transcribe(wav, language)


def wait_for(cond, timeout=25):
    app = QApplication.instance()
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def win(qapp, tmp_path, monkeypatch):
    paths = Paths(tmp_path / "home").ensure()
    settings = Settings(asr_model_id="m1", chunk_seconds=5, output_dir=str(tmp_path / "out"))
    mm = ModelManager(paths)
    mm.import_model("asr", make_asr_dir(tmp_path / "asrsrc"), "m1")
    mm.import_model("mt", make_argos(tmp_path / "t.argosmodel"), "en-ar")
    asr_cls = {"cls": FakeASR}
    ff = FFmpeg()
    builder = lambda o, prog, cancel: Pipeline(paths, ff, lambda: asr_cls["cls"](), lambda: FakeMT(), None, prog, cancel)
    ctx = Ctx(paths, settings, mm, builder, lambda: ff)
    netguard.enable_offline(paths.logs / "network.log")
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a[2] if len(a) > 2 else ""))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: warnings.append("info"))
    w = MainWindow(ctx)
    w.show()
    w.warnings, w.asr_cls = warnings, asr_cls
    FakeASR.calls = 0
    yield w
    home = w.pages["nav_home"]
    if home.thread:
        home.thread.request_cancel(); home.thread.wait(5000)
    w.close()
    netguard.disable_offline()


def test_builds_all_pages_both_languages_and_themes(win):
    assert win.nav.count() == 7 and win.windowTitle() == "مترجم الفيديو بدون إنترنت"
    assert QApplication.instance().layoutDirection() == Qt.RightToLeft
    win.ctx.settings.ui_language, win.ctx.settings.theme = "en", "dark"
    win.build()
    assert win.windowTitle() == "Offline Video Translator" and QApplication.instance().layoutDirection() == Qt.LeftToRight
    assert win.nav.item(0).text() == "Home" and "ON" in win.banner.text()
    for row in range(win.nav.count()):
        win.nav.setCurrentRow(row)                 # every page opens without error


def test_add_files_arabic_path_and_bad_file(win, media):
    home = win.pages["nav_home"]
    home.add_paths([str(media["multi"]), str(media["corrupt"]), str(media["multi"])])
    assert len(win.ctx.items) == 1 and len(win.warnings) == 1          # corrupt rejected with a message, duplicate ignored
    assert home.table.item(0, 0).text() == "فيديو اختبار 1.mkv"
    assert home.table.item(0, 3).text() == "2" and home.table.item(0, 2).text().strip("\u200e") == "0:00:12"


def run_to_end(win, expect="done"):
    home = win.pages["nav_home"]
    home.start_all()
    assert wait_for(lambda: home.thread is None and all(i.status != "running" for i in win.ctx.items))


def test_full_job_through_the_gui(win, media, tmp_path):
    home = win.pages["nav_home"]
    home.add_paths([str(media["multi"])])
    run_to_end(win)
    it = win.ctx.items[0]
    assert it.status == "done" and it.progress == 1.0
    srt = Path(it.result.outputs["srt"])
    assert srt.parent == tmp_path / "out" and sub.validate_srt(srt) == []
    assert home.table.cellWidget(0, 6).value() == 1000 and home.overall.value() == 1000
    home.table.setCurrentCell(0, 0)
    assert home.btn["report"].isEnabled() and "ar.srt" in home.report_text(it)


def test_missing_asr_model_blocks_with_clear_message(win, media):
    win.ctx.settings.asr_model_id = ""
    home = win.pages["nav_home"]
    home.add_paths([str(media["multi"])])
    run_to_end(win)
    assert win.ctx.items[0].status == "error" and "Models" in win.warnings[-1] or "النماذج" in win.warnings[-1]


def test_pause_resume_cancel_remove(win, media):
    win.asr_cls["cls"] = SlowASR
    home = win.pages["nav_home"]
    home.add_paths([str(media["multi"])])
    it = win.ctx.items[0]
    home.table.setCurrentCell(0, 0)
    home.start_all()
    assert wait_for(lambda: it.progress > 0.1)
    home.btn["pause"].click()
    assert wait_for(lambda: home.thread is None)
    assert it.status == "paused" and it.progress < 1.0
    home.table.setCurrentCell(0, 0)
    assert home.btn["resume"].isEnabled()
    home.btn["resume"].click()
    assert wait_for(lambda: home.thread is None and it.status != "running")
    assert it.status == "done" and it.result.chunks_resumed >= 1            # finished chunks were not redone
    # cancel a queued item then remove it
    win.ctx.items.append(type(it)(it.path.replace("1.mkv", "2.mkv"), it.info))
    home.refresh(); home.table.setCurrentCell(1, 0)
    home.btn["cancel"].click()
    assert win.ctx.items[1].status == "cancelled"
    home.btn["remove"].click()
    assert len(win.ctx.items) == 1


def test_running_item_cannot_be_removed(win, media):
    win.asr_cls["cls"] = SlowASR
    home = win.pages["nav_home"]
    home.add_paths([str(media["multi"])])
    home.table.setCurrentCell(0, 0); home.start_all()
    assert wait_for(lambda: win.ctx.items[0].status == "running")
    home.remove_selected()
    assert len(win.ctx.items) == 1
    home.btn["cancel"].click()
    assert wait_for(lambda: home.thread is None)
    assert win.ctx.items[0].status == "cancelled"


def test_editor_edit_shift_reset_save_and_open_from_result(win, media, tmp_path):
    home = win.pages["nav_home"]
    home.add_paths([str(media["multi"])]); run_to_end(win)
    home.table.setCurrentCell(0, 0); home.btn["edit_result"].click()
    ed = win.pages["nav_editor"]
    assert win.stack.currentWidget() is ed and len(ed.cues) == 3 and ed.table.item(0, 3).text().startswith("This is")
    ed.table.item(0, 4).setText("نص معدّل يدويًا")
    assert ed.cues[0].text == "نص معدّل يدويًا"
    old = ed.cues[0].start
    ed.table.item(0, 1).setText("garbage")
    assert ed.cues[0].start == old and ed.ctx.t("e_bad_time") in ed.lbl.text()
    ed.table.item(0, 1).setText("00:00:00,500")
    assert abs(ed.cues[0].start - 0.5) < 1e-6
    ed.shift(2.0); assert abs(ed.cues[0].start - 2.5) < 1e-6
    ed.reset_timing(); assert abs(ed.cues[0].start - ed.cues[0].orig_start) < 1e-9
    out = tmp_path / "edited.srt"
    assert ed.save_srt(out) == [] and "نص معدّل يدويًا" in out.read_text(encoding="utf-8")


def test_models_page_import_verify_default_delete(win, tmp_path, monkeypatch):
    mp = win.pages["nav_models"]
    mp.do_import("asr", str(make_asr_dir(tmp_path / "src2")), "second", "MIT")
    assert wait_for(lambda: mp.job is None and mp.table.rowCount() == 3)
    ids = [mp.table.item(r, 1).text() for r in range(mp.table.rowCount())]
    assert "second" in ids
    mp.table.setCurrentCell(ids.index("second"), 0)
    mp.verify(); assert wait_for(lambda: mp.job is None and "OK" in mp.status.text() or "سليم" in mp.status.text())
    mp.set_default(); assert win.ctx.settings.asr_model_id == "second"
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    mp.delete()
    assert mp.table.rowCount() == 2 and win.ctx.settings.asr_model_id == ""


def test_settings_persist_and_reset_keeps_models(win, tmp_path):
    sp = win.pages["nav_settings"]
    sp.sp_chars.setValue(37); sp.cb_bom.setChecked(True); sp.cb_txt.setChecked(True)
    sp.save()
    on_disk = Settings.load(win.ctx.paths)
    assert on_disk.max_chars_per_line == 37 and on_disk.utf8_bom and on_disk.write_txt
    sp.reset()
    assert Settings.load(win.ctx.paths).max_chars_per_line == 42
    assert len(win.ctx.models.list()) == 2 and netguard.is_offline()


def test_language_switch_via_settings_rebuilds_ui(win):
    sp = win.pages["nav_settings"]
    sp.cb_ui.setCurrentIndex(sp.cb_ui.findData("en"))
    sp.save()
    assert win.windowTitle() == "Offline Video Translator" and Settings.load(win.ctx.paths).ui_language == "en"


def test_privacy_test_really_blocks_and_logs_and_clear(win):
    pp = win.pages["nav_privacy"]
    assert pp.test_block() is True
    assert "93.184.216.34" in (win.ctx.paths.logs / "network.log").read_text(encoding="utf-8")
    (win.ctx.paths.temp / "jobs").mkdir(parents=True, exist_ok=True)
    (win.ctx.paths.temp / "jobs" / "x.json").write_text("{}")
    pp.wipe(win.ctx.paths.temp / "jobs")
    assert not list((win.ctx.paths.temp / "jobs").iterdir())


def test_logs_page_shows_job_errors_and_exports(win, media, tmp_path):
    class Boom(FakeASR):
        def transcribe(self, wav, language=None):
            raise RuntimeError("boom")
    win.asr_cls["cls"] = Boom
    home = win.pages["nav_home"]
    home.add_paths([str(media["multi"])]); run_to_end(win)
    assert win.ctx.items[0].status == "error" and "boom" in win.ctx.items[0].error
    home.table.setCurrentCell(0, 0); assert home.btn["retry"].isEnabled()
    lp = win.pages["nav_logs"]; lp.refresh()
    assert "boom" in lp.view.toPlainText()
    dest = tmp_path / "exported.log"; lp.export_to(dest)
    assert "boom" in dest.read_text(encoding="utf-8")


def test_export_video_button_creates_new_file_and_keeps_original(win, media, tmp_path):
    import hashlib
    home = win.pages["nav_home"]
    home.add_paths([str(media["multi"])]); run_to_end(win)
    home.table.setCurrentCell(0, 0)
    assert home.btn["export_video"].isEnabled()
    before = hashlib.sha256(media["multi"].read_bytes()).hexdigest()
    home.btn["export_video"].click()
    assert wait_for(lambda: "video" in win.ctx.items[0].result.outputs)
    out = Path(win.ctx.items[0].result.outputs["video"])
    assert out.exists() and out.parent == tmp_path / "out" and len(FFmpeg().probe(out).subtitle_tracks) == 2
    assert hashlib.sha256(media["multi"].read_bytes()).hexdigest() == before


def test_download_translation_from_gui_uses_one_time_window_and_restores_offline(win, tmp_path, monkeypatch):
    """Real HTTP download from a local server; only the index lookup is redirected to it."""
    import threading, http.server
    from ovt import downloader
    from test_downloader import Handler
    from test_models import make_argos
    Handler.files = {"/fr_en.argosmodel": make_argos(tmp_path / "a.argosmodel", "fr", "en").read_bytes(),
                     "/en_ar.argosmodel": make_argos(tmp_path / "b.argosmodel", "en", "ar").read_bytes()}
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:%d" % srv.server_address[1]
    index = [{"from": "fr", "to": "en", "url": base + "/fr_en.argosmodel", "version": "1"},
             {"from": "en", "to": "ar", "url": base + "/en_ar.argosmodel", "version": "1"}]
    monkeypatch.setattr(downloader, "list_argos_packages", lambda *a, **k: index)
    asked = []
    mp = win.pages["nav_models"]
    monkeypatch.setattr(mp, "confirm_online", lambda what: asked.append(what) or True)
    before = mp.table.rowCount()
    mp.download_mt("fr")
    assert wait_for(lambda: mp.job is None and mp.table.rowCount() == before + 2)
    assert "fr→en + en→ar" in asked[0] and netguard.is_offline()
    ids = [mp.table.item(r, 1).text() for r in range(mp.table.rowCount())]
    assert "argos-fr-en" in ids and "argos-en-ar" in ids
    srv.shutdown()


def test_glossary_editor_roundtrip_and_used_by_engine(win, tmp_path):
    from ovt.gui.pages_settings import GlossaryDialog
    from ovt.glossary import Glossary
    from ovt.subtitles import Cue
    from ovt.translate.service import translate_cues
    dlg = GlossaryDialog(win.ctx, "")
    dlg.setAttribute(Qt.WA_DeleteOnClose, False)
    dlg.table.item(0, 0).setText("Hogwarts")                 # empty right column = keep as is
    dlg.add_row("Dr.", "د."); dlg.add_row("", "ignored")     # rows without a term are dropped
    f = dlg.save(tmp_path / "g.json")
    assert dlg.to_json() == {"keep": ["Hogwarts"], "map": {"Dr.": "د."}}
    again = GlossaryDialog(win.ctx, str(f))
    assert again.to_json() == dlg.to_json()
    dlg.deleteLater(); again.deleteLater()
    cues = [Cue(0, 2, "", source_text="Dr. Snape teaches at Hogwarts.")]
    translate_cues(FakeMT(), cues, "en", "ar", Glossary.load(f))
    assert "Hogwarts" in cues[0].text and "د." in cues[0].text and "XQ" not in cues[0].text
