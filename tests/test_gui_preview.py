"""Preview page + live panel. Playback tests use REAL QtMultimedia when it is installed, else they are skipped."""
import os
import time
from pathlib import Path

os.environ.setdefault("OVT_PREVIEW_AUDIO", "0")      # sandbox has no audio stack; video/subtitle sync is what is tested

import pytest
from PySide6.QtWidgets import QApplication

from test_gui import win, wait_for, run_to_end, qapp, SlowASR  # noqa: F401  (fixtures)
from ovt.gui.pages_preview import HAVE_MM
from ovt.gui.main_window import NAV


def finished_job(win, media):
    home = win.pages["nav_home"]
    home.add_paths([str(media["multi"])]); run_to_end(win)
    home.table.setCurrentCell(0, 0)
    return home


def test_live_panel_shows_recognised_and_translated_text(win, media):
    home = finished_job(win, media)
    live = home.live.toPlainText()
    assert "[0:01] This is sentence number 1." in live          # recognised text, with its time
    assert "EN: This is sentence number" in live and "AR: " in live  # source -> translation pairs
    assert "فيديو اختبار 1.mkv" in live


def test_preview_button_opens_page_with_synced_subtitles(win, media):
    home = finished_job(win, media)
    assert home.btn["preview_btn"].isEnabled()
    home.btn["preview_btn"].click()
    pv = win.pages["nav_preview"]
    assert win.stack.currentWidget() is pv and len(pv.cues) == 3 and pv.table.rowCount() == 3
    assert pv.video_path.endswith("فيديو اختبار 1.mkv")
    pv.on_position(1500)                                         # 1.5 s: inside cue 1 (1.0-2.5)
    assert pv.cues[0].text in pv.lbl_tr.text() and "This is sentence number" in pv.lbl_src.text()
    assert pv.table.currentRow() == 0
    pv.on_position(4000)                                         # gap between cues
    assert pv.lbl_tr.text() == "" and pv.lbl_src.text() == ""
    pv.on_position(6500)
    assert pv.table.currentRow() == 1 and pv.cues[1].text in pv.lbl_tr.text()


def test_clicking_a_row_and_prev_next_jump_between_cues(win, media):
    home = finished_job(win, media)
    home.btn["preview_btn"].click()
    pv = win.pages["nav_preview"]
    pv.on_row_clicked(2, 0)
    assert pv.cur == 2 and pv.cues[2].text in pv.lbl_tr.text()
    pv.on_position(int(pv.cues[2].start * 1000) + 100)
    pv.step(-1)
    assert pv.cur == 1
    pv.step(+1)
    assert pv.cur == 2


def test_preview_opens_plain_srt_and_asks_for_video_when_unknown(win, media, tmp_path):
    pv = win.pages["nav_preview"]
    pv.load_subs(str(media["srt"]))
    assert len(pv.cues) == 2 and pv.video_path == ""
    assert pv.ctx.t("pv_need_video") in pv.status.text()
    pv.on_position(1500)
    assert pv.lbl_tr.text() == "Hello world."


@pytest.mark.skipif(not HAVE_MM, reason="PySide6-Addons (QtMultimedia) not installed")
def test_real_playback_advances_and_subtitle_follows(win, media, tmp_path):
    from PySide6.QtMultimedia import QMediaPlayer
    pv = win.pages["nav_preview"]
    from ovt import subtitles as sub
    pv.set_cues([sub.Cue(0.0, 3.0, "مرحبا", source_text="hello"), sub.Cue(3.0, 8.0, "العالم", source_text="world")])
    errors = []
    pv.player.errorOccurred.connect(lambda e, m: errors.append(m))
    pv.load_video(str(media["multi"]))
    pv.player.play()
    assert wait_for(lambda: pv.player.position() > 1200, timeout=20), "playback did not advance; errors=%s" % errors
    assert not errors and pv.player.duration() > 10000
    assert pv.lbl_tr.text() in ("مرحبا", "العالم")                  # label driven by the REAL position signal
    pv.player.pause()
    pv.seek_ms(5000)
    assert wait_for(lambda: pv.lbl_tr.text() == "العالم", timeout=10)
    assert pv.slider.value() >= 4000
    pv.player.stop()


def test_without_multimedia_the_page_still_loads(win, media, monkeypatch):
    """Graceful degradation path: pages_preview handles HAVE_MM == False (labels/table keep working)."""
    pv = win.pages["nav_preview"]
    pv.load_subs(str(media["srt"]))
    pv.show_cue(1)
    assert pv.lbl_tr.text() == "Second line here."


def _changed(a, b, band, thr=60):
    """Pixels inside a horizontal band (fractions of the height) that differ between two grabbed images.
    Comparing 'with subtitle' against 'without' is font-independent (Windows draws thinner text than Linux)."""
    h, w = a.height(), a.width()
    n = 0
    for y in range(int(h * band[0]), int(h * band[1]), 2):
        for x in range(0, w, 2):
            ca, cb = a.pixelColor(x, y), b.pixelColor(x, y)
            if abs(ca.red() - cb.red()) + abs(ca.green() - cb.green()) + abs(ca.blue() - cb.blue()) > thr:
                n += 1
    return n


def _bright(img, band):
    h, w = img.height(), img.width()
    return sum(1 for y in range(int(h * band[0]), int(h * band[1]), 2) for x in range(0, w, 2)
               if min(img.pixelColor(x, y).red(), img.pixelColor(x, y).green(), img.pixelColor(x, y).blue()) > 200)


def test_subtitle_is_painted_on_the_picture_itself(qapp):
    from PySide6.QtGui import QColor, QImage
    from ovt.gui.pages_preview import VideoCanvas
    cv = VideoCanvas(); cv.resize(640, 360)
    frame = QImage(1280, 720, QImage.Format_RGB32); frame.fill(QColor(20, 60, 200))
    cv.set_frame(frame)
    blank = cv.grab().toImage()
    assert blank.pixelColor(320, 120).blue() > 150                                            # the frame itself is drawn
    cv.set_subtitle("مرحبا بالعالم، هذا اختبار للترجمة")
    with_sub = cv.grab().toImage()
    assert _changed(with_sub, blank, (0.78, 0.97)) > 300                                      # box + text over the lower part
    assert _bright(with_sub, (0.78, 0.97)) >= 8                                               # some bright text pixels
    assert _changed(with_sub, blank, (0.0, 0.5)) == 0                                         # nothing on the top half
    cv.set_subtitle("")
    assert _changed(cv.grab().toImage(), blank, (0.0, 1.0)) == 0                              # cleared -> identical again
    # 4:3 picture inside a wide widget keeps its aspect ratio (black bars left and right)
    sq = QImage(400, 300, QImage.Format_RGB32); sq.fill(QColor(200, 30, 30)); cv.resize(800, 300); cv.set_frame(sq)
    vr = cv.video_rect()
    assert abs(vr.width() / vr.height() - 4 / 3) < 0.01 and vr.x() > 100


def test_page_overlay_follows_the_current_cue_and_optional_source_line(win, media):
    from PySide6.QtGui import QColor, QImage
    pv = win.pages["nav_preview"]
    frame = QImage(640, 360, QImage.Format_RGB32); frame.fill(QColor(30, 30, 30))
    pv.canvas.set_frame(frame); pv.canvas.resize(640, 360)
    pv.load_subs(str(media["srt"]))
    pv.on_position(1500)
    assert pv.canvas.text == "Hello world." and pv.canvas.src == ""
    pv.cb_src_overlay.setChecked(True)
    pv.show_cue(pv.cur)
    pv.on_position(1500)
    pv.on_position(4500)
    pv.on_position(6500)
    assert pv.canvas.text == "" or pv.canvas.text == "Second line here."
    pv.on_position(4500)
    assert pv.canvas.text == "Second line here."


@pytest.mark.skipif(not HAVE_MM, reason="PySide6-Addons (QtMultimedia) not installed")
def test_real_decoded_frames_reach_the_overlay_canvas(win, media):
    pv = win.pages["nav_preview"]
    from ovt import subtitles as sub
    from ovt.gui.pages_preview import VideoCanvas
    pv.set_cues([sub.Cue(0.0, 8.0, "ترجمة فوق الفيديو", source_text="subtitle on video")])
    pv.load_video(str(media["multi"]))
    pv.player.play()
    assert wait_for(lambda: pv.canvas.frame is not None and pv.player.position() > 800, timeout=20)
    assert (pv.canvas.frame.width(), pv.canvas.frame.height()) == (160, 120)                 # the real decoded frame size
    assert pv.canvas.text == "ترجمة فوق الفيديو"
    pv.player.pause()
    real_frame = pv.canvas.frame.copy()                                                       # a REAL decoded video frame
    # Paint it on a STANDALONE canvas: a canvas inside the page is resized by its layout (the first CI run on Windows
    # showed 493x252 although we asked for 960x540), which made the glyphs tiny and the pixel count platform-dependent.
    big = VideoCanvas(); big.resize(960, 540); big.set_frame(real_frame)
    without = big.grab().toImage()
    big.set_subtitle(pv.canvas.text, "")
    with_text = big.grab().toImage()
    assert (with_text.width(), with_text.height()) == (960, 540)
    # The test video is BLACK, so the dark subtitle box changes nothing; only glyph pixels differ (platform font).
    assert _changed(with_text, without, (0.55, 1.0)) > 50                                     # text painted over real video
    pv.player.stop()
