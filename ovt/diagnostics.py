"""System self-check. Each check really exercises the component (no 'import succeeded' = OK shortcuts
where a cheap real call is possible). Used by the GUI (Logs page), `ovt diag` and `OfflineVideoTranslator.exe --diag`."""
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Tuple

Check = Tuple[str, bool, str]


def _try(name, fn) -> Check:
    try:
        return (name, True, str(fn()))
    except Exception as e:                       # a failing check must never crash the report
        return (name, False, "%s: %s" % (type(e).__name__, str(e)[:240]))


def run_diagnostics(paths) -> List[Check]:
    out: List[Check] = []

    out.append(("platform", True, "%s | python %s | frozen=%s" % (platform.platform(), platform.python_version(),
                                                                    getattr(sys, "frozen", False))))

    def folders():
        for d in (paths.models, paths.settings, paths.output, paths.logs, paths.temp):
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".write_test"
            probe.write_text("x", encoding="utf-8"); probe.unlink()
        return "all data folders writable under %s" % paths.home
    out.append(_try("writable folders", folders))

    def arabic_path():
        d = Path(tempfile.mkdtemp(prefix="ovt_")) / "مجلد عربي بمسافات"
        d.mkdir(); f = d / "ملف اختبار.txt"
        f.write_text("مرحبا", encoding="utf-8")
        ok = f.read_text(encoding="utf-8") == "مرحبا"
        shutil.rmtree(d.parent, ignore_errors=True)
        if not ok:
            raise RuntimeError("round-trip failed")
        return "Arabic file names with spaces OK"
    out.append(_try("arabic paths", arabic_path))

    def ffmpeg():
        from .media import FFmpeg
        ff = FFmpeg(paths.engines)
        v = subprocess.run([ff.ffmpeg, "-version"], capture_output=True, timeout=30).stdout.decode("utf-8", "replace").splitlines()[0]
        tmp = Path(tempfile.mkdtemp(prefix="ovt_")) / "t.wav"
        r = subprocess.run([ff.ffmpeg, "-v", "error", "-f", "lavfi", "-i", "sine=f=300:d=1", str(tmp)], capture_output=True, timeout=60)
        if r.returncode != 0 or not tmp.exists():
            raise RuntimeError("ffmpeg cannot encode: " + r.stderr.decode("utf-8", "replace")[:150])
        info = ff.probe(tmp)
        shutil.rmtree(tmp.parent, ignore_errors=True)
        return "%s | probe+encode OK (%.1fs)" % (v[:60], info.duration)
    out.append(_try("ffmpeg/ffprobe", ffmpeg))

    def ct2():
        import ctranslate2
        return "ctranslate2 %s | cpu types: %s | cuda devices: %d" % (
            ctranslate2.__version__, ",".join(sorted(ctranslate2.get_supported_compute_types("cpu"))),
            ctranslate2.get_cuda_device_count())
    out.append(_try("ctranslate2 (engine)", ct2))

    def spm():
        import sentencepiece
        return "sentencepiece %s" % sentencepiece.__version__
    out.append(_try("sentencepiece (tokenizer)", spm))

    def fw():
        import faster_whisper
        return "faster-whisper %s" % faster_whisper.__version__
    out.append(_try("faster-whisper (speech)", fw))

    def guard():
        import socket
        from . import netguard
        was = netguard.is_offline()
        netguard.enable_offline(paths.logs / "network.log")
        try:
            try:
                socket.create_connection(("93.184.216.34", 80), timeout=2).close()
            except Exception as e:
                if e.__class__.__name__ != "OfflineViolation":
                    raise RuntimeError("connection failed for another reason: %r" % e)
                return "external connection blocked by offline guard"
            raise RuntimeError("external connection was NOT blocked")
        finally:
            if not was:
                netguard.disable_offline()
    out.append(_try("offline guard", guard))

    def models():
        from .models import ModelManager
        ms = ModelManager(paths).list()
        asr = [m["id"] for m in ms if m["kind"] == "asr"]
        mt = [m["id"] for m in ms if m["kind"] == "mt"]
        if not asr or not mt:
            raise RuntimeError("missing models -> speech: %s | translation: %s (import them on the Models page)" % (asr or "none", mt or "none"))
        return "speech: %s | translation: %s" % (", ".join(asr), ", ".join(mt))
    out.append(_try("models installed", models))

    def disk():
        free = shutil.disk_usage(str(paths.home)).free
        return "%.1f GB free where the program lives" % (free / 2**30)
    out.append(_try("disk space", disk))
    return out


def format_report(checks: List[Check]) -> str:
    lines = []
    for name, ok, detail in checks:
        lines.append("[%s] %-26s %s" % ("PASS" if ok else "FAIL", name, detail))
    bad = sum(1 for _, ok, _ in checks if not ok)
    lines.append("")
    lines.append("RESULT: %d passed, %d failed" % (len(checks) - bad, bad))
    return "\n".join(lines)
