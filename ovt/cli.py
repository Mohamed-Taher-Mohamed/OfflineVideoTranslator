"""TEMPORARY command-line front end used to verify the engine. The product UI is the GUI (next step)."""
import argparse
import json
import sys
import threading
from pathlib import Path

from . import __version__, netguard
from .config import Paths, Settings
from .errors import OVTError
from .media import FFmpeg
from .models import ASR_CATALOG, ModelManager
from .pipeline import JobOptions, Pipeline
from .preview_logic import parse_clock
from .translate.llm_backend import engine_tag


def _p(args) -> Paths:
    return Paths(args.home).ensure() if args.home else Paths().ensure()


def build_pipeline(paths, settings, opts: JobOptions, progress=None, cancel=None) -> Pipeline:
    mm = ModelManager(paths)
    ff = FFmpeg(paths.engines)

    def asr_factory():
        from .asr.faster_whisper_backend import FasterWhisperBackend
        m = mm.require("asr", opts.asr_model_id)
        return FasterWhisperBackend(m["path"], settings.device, settings.compute_type, beam_size=settings.beam_size,
                                     sensitivity=settings.speech_sensitivity, multilingual=settings.mixed_languages,
                                     word_timestamps=settings.smart_split,
                                     vad_min_silence_ms=500 if settings.smart_split else 2000)

    def mt_factory():
        if settings.mt_engine == "llm":
            from .translate.llm_backend import LLMTranslator
            return LLMTranslator(settings.llm_base_url, settings.llm_model, settings.llm_api_key, settings.llm_cloud_ok)
        from .translate.ct2_backend import CT2Translator
        pk = []
        for m in mm.list("mt"):
            if opts.mt_model_id and m["id"] != opts.mt_model_id:
                continue
            mm.require("mt", m["id"])
            if m.get("mt_type") == "nllb":
                pk.append(("nllb", Path(m["path"])))
            else:
                pk += [("argos", Path(m["path"]) / f) for f in m["files"] if f.endswith(".argosmodel")]
        if not pk:
            from .errors import ModelMissingError
            raise ModelMissingError("No translation model is installed. Import one (Models page / `models import --kind mt`).")
        return CT2Translator(pk, paths.models / "mt-extracted", settings.device, settings.compute_type)

    return Pipeline(paths, ff, asr_factory, mt_factory, mm, progress, cancel)


def _log_error(paths, exc) -> Path:
    import datetime
    import traceback
    f = paths.logs / "errors.log"
    try:
        with open(f, "a", encoding="utf-8") as fh:
            fh.write("=== %s ===\n%s\n" % (datetime.datetime.now().isoformat(timespec="seconds"),
                     "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))))
    except OSError:
        pass
    return f


def force_utf8_console():
    """Windows consoles/pipes often use cp1252/cp437: printing Arabic paths or JSON would crash."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass            # frozen windowed exe has no console streams


def main(argv=None) -> int:
    force_utf8_console()
    ap = argparse.ArgumentParser(prog="ovt", description="Offline Video Translator (engine test CLI)")
    ap.add_argument("--home", help="app home (default: program folder or $OVT_HOME)")
    ap.add_argument("--allow-network", action="store_true", help="only for the download command")
    sp = ap.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("probe"); s.add_argument("file")
    s = sp.add_parser("translate")
    s.add_argument("file"); s.add_argument("--out-dir", default="")
    s.add_argument("--asr-model", default=""); s.add_argument("--mt-model", default="")
    s.add_argument("--src-lang", default=""); s.add_argument("--tgt-lang", default="ar")
    s.add_argument("--audio-track", type=int, default=0)
    s.add_argument("--use-embedded-subs", type=int, default=None, metavar="N")
    s.add_argument("--subs-file", default="")
    s.add_argument("--glossary", default=""); s.add_argument("--txt", action="store_true")
    s.add_argument("--source-srt", action="store_true"); s.add_argument("--chunk-seconds", type=int, default=300)
    s.add_argument("--from", dest="t_from", default="", help="start time, e.g. 9:15 or 1:02:03")
    s.add_argument("--to", dest="t_to", default="", help="end time, e.g. 12:15")
    s.add_argument("--overwrite", action="store_true"); s.add_argument("--keep-temp", action="store_true")
    m = sp.add_parser("models"); ms = m.add_subparsers(dest="mcmd", required=True)
    ms.add_parser("list")
    i = ms.add_parser("import")
    i.add_argument("--kind", choices=["asr", "mt"], required=True); i.add_argument("--id", required=True)
    i.add_argument("--path", required=True); i.add_argument("--license", default=""); i.add_argument("--source", default="")
    v = ms.add_parser("verify"); v.add_argument("--kind", required=True); v.add_argument("--id", required=True)
    d = ms.add_parser("delete"); d.add_argument("--kind", required=True); d.add_argument("--id", required=True)
    dl = ms.add_parser("download"); dl.add_argument("key", choices=list(ASR_CATALOG))
    s = sp.add_parser("selftest-mt"); s.add_argument("--src-lang", required=True); s.add_argument("--tgt-lang", default="ar")
    sp.add_parser("diag")
    sp.add_parser("reset-settings")
    sp.add_parser("clean-jobs")
    a = ap.parse_args(argv)

    paths = _p(a)
    settings = Settings.load(paths)
    allow_net = a.allow_network and a.cmd == "models" and getattr(a, "mcmd", "") == "download"
    if settings.offline_mode and not allow_net:
        netguard.enable_offline(paths.logs / "network.log")
    try:
        if a.cmd == "probe":
            print(json.dumps(FFmpeg(paths.engines).probe(a.file).to_dict(), ensure_ascii=False, indent=2))
        elif a.cmd == "models":
            mm = ModelManager(paths)
            if a.mcmd == "list":
                for x in mm.list():
                    print("%-4s %-20s %-10s %8.1f MB  %s  [%s]" % (x["kind"], x["id"], x["version"] or "-",
                          x["size_bytes"] / 2**20, ",".join(x["languages"]), x["license"]))
            elif a.mcmd == "import":
                x = mm.import_model(a.kind, a.path, a.id, license=a.license, source=a.source)
                print("imported", x["id"], "(%d files, %.1f MB)" % (len(x["files"]), x["size_bytes"] / 2**20))
            elif a.mcmd == "verify":
                pr = mm.verify(a.kind, a.id)
                print("OK" if not pr else "PROBLEMS: " + "; ".join(pr)); return 0 if not pr else 2
            elif a.mcmd == "delete":
                mm.delete(a.kind, a.id); print("deleted")
            elif a.mcmd == "download":
                print("downloading", a.key); mm.download_asr(a.key); print("done")
        elif a.cmd == "selftest-mt":
            from .selftest import run_mt_selftest
            opts0 = JobOptions("")
            opts0.mt_model_id = ""
            tr = build_pipeline(paths, settings, opts0).translator_factory()
            print(json.dumps(run_mt_selftest(tr, a.src_lang, a.tgt_lang), ensure_ascii=False, indent=2))
        elif a.cmd == "diag":
            from .diagnostics import format_report, run_diagnostics
            rep = format_report(run_diagnostics(paths))
            print(rep)
            return 0 if "0 failed" in rep.splitlines()[-1] else 4
        elif a.cmd == "reset-settings":
            Settings.reset(paths); print("settings reset (models and output untouched)")
        elif a.cmd == "clean-jobs":
            import shutil
            shutil.rmtree(paths.temp / "jobs", ignore_errors=True); print("job data removed")
        elif a.cmd == "translate":
            mode = "embedded" if a.use_embedded_subs is not None else ("file" if a.subs_file else "asr")
            opts = JobOptions(
                input_path=a.file, out_dir=a.out_dir, audio_track=a.audio_track, source_lang=a.src_lang,
                target_lang=a.tgt_lang, asr_model_id=a.asr_model or settings.asr_model_id,
                mt_model_id=a.mt_model or settings.mt_model_id, source_mode=mode,
                subtitle_track=a.use_embedded_subs or 0, subtitle_file=a.subs_file, chunk_seconds=a.chunk_seconds,
                glossary_path=a.glossary or settings.glossary_path, context_merge=settings.context_merge,
                write_txt=a.txt or settings.write_txt, write_source_srt=a.source_srt or settings.write_source_srt,
                keep_temp=a.keep_temp, overwrite=a.overwrite, max_chars=settings.max_chars_per_line,
                max_lines=settings.max_lines, min_cue_seconds=settings.min_cue_seconds, bom=settings.utf8_bom,
                rtl_marks=settings.rtl_marks, translator_tag=engine_tag(settings),
                speech_sensitivity=settings.speech_sensitivity, mixed_languages=settings.mixed_languages,
                range_start=parse_clock(a.t_from), range_end=parse_clock(a.t_to),
                smart_split=settings.smart_split, split_pause=settings.split_pause)
            cancel = threading.Event()
            last = [""]

            def prog(p):
                line = "[%-9s] %5.1f%%  %s" % (p.stage, p.fraction * 100, p.message)
                if line != last[0]:
                    print(line, file=sys.stderr); last[0] = line
            pl = build_pipeline(paths, settings, opts, prog, cancel)
            try:
                r = pl.run(opts)
            except KeyboardInterrupt:
                cancel.set(); print("interrupted; progress kept, re-run to resume", file=sys.stderr); return 130
            print(json.dumps(r.to_dict(), ensure_ascii=False, indent=2))
        return 0
    except OVTError as e:
        _log_error(paths, e)
        print("ERROR [%s]: %s" % (e.code, e), file=sys.stderr)
        return 1
    except Exception as e:                      # last-resort net: never show a raw traceback to the user
        log = _log_error(paths, e)
        print("ERROR [internal]: %s: %s\n(details saved to %s)" % (type(e).__name__, str(e)[:300], log), file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
