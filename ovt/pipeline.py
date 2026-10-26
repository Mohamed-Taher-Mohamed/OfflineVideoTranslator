"""Job pipeline: probe -> (ASR in chunks | existing subtitles) -> translate -> write -> verify.

UI-independent. The UI supplies: progress callback, cancel Event, and backend factories.
Resume model (state lives in temp/jobs/<job_id>/, never next to the video):
  chunks/asr_NNNN.json   one per finished audio window, tagged with an ASR fingerprint
  translation.json       partial/complete translation cache, tagged with a translation fingerprint
Changing ASR settings invalidates ASR chunks (and therefore translation). Changing only
translation settings (glossary, MT model, target) keeps the ASR chunks and redoes translation only.
"""
import hashlib
import json
import math
import shutil
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import __version__
from .asr.base import Segment, dedupe_repeats
from .errors import (CancelledError, DiskSpaceError, ModelMissingError, OVTError,
                     UnsupportedLanguageError, UnsupportedMediaError)
from .fsutil import atomic_write_json, atomic_write_text, ensure_free_space, read_json, unique_path
from .glossary import Glossary
from .media import FFmpeg, MediaInfo, iso3_to_1
from . import subtitles as sub
from .translate.service import translate_cues


@dataclass
class JobOptions:
    input_path: str
    out_dir: str = ""                  # "" = same folder as the video (originals are never modified)
    audio_track: int = 0
    source_lang: str = ""              # "" = auto
    target_lang: str = "ar"
    asr_model_id: str = ""
    mt_model_id: str = ""
    source_mode: str = "asr"           # asr | embedded | file
    subtitle_track: int = 0            # for embedded
    subtitle_file: str = ""            # for file
    chunk_seconds: int = 300
    glossary_path: str = ""
    context_merge: bool = True
    write_txt: bool = False
    write_source_srt: bool = False
    keep_temp: bool = False
    overwrite: bool = False
    purge_job_data: bool = False
    max_chars: int = 42
    max_lines: int = 2
    min_cue_seconds: float = 0.8
    bom: bool = False
    rtl_marks: bool = False
    snap_to_silence: bool = True       # move chunk borders to pauses so sentences are not cut (kept LAST: positional-arg safe)


@dataclass
class Progress:
    stage: str
    fraction: float
    message: str = ""
    chunk: int = 0
    chunks: int = 0
    elapsed: float = 0.0
    eta: Optional[float] = None


@dataclass
class JobResult:
    job_id: str
    outputs: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    source_language: str = ""
    cue_count: int = 0
    chunks_total: int = 0
    chunks_resumed: int = 0
    no_speech: bool = False
    seconds: float = 0.0

    def to_dict(self):
        return asdict(self)


def job_id_for(path: Path, audio_track: int) -> str:
    st = Path(path).stat()
    key = "%s|%d|%d|%d" % (Path(path).resolve(), st.st_size, st.st_mtime_ns, audio_track)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _fp(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:16]


class Pipeline:
    def __init__(self, paths, ffmpeg: FFmpeg, asr_factory: Callable, translator_factory: Callable,
                 model_manager=None, progress: Optional[Callable[[Progress], None]] = None,
                 cancel: Optional[threading.Event] = None):
        self.paths, self.ff = paths, ffmpeg
        self.asr_factory, self.translator_factory = asr_factory, translator_factory
        self.models = model_manager
        self._progress = progress or (lambda p: None)
        self.cancel = cancel or threading.Event()
        self._t0 = time.time()

    # ---- helpers ----
    def _check_cancel(self):
        if self.cancel.is_set():
            raise CancelledError("Job cancelled. Progress is saved; run again to resume.")

    def _emit(self, stage, frac, msg="", chunk=0, chunks=0):
        el = time.time() - self._t0
        eta = (el / frac - el) if frac > 0.03 else None
        self._progress(Progress(stage, min(frac, 1.0), msg, chunk, chunks, el, eta))

    def _model_fp(self, kind, model_id):
        if not self.models or not model_id:
            return model_id or ""
        m = self.models.get(kind, model_id)
        key = m["files"].get("model.bin") or next(iter(m["files"].values()))
        return "%s:%s:%s" % (m["id"], m["version"], key["sha256"][:12])

    # ---- main ----
    def run(self, o: JobOptions) -> JobResult:
        self._t0 = time.time()
        src_path = Path(o.input_path)
        self._emit("probe", 0.0, "Reading file")
        info = self.ff.probe(src_path)
        if o.audio_track >= len(info.audio_tracks):
            raise UnsupportedMediaError("Audio track %d does not exist (file has %d)." % (o.audio_track, len(info.audio_tracks)))
        jid = job_id_for(src_path, o.audio_track)
        jdir = self.paths.temp / "jobs" / jid
        (jdir / "chunks").mkdir(parents=True, exist_ok=True)
        out_dir = Path(o.out_dir) if o.out_dir else src_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        ensure_free_space(out_dir, 20 * 2**20, "output files")
        res = JobResult(job_id=jid)

        glossary = Glossary.load(o.glossary_path)

        if o.source_mode == "asr":
            segs, lang = self._stage_asr(o, info, jdir, res)
            cues = [sub.Cue(s.start, s.end, "", source_text=s.text, confidence=s.confidence) for s in segs]
            asr_fp = self._asr_fp(o)
        else:
            cues, lang = self._stage_existing_subs(o, info, jdir)
            asr_fp = _fp("existing", o.source_mode, o.subtitle_track, o.subtitle_file)
        res.source_language = lang

        if not cues:
            res.no_speech = True
            res.warnings.append("No speech/subtitle text was found in this file. Nothing was written.")
            res.seconds = time.time() - self._t0
            self._emit("done", 1.0, "No speech detected")
            return res

        self._stage_translate(o, cues, lang, glossary, jdir, asr_fp, res)
        self._stage_write(o, src_path, out_dir, cues, res, info)
        if o.purge_job_data:
            shutil.rmtree(jdir, ignore_errors=True)
        res.cue_count = len(cues)
        res.seconds = time.time() - self._t0
        self._emit("done", 1.0, "Done")
        return res

    # ---- ASR ----
    def _asr_fp(self, o: JobOptions) -> str:
        return _fp("asr", self._model_fp("asr", o.asr_model_id), o.audio_track, o.chunk_seconds, o.source_lang,
                   o.snap_to_silence, "plan-v2")

    def _stage_asr(self, o, info: MediaInfo, jdir: Path, res: JobResult):
        fp = self._asr_fp(o)
        plan = self._plan(o, info, jdir, fp)
        n = len(plan)
        res.chunks_total = n
        ensure_free_space(self.paths.temp, int(o.chunk_seconds * 32000 * 2) + 50 * 2**20, "temporary audio")
        backend = None
        lang = o.source_lang or None
        segments: List[Segment] = []
        t_asr0 = time.time()
        try:
            for k in range(n):
                self._check_cancel()
                cf = jdir / "chunks" / ("asr_%04d.json" % k)
                saved = read_json(cf)
                start, end = plan[k]
                if (saved and saved.get("fingerprint") == fp and abs(saved.get("chunk_start", -1) - start) < 1e-3
                        and abs(saved.get("chunk_end", -1) - end) < 1e-3):
                    res.chunks_resumed += 1
                    if saved["segments"] and not lang:
                        lang = saved["language"]
                    segments += [Segment(**s) for s in saved["segments"]]
                    self._emit("asr", 0.7 * (k + 1) / n, "Resumed chunk %d/%d" % (k + 1, n), k + 1, n)
                    continue
                if backend is None:
                    backend = self.asr_factory()
                dur = end - start
                if dur < 0.2:
                    continue
                wav = jdir / ("tmp_%04d.wav" % k)
                self._emit("asr", 0.7 * k / n, "Extracting audio %d/%d" % (k + 1, n), k + 1, n)
                try:
                    self.ff.extract_wav(o.input_path, o.audio_track, start, dur, wav)
                    self._check_cancel()
                    self._emit("asr", 0.7 * (k + 0.4) / n, "Recognising speech %d/%d" % (k + 1, n), k + 1, n)
                    clang, prob, segs = backend.transcribe(wav, lang)
                finally:
                    if not o.keep_temp and wav.exists():
                        wav.unlink()
                for s in segs:
                    s.start += start
                    s.end += start
                segs = dedupe_repeats(segs)
                if segs and not lang:
                    lang = clang
                atomic_write_json(cf, {"fingerprint": fp, "language": clang, "language_prob": prob,
                                       "chunk_start": start, "chunk_end": end, "segments": [s.to_dict() for s in segs]})
                segments += segs
                self._emit("asr", 0.7 * (k + 1) / n, "Chunk %d/%d done" % (k + 1, n), k + 1, n)
        finally:
            if backend is not None:
                backend.close()
        segments.sort(key=lambda s: s.start)
        return segments, (lang or "")

    def _plan(self, o, info, jdir: Path, fp: str):
        """Chunk plan is computed once and stored, so a resumed job uses exactly the same borders."""
        pf = jdir / "plan.json"
        saved = read_json(pf)
        if saved and saved.get("fingerprint") == fp:
            return [tuple(x) for x in saved["plan"]]
        self._emit("asr", 0.0, "Planning chunks")
        plan = self.ff.plan_chunks(o.input_path, o.audio_track, info.duration, o.chunk_seconds, o.snap_to_silence)
        atomic_write_json(pf, {"fingerprint": fp, "plan": plan})
        return plan

    # ---- existing subtitles ----
    def _stage_existing_subs(self, o, info: MediaInfo, jdir: Path):
        if o.source_mode == "embedded":
            if o.subtitle_track >= len(info.subtitle_tracks):
                raise UnsupportedMediaError("Subtitle track %d does not exist." % o.subtitle_track)
            tr = info.subtitle_tracks[o.subtitle_track]
            if not tr.is_text:
                raise UnsupportedMediaError("Track %d is image-based (%s); OCR is not included." % (o.subtitle_track, tr.codec))
            f = self.ff.extract_subtitle(o.input_path, o.subtitle_track, jdir / "embedded.srt")
            lang = o.source_lang or iso3_to_1(tr.language)
        else:
            f, lang = Path(o.subtitle_file), o.source_lang
        cues = sub.load_subtitles(f)
        for c in cues:
            c.source_text, c.text = c.text, ""
        return cues, lang

    # ---- translation ----
    def _stage_translate(self, o, cues, lang, glossary, jdir, asr_fp, res):
        tgt = o.target_lang
        if not lang:
            raise UnsupportedLanguageError("Source language could not be determined. Pass --src-lang.")
        if lang == tgt:
            for c in cues:
                c.text = c.source_text
            res.warnings.append("Source language equals target language: text copied without translation.")
            return
        translator = self.translator_factory()
        try:
            sup = translator.supports(lang, tgt)
            if not sup.ok:
                raise UnsupportedLanguageError(
                    "Cannot translate '%s' -> '%s': %s. Import a suitable translation model; no fake translation is produced."
                    % (lang, tgt, sup.reason), src=lang, tgt=tgt)
            if "pivot" in sup.route:
                res.warnings.append("Translation goes %s: quality is lower than a direct model." % sup.route)
            tfp = _fp("mt", asr_fp, self._model_fp("mt", o.mt_model_id), glossary.fingerprint(), lang, tgt, o.context_merge, len(cues))
            tf = jdir / "translation.json"
            saved = read_json(tf)
            cache = saved["cache"] if saved and saved.get("fingerprint") == tfp else {}

            def on_batch(c, done, total):
                atomic_write_json(tf, {"fingerprint": tfp, "cache": c})
                self._emit("translate", 0.7 + 0.25 * done / max(total, 1), "Translating %d/%d" % (done, total))

            self._emit("translate", 0.7, "Translating")
            res.warnings += translate_cues(translator, cues, lang, tgt, glossary, o.context_merge,
                                           cache=cache, on_batch=on_batch, check_cancel=self._check_cancel)
            atomic_write_json(tf, {"fingerprint": tfp, "cache": cache})
        finally:
            translator.close()

    # ---- output ----
    def _target(self, out_dir: Path, src_path: Path, suffix: str, o) -> Path:
        p = out_dir / (src_path.stem + suffix)
        if p.resolve() == src_path.resolve():
            raise OVTError("Refusing to overwrite the original file")
        return p if o.overwrite else unique_path(p)

    def _stage_write(self, o, src_path, out_dir, cues, res, info):
        self._emit("write", 0.96, "Writing files")
        sub.fix_timing(cues, o.min_cue_seconds)
        final = sub.reflow(cues, o.max_chars, o.max_lines)
        final = sub.fix_timing(final, 0.3)
        srt = self._target(out_dir, src_path, ".%s.srt" % o.target_lang, o)
        sub.write_srt(final, srt, o.bom, o.rtl_marks)
        res.outputs["srt"] = str(srt)
        if o.write_txt:
            p = self._target(out_dir, src_path, ".transcript.txt", o)
            atomic_write_text(p, sub.to_txt(cues))
            res.outputs["txt"] = str(p)
        if o.write_source_srt:
            src_cues = [sub.Cue(c.start, c.end, c.source_text) for c in cues]
            p = self._target(out_dir, src_path, ".source.srt", o)
            sub.write_srt(sub.fix_timing(src_cues, o.min_cue_seconds), p, o.bom)
            res.outputs["source_srt"] = str(p)
        proj = self._target(out_dir, src_path, ".ovtproject.json", o)
        atomic_write_json(proj, {"ovt_version": __version__, "source_file": src_path.name,
                                 "source_language": res.source_language, "target_language": o.target_lang,
                                 "options": asdict(o), "warnings": res.warnings,
                                 "cues": [c.to_dict() for c in cues]})
        res.outputs["project"] = str(proj)
        # verify what is actually on disk before claiming success
        problems = sub.validate_srt(srt)
        if problems:
            res.warnings.append("OUTPUT VALIDATION FAILED: " + "; ".join(problems[:5]))
            raise OVTError("Generated SRT failed validation: " + "; ".join(problems[:5]))
        if not srt.stat().st_size:
            raise OVTError("Generated SRT is empty")
