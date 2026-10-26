# Offline Video Translator — Phase 1 (engine + GUI + packaging scripts)

Status: **engine and GUI tested headless; Windows EXE not yet built/tested; real models not yet tested**. See `docs/STATUS.md` for exactly what was and was not verified.

## Layout
```
ovt/                 engine (no UI code inside)
  media.py           ffprobe/ffmpeg: probe, window extraction (16 kHz mono WAV), subtitle-track extraction
  asr/               ASRBackend interface + faster-whisper backend (swappable)
  translate/         Translator interface + Argos backend + context-aware batch service
  subtitles.py       SRT/VTT/ASS read, SRT write+validate, wrap/split, timing shift/reset
  pipeline.py        probe -> ASR chunks -> translate -> write -> verify; resume via fingerprints
  models.py          import / verify / delete models (download is a separate explicit command)
  netguard.py        offline enforcement + network.log
  cli.py             test CLI (kept for diagnostics / automation)
  gui/               PySide6 GUI (queue, editor, models, settings, logs, privacy)
  diagnostics.py     system self-check
  translate/ct2_backend.py   light MT engine: CTranslate2+SentencePiece (Argos packages + NLLB folders)
tests/               95 tests (+1 opt-in 3-hour-file test)
packaging/           PyInstaller spec, Windows build script, portable README
.github/workflows/   build-windows.yml (builds the portable zip on GitHub, nothing to install locally)
engines/ models/ settings/ output/ logs/ temp/   portable data folders (everything lives next to the program)
```

## Run the tests
```
pip install -r requirements.txt
python -m pytest                 # fast suite
python -m pytest -m slow -s      # 3-hour-file streaming test (~90 s)
```

## Run the GUI (development)
```
python -m ovt.gui          # or: python OfflineVideoTranslator.py
```

## Use (CLI)
```
python -m ovt probe "D:\فيديو\film.mkv"
python -m ovt models import --kind asr --id whisper-small --path "D:\models\faster-whisper-small" --license MIT
python -m ovt models import --kind mt  --id en-ar --path "D:\models\translate-en_ar-1_0.argosmodel"
python -m ovt models verify --kind asr --id whisper-small
python -m ovt selftest-mt --src-lang en                      # real-model check on your machine
python -m ovt translate "D:\فيديو\film.mkv" --asr-model whisper-small --mt-model en-ar --txt --source-srt
python -m ovt translate film.mkv --asr-model whisper-small --use-embedded-subs 0     # translate an existing track
python -m ovt reset-settings        # settings only; models/outputs untouched
python -m ovt clean-jobs            # delete saved job state (transcripts of partial jobs)
```
Offline mode is ON by default: any non-loopback connection raises an error and is logged in `logs/network.log`.
Only `models download <key> --allow-network` may use the network.

## Outputs (never overwrite the original video; existing results get a `.1` suffix unless `--overwrite`)
`Movie.ar.srt`, optional `Movie.transcript.txt` (the Arabic text) and `Movie.source.srt`, and `Movie.ovtproject.json` (all cues, times, source text, options, warnings).
