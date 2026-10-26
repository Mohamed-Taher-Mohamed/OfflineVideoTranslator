# Status report (engine + CLI + GUI + packaging) — updated

## 1. Verified for real (automated tests, real ffmpeg / CTranslate2 / SentencePiece / Qt / PyInstaller)
| Area | Evidence |
|---|---|
| Probe, multi audio tracks, subtitle tracks, Arabic paths with spaces, 16 kHz mono extraction | test_media |
| 3-hour file: window at 2h30 extracted in 0.5 s, Python peak RAM 35 MB | `pytest -m slow -s` |
| Corrupt/empty/no-audio/missing/wrong-track -> clear typed errors | test_media, test_cli |
| SRT write/validate (UTF-8, order, overlap), VTT/ASS/SRT read, cp1256 fallback, no text dropped on long lines | test_subtitles |
| Our Arabic SRT muxed by ffmpeg and read back with identical text/timing | test_selftest_and_mux |
| Offline guard: external connect + DNS blocked and logged; loopback allowed | test_offline, GUI "Privacy" test button |
| Model manager: import/verify/delete, ids, path traversal, failed re-import keeps old, checksums | test_models |
| Original video never modified; results never overwritten without --overwrite | test_pipeline |
| **Light translation engine (CTranslate2 + SentencePiece, no torch)**: Argos-package and NLLB-folder loading, pivot via English, language-token handling, zip-slip protection, output-length cap — executed for real with a TINY RANDOM-WEIGHT model (gibberish output by design) | test_ct2_backend |
| Same engine running **inside the frozen PyInstaller build** (model import + selftest-mt) | manual run, see section 5 |
| GUI (headless): build in AR/EN + light/dark, queue, add Arabic-named file, bad/duplicate file, run in background thread, pause/resume (finished chunks reused), cancel, remove, retry, editor edit/shift/reset/save+validate, models import/verify/default/delete, settings persist/reset, language switch, privacy test, error log export | test_gui |
| Export new video with the Arabic track added (stream copy, MKV/MP4, verified by probe, never overwrites, GUI button) | test_export, test_gui |
| **Model downloader** (resumable, size/checksum verified, cancellable) tested against a REAL local HTTP server incl. an interrupted transfer; one-time online window restores offline mode even after errors and is written to network.log | test_downloader, test_gui |
| Argos index fetched for real through the guarded window (100 packages); route logic checked on real data: every language reaches Arabic via `xx>en + en>ar` (only en/ar have direct pairs to Arabic) | manual run |
| **Silence-aware chunk borders** (ffmpeg silencedetect; border moved into the longest pause within ±20 s; plan stored so resume reuses identical borders; different plan = no mixing) | test_chunk_plan |
| Glossary editor dialog (keep-as-is vs forced translation) feeding the engine | test_gui |
| MT repetition-loop collapse + warning (normal repeated words untouched) | test_translate_service |
| **Frozen build run end to end (Linux)**: GUI starts and closes itself (rc 0); `ovt-cli translate --subs-file` -> real CTranslate2 translation -> valid SRT/TXT/source SRT/project JSON | manual run |
| System diagnostics (`ovt diag`, `--diag`, Logs page button) | test_diagnostics + frozen build run |
| PyInstaller build of GUI exe + console CLI exe built and run on Linux; faster-whisper VAD asset, ctranslate2, sentencepiece, onnxruntime bundled | manual build |
| Real faster-whisper loader given a damaged model -> clear error + errors.log | test_cli |

## 2. Verified only with SIMULATED ASR/MT (logic, not quality)
Chunking, progress, cancel+resume, fingerprints (ASR change redoes ASR; glossary change redoes only translation), no-speech handling, unsupported-pair refusal, glossary, disk-full refusal, GUI job flow. (conftest.Fake*).

## 3. NOT verified — do not assume it works
- **Real Whisper recognition and real translation quality (any language).** Hugging Face / model hosts are unreachable from the sandbox.
- Real NLLB / Argos model files (only tiny random stand-ins were run). Argos packages using a BPE tokenizer instead of SentencePiece are rejected with a message (not supported yet).
- NLLB source-token convention (`[src_lang] tokens </s>` with `[tgt_lang]` target prefix) follows the CTranslate2 documentation; if a specific conversion differs, output will be poor — selftest-mt will show it.
- **Windows**: the .exe was never built or run on Windows. The GitHub workflow (`.github/workflows/build-windows.yml`) was written but not executed. Windows paths/console behaviour untested.
- The real download hosts (Hugging Face for Whisper, argos-net.com for Argos packages) were unreachable from the sandbox: the downloader logic is tested, the real URLs are not. `download_asr` uses `huggingface_hub.snapshot_download` and the repo names `Systran/faster-whisper-*` (from memory).
- Real speed/RAM of ASR, GPU, Android, iOS, playback in VLC/MPC.
- OS-level proof that no byte leaves the PC (guard covers the Python process; ffmpeg limited by `-protocol_whitelist file,pipe`).
- Visual design was reviewed only through offscreen screenshots (Linux fonts), not on a real Windows display.

## 4. Known issues / limits
1. Pause == cancel with saved state; resume granularity = one audio chunk (default 300 s) / one translation batch.
2. Chunk borders now snap to pauses when a pause exists within ±20 s; continuous speech with no pause is still cut at the nominal border.
3. Language detected from the first chunk with speech; mixed-language videos not handled.
4. Context merge re-splits Arabic text over cues by source length (approximate boundaries).
5. Pivot translation (xx->en->ar) lower quality (flagged). NLLB gives direct pairs (preferred when installed).
6. Image subtitles (PGS/DVD) refused (no OCR).
7. Hashes are taken at import: they detect damage, not provenance. Catalog sizes are approximate.
8. Job state with recognised TEXT stays in `temp/jobs/` after success (for cheap re-translation); audio temp files are deleted. Privacy page can wipe it.
9. SRT is UTF-8 without BOM by default (setting available).
10. NOT implemented: burned-in (hard) subtitles and video export with a dubbed audio track, dubbing, Android app, iOS, glossary editor UI (glossary is a JSON file chosen in Settings), speaker diarization.
11. The PyInstaller output is large (≈626 MB on Linux) mainly because of ctranslate2/onnxruntime/Qt; size on Windows will differ.

## 5. Frozen-build evidence (Linux, this sandbox)
`ovt-cli models import` (tiny NLLB-style folder) -> OK; `ovt-cli selftest-mt --src-lang en` -> ran CTranslate2 + SentencePiece from the frozen app and honestly reported `has_target_script:false` (random model);
`OfflineVideoTranslator --diag` -> 9 of 10 checks pass, only "models installed" fails (no real models) — the expected honest result.

## 6. What YOU do at the end (one session, in this order)
A. Get the Windows build without installing anything: create a GitHub repo, upload this project, Actions -> "Build Windows Portable" -> Run workflow, download the artifact zip. If a step fails, send me the red log text.
B. Unzip; double-click `OfflineVideoTranslator.exe`; Logs page -> **System check**; send me the text (only "models installed" should fail).
C. Models page -> **Download**: (1) a speech model (start with `whisper-small`, ~480 MB), (2) "Translation to Arabic from…" -> English (downloads `en>ar`). Confirm the one-time internet window. If a download fails, send me the message. (Fallback: download the files in a browser and use Import.)
D. `ovt-cli.exe selftest-mt --src-lang en` -> send me the JSON.
E. Translate a 1-2 minute English clip; open the SRT in your player; tell me what is wrong in the Arabic. Then try one clip in another language (French/Spanish/...: it will pivot through English; download that language's package first).
F. Disconnect the internet and repeat E; confirm `logs\network.log` has no new BLOCKED lines and Privacy -> "Test internet blocking" says blocked.
G. Optional: "Export video with subtitles" on a finished job; check the new file in your player.
