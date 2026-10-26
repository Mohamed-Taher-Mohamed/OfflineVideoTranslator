# Supported-language register (real results only)

| Source -> Arabic | ASR tested | MT tested | Tester / date | Notes |
|---|---|---|---|---|
| (none yet) | no | no | - | No real speech or translation model could be run in the development sandbox (model hosts unreachable). |

Fill this table with the output of `python -m ovt selftest-mt --src-lang <xx>` and a real short clip per language.
Argos has a direct package for en->ar only (per the official index checked on 2026-10-26); every other language reaches Arabic
by pivoting through English (two translations -> lower quality) and only if both packages are installed.
