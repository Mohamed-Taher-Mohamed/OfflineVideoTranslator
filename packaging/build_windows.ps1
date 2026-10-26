# Local build on a Windows PC that already has Python 3.12 (alternative to the GitHub workflow).
# Run from the project root in PowerShell:  .\packaging\build_windows.ps1
$ErrorActionPreference = "Stop"
python -m pip install -r requirements.txt
if (-not (Test-Path engines\ffmpeg.exe)) { Write-Error "Put ffmpeg.exe and ffprobe.exe in engines\ first"; exit 1 }
$env:QT_QPA_PLATFORM = "offscreen"; python -m pytest -q; Remove-Item Env:QT_QPA_PLATFORM
python -m PyInstaller --noconfirm --clean packaging\ovt.spec
$root = "dist\OfflineVideoTranslator"
foreach ($d in "engines","models","settings","output","logs","temp") { New-Item -ItemType Directory -Force "$root\$d" | Out-Null }
Copy-Item engines\* "$root\engines" -Force
Copy-Item packaging\README_PORTABLE_AR.txt "$root\README.txt"
Write-Host "Done: $root\OfflineVideoTranslator.exe"
