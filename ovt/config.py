"""Application home + persisted settings. Portable: everything lives under APP_HOME.

APP_HOME resolution: $OVT_HOME, else the folder that contains the program
(so copying the folder to another PC / USB drive carries settings and models).
"""
import os
import sys
from dataclasses import dataclass, asdict, fields
from pathlib import Path

from .fsutil import atomic_write_json, read_json


def app_home() -> Path:
    env = os.environ.get("OVT_HOME")
    if env:
        return Path(env)
    if getattr(sys, "frozen", False):  # PyInstaller
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


class Paths:
    def __init__(self, home=None):
        self.home = Path(home) if home else app_home()
        self.models = self.home / "models"
        self.settings = self.home / "settings"
        self.output = self.home / "output"
        self.logs = self.home / "logs"
        self.temp = self.home / "temp"
        self.engines = self.home / "engines"

    def ensure(self):
        for p in (self.models, self.settings, self.output, self.logs, self.temp):
            p.mkdir(parents=True, exist_ok=True)
        return self


@dataclass
class Settings:
    offline_mode: bool = True            # default ON: no network allowed
    ui_language: str = "ar"
    target_language: str = "ar"
    source_language: str = ""            # "" = auto-detect
    asr_model_id: str = ""
    mt_model_id: str = ""
    audio_track: int = 0                 # index among audio streams
    chunk_seconds: int = 300
    device: str = "auto"                 # auto|cpu|cuda
    compute_type: str = "auto"
    max_chars_per_line: int = 42
    max_lines: int = 2
    min_cue_seconds: float = 0.8
    utf8_bom: bool = False
    rtl_marks: bool = False
    keep_temp: bool = False
    write_txt: bool = False
    write_source_srt: bool = False
    glossary_path: str = ""
    output_dir: str = ""                 # "" = next to each video
    context_merge: bool = True
    beam_size: int = 3                   # 1 = fastest, 3 = balanced, 5 = most accurate
    theme: str = "light"
    speech_sensitivity: str = "normal"   # "normal" | "high" (also catches quiet / background speech)
    mixed_languages: bool = False        # video contains several spoken languages
    mt_engine: str = "nmt"               # "nmt" = built-in (Argos/NLLB), "llm" = language model via an OpenAI-compatible server
    llm_base_url: str = "http://127.0.0.1:11434/v1"
    llm_model: str = ""
    llm_api_key: str = ""
    llm_cloud_ok: bool = False           # explicit consent to send subtitle TEXT to a non-local server

    @classmethod
    def load(cls, paths: Paths) -> "Settings":
        data = read_json(paths.settings / "settings.json", {}) or {}
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})

    def save(self, paths: Paths) -> None:
        atomic_write_json(paths.settings / "settings.json", asdict(self))

    @staticmethod
    def reset(paths: Paths) -> None:
        """Reset settings only; never touches models/ or output/."""
        f = paths.settings / "settings.json"
        if f.exists():
            f.unlink()
