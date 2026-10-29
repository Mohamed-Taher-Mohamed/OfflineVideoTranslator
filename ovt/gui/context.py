"""Shared state handed to every page (keeps pages independent of each other)."""
from dataclasses import dataclass, field
from typing import Any, List, Optional

from ..media import MediaInfo
from .i18n import I18n


@dataclass
class QueueItem:
    path: str
    info: MediaInfo
    audio_track: int = 0
    source_mode: str = "asr"
    subtitle_track: int = 0
    subtitle_file: str = ""
    src_lang: str = ""
    status: str = "queued"      # queued running paused done error cancelled nospeech
    progress: float = 0.0
    message: str = ""
    result: Any = None
    error: str = ""
    detected_lang: str = ""
    range_start: float = 0.0       # translate only this part (seconds, absolute); 0/0 = whole file
    range_end: float = 0.0


class Ctx:
    def __init__(self, paths, settings, models, builder, ffmpeg_factory):
        self.paths, self.settings, self.models = paths, settings, models
        self.builder = builder                  # (JobOptions, progress_cb, cancel_event) -> Pipeline
        self.ffmpeg_factory = ffmpeg_factory
        self.i18n = I18n(settings.ui_language)
        self.items: List[QueueItem] = []

    def t(self, key):
        return self.i18n.t(key)
