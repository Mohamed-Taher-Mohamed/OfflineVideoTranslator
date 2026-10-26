"""Typed errors. Each has a stable `code` so any UI (CLI, Qt, Android bridge) can localise it."""


class OVTError(Exception):
    code = "error"

    def __init__(self, message: str, **details):
        super().__init__(message)
        self.details = details


class MediaError(OVTError):
    code = "media"


class UnsupportedMediaError(MediaError):
    code = "media_unsupported"


class NoAudioError(MediaError):
    code = "media_no_audio"


class ModelMissingError(OVTError):
    code = "model_missing"


class ModelIntegrityError(OVTError):
    code = "model_integrity"


class UnsupportedLanguageError(OVTError):
    code = "language_unsupported"


class OfflineViolation(OVTError):
    code = "offline_violation"


class DiskSpaceError(OVTError):
    code = "disk_space"


class CancelledError(OVTError):
    code = "cancelled"


class SubtitleFormatError(OVTError):
    code = "subtitle_format"
