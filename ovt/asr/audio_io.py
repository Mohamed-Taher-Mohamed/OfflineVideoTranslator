"""Read the 16 kHz mono PCM WAV that ffmpeg produced into the float32 array faster-whisper accepts.
Reading it ourselves avoids faster-whisper's PyAV decoding path (which broke with newer `av` releases)."""
import wave

import numpy as np

from ..errors import MediaError


def read_wav_float32(path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1 or w.getframerate() != 16000:
            raise MediaError("Expected 16 kHz mono 16-bit WAV, got %d Hz / %d ch / %d-bit"
                             % (w.getframerate(), w.getnchannels(), w.getsampwidth() * 8))
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
