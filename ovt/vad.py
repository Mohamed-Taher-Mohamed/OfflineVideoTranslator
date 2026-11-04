"""Speech regions from the audio itself (Silero VAD bundled with faster-whisper)."""
from typing import List, Tuple


def speech_regions(samples, min_silence_ms: int = 300, sr: int = 16000) -> List[Tuple[float, float]]:
    """[(start, end)] seconds of detected speech, no padding (true edges)."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    chunks = get_speech_timestamps(samples, VadOptions(min_silence_duration_ms=min_silence_ms, speech_pad_ms=0))
    return [(c["start"] / sr, c["end"] / sr) for c in chunks]
