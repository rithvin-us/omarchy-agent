"""Local speech-to-text (faster-whisper) and text-to-speech (Piper)."""

from __future__ import annotations

import os
import re
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np

from .audio import AudioError, play_wav

DATA_DIR = Path("~/.local/share/omarchy-agent").expanduser()
WHISPER_DIR = DATA_DIR / "whisper"
VOICE_DIR = DATA_DIR / "voices"
DEFAULT_WHISPER = "small.en"
DEFAULT_VOICE = "en_US-lessac-medium"
MAX_SPOKEN_CHARS = 1200

__all__ = ["Speaker", "Transcriber", "speakable"]


def speakable(markdown: str) -> str:
    """Strip code blocks, markdown marks, and URLs so speech sounds natural."""
    text = re.sub(r"```.*?```", " (code omitted) ", markdown, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"https?://\S+", " link ", text)
    text = re.sub(r"[*_#>|~]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_SPOKEN_CHARS]


class Transcriber:
    def __init__(self, model: str = DEFAULT_WHISPER, device: str = "auto") -> None:
        self.model_name = model
        self.device = device
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from faster_whisper import WhisperModel

            device = self.device
            compute = "int8"
            if device == "auto":
                device = "cpu"  # CUDA needs extra libs; int8 CPU is fast enough for small.en
            self._model = WhisperModel(
                self.model_name,
                device=device,
                compute_type=compute,
                download_root=str(WHISPER_DIR),
            )
        return self._model

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _info = self._load().transcribe(audio, language="en", vad_filter=True)
        return " ".join(seg.text.strip() for seg in segments).strip()


class Speaker:
    def __init__(self, voice: str = DEFAULT_VOICE) -> None:
        self.voice_name = voice
        self._voice: Any = None

    def _load(self) -> Any:
        if self._voice is None:
            model = VOICE_DIR / f"{self.voice_name}.onnx"
            if not model.is_file():
                raise AudioError(f"Piper voice not found: {model}")
            from piper import PiperVoice

            self._voice = PiperVoice.load(str(model))
        return self._voice

    def say(self, text: str) -> None:
        spoken = speakable(text)
        if not spoken:
            return
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="omarchy-tts-")
        os.close(fd)
        try:
            with wave.open(path, "wb") as wav:
                self._load().synthesize_wav(spoken, wav)
            play_wav(path)
        finally:
            Path(path).unlink(missing_ok=True)
