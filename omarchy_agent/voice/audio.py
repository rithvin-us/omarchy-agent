"""Microphone capture with simple energy-based endpointing, and WAV playback."""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Callable

import numpy as np

SAMPLE_RATE = 16_000
CHUNK_MS = 30
CHUNK_BYTES = SAMPLE_RATE * CHUNK_MS // 1000 * 2  # s16 mono


class AudioError(RuntimeError):
    """Recording or playback is unavailable."""


def _rms(chunk: bytes) -> float:
    samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0


def record_utterance(
    *,
    wait_seconds: float = 10.0,
    silence_seconds: float = 0.9,
    max_seconds: float = 30.0,
    should_stop: Callable[[], bool] = lambda: False,
) -> np.ndarray | None:
    """Record one spoken utterance. Returns float32 mono 16 kHz audio, or None if silent."""
    if not shutil.which("arecord"):
        raise AudioError("arecord not found (install alsa-utils)")
    proc = subprocess.Popen(  # noqa: S603
        ["arecord", "-q", "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "1", "-t", "raw"],  # noqa: S607
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if proc.stdout is None:
        raise AudioError("could not open microphone stream")
    frames: list[bytes] = []
    noise: list[float] = []
    threshold = 0.0
    started = False
    silent_chunks = 0
    t0 = time.monotonic()
    need_silent = int(silence_seconds * 1000 / CHUNK_MS)
    try:
        while time.monotonic() - t0 < max_seconds + wait_seconds and not should_stop():
            chunk = proc.stdout.read(CHUNK_BYTES)
            if len(chunk) < CHUNK_BYTES:
                break
            level = _rms(chunk)
            if len(noise) < 10:  # ~0.3 s ambient calibration
                noise.append(level)
                threshold = max(300.0, float(np.mean(noise)) * 3.0)
                continue
            if not started:
                if level > threshold:
                    started = True
                    frames.append(chunk)
                elif time.monotonic() - t0 > wait_seconds:
                    return None
                continue
            frames.append(chunk)
            silent_chunks = silent_chunks + 1 if level < threshold else 0
            if silent_chunks >= need_silent:
                break
    finally:
        proc.terminate()
        proc.wait(timeout=2)
    if not started:
        return None
    pcm = np.frombuffer(b"".join(frames), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def play_wav(path: str) -> None:
    for player in ("pw-play", "paplay", "aplay"):
        exe = shutil.which(player)
        if exe:
            subprocess.run([exe, path], check=False, timeout=300)  # noqa: S603
            return
    raise AudioError("no audio player found (pw-play, paplay or aplay)")
