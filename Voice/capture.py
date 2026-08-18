"""Microphone capture (sounddevice): raw float32 mono chunks at 16 kHz."""
from __future__ import annotations

import queue
from typing import Optional

import numpy as np


class MicrophoneCapture:
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._queue: queue.Queue = queue.Queue()
        self._stream = None

    def start(self):
        import sounddevice as sd  # deferred: cần thiết bị audio

        def callback(indata, frames, t_info, status):
            if status:
                return
            self._queue.put(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            callback=callback,
        )
        self._stream.start()

    def poll(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
