from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VadConfig:
    sample_rate: int = 16000
    frame_ms: int = 20
    noise_calibration_ms: int = 300
    noise_floor_alpha: float = 0.95
    speech_start_ratio: float = 3.0
    speech_stop_ratio: float = 1.5
    min_start_frames: int = 3
    pre_roll_ms: int = 100
    min_speech_ms: int = 200
    silence_ms: int = 500
    max_utterance_ms: int = 15000

    def __post_init__(self):
        if self.sample_rate <= 0 or not 20 <= self.frame_ms <= 30:
            raise ValueError("VAD_CONFIG_INVALID")
        if not 0.0 < self.noise_floor_alpha < 1.0:
            raise ValueError("VAD_CONFIG_INVALID")
        if self.speech_start_ratio <= self.speech_stop_ratio or self.speech_stop_ratio <= 1.0:
            raise ValueError("VAD_CONFIG_INVALID")
        if min(self.min_start_frames, self.min_speech_ms, self.silence_ms, self.max_utterance_ms) <= 0:
            raise ValueError("VAD_CONFIG_INVALID")


class UtteranceSegmenter:
    def __init__(self, config: VadConfig = VadConfig()):
        self.config = config
        self.state = "IDLE"
        self.noise_floor = 0.0
        self._calibration: list[float] = []
        pre_roll_frames = max(1, config.pre_roll_ms // config.frame_ms)
        self._pre_roll: deque[np.ndarray] = deque(maxlen=pre_roll_frames + config.min_start_frames)
        self._candidate = 0
        self._frames: list[np.ndarray] = []
        self._speech_frames = 0
        self._silence_frames = 0
        self._listening_frames = 0

    @staticmethod
    def _rms(frame: np.ndarray) -> float:
        samples = np.asarray(frame, dtype=np.float32).reshape(-1)
        return float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0

    def feed(self, frame: np.ndarray) -> list[np.ndarray]:
        samples = np.asarray(frame, dtype=np.float32).reshape(-1)
        rms = self._rms(samples)
        calibration_frames = max(1, self.config.noise_calibration_ms // self.config.frame_ms)
        if len(self._calibration) < calibration_frames:
            self._calibration.append(rms)
            self.noise_floor = float(np.mean(self._calibration))
            self._pre_roll.append(samples.copy())
            return []

        start_threshold = max(self.noise_floor * self.config.speech_start_ratio, 1e-6)
        stop_threshold = max(self.noise_floor * self.config.speech_stop_ratio, 1e-6)
        if self.state == "IDLE":
            self._pre_roll.append(samples.copy())
            if rms >= start_threshold:
                self._candidate += 1
                if self._candidate >= self.config.min_start_frames:
                    self.state = "LISTENING"
                    self._frames = list(self._pre_roll)
                    self._speech_frames = self._candidate
                    self._listening_frames = self._candidate
                    self._silence_frames = 0
            else:
                self._candidate = 0
                alpha = self.config.noise_floor_alpha
                self.noise_floor = alpha * self.noise_floor + (1.0 - alpha) * rms
            return []

        self._frames.append(samples.copy())
        self._listening_frames += 1
        if rms <= stop_threshold:
            self._silence_frames += 1
        else:
            self._speech_frames += 1
            self._silence_frames = 0
        silence_frames = max(1, self.config.silence_ms // self.config.frame_ms)
        max_frames = max(1, self.config.max_utterance_ms // self.config.frame_ms)
        if self._silence_frames >= silence_frames or self._listening_frames >= max_frames:
            audio = self._finalize()
            return [audio] if audio is not None else []
        return []

    def flush(self) -> np.ndarray | None:
        return self._finalize() if self.state == "LISTENING" else None

    def _finalize(self) -> np.ndarray | None:
        self.state = "FINALIZING"
        min_frames = max(1, self.config.min_speech_ms // self.config.frame_ms)
        audio = np.concatenate(self._frames) if self._speech_frames >= min_frames and self._frames else None
        self.state = "IDLE"
        self._candidate = 0
        self._frames = []
        self._speech_frames = 0
        self._silence_frames = 0
        self._listening_frames = 0
        self._pre_roll.clear()
        return audio
