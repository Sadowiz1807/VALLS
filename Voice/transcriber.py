"""Faster-Whisper transcription: audio array -> (text, language, confidence)."""
from __future__ import annotations

from typing import Optional, Tuple

class Transcriber:
    def __init__(self, model_size: str = "small", device: str = "auto",
                 compute_type: str = "float16", language: str = "vi"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None

    def load(self) -> "Transcriber":
        from faster_whisper import WhisperModel  # deferred: nặng
        self._model = WhisperModel(self.model_size, device=self.device,
                                   compute_type=self.compute_type)
        return self

    def transcribe(self, audio) -> Tuple[str, str, float]:
        """audio: float32 mono np.ndarray tại 16 kHz → (text, language, confidence)."""
        if self._model is None:
            self.load()
        segments, info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=1,
            vad_filter=True,  # faster-whisper dùng Silero VAD nội bộ
        )
        text = " ".join(seg.text for seg in segments)
        return text.strip(), info.language or self.language, float(info.avg_logprob)
