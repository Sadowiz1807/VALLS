"""NVIDIA Parakeet transcription: float32 mono 16 kHz audio -> text."""
from __future__ import annotations

import tempfile
import wave
from pathlib import Path
from typing import Tuple

import numpy as np


class Transcriber:
    def __init__(
        self,
        model_name: str = "nvidia/parakeet-ctc-0.6b-vi",
        sample_rate: int = 16000,
        min_rms: float = 0.02,
    ):
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.min_rms = min_rms
        self._model = None

    def load(self) -> "Transcriber":
        import nemo.collections.asr as nemo_asr

        self._model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name).eval()
        return self

    def transcribe(self, audio) -> Tuple[str, str, float]:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not samples.size or float(np.sqrt(np.mean(samples * samples))) < self.min_rms:
            return "", "vi", 0.0
        if self._model is None:
            self.load()

        path = None
        try:
            with tempfile.NamedTemporaryFile(prefix="parakeet-", suffix=".wav", delete=False) as handle:
                path = Path(handle.name)
            pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
            with wave.open(str(path), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(self.sample_rate)
                stream.writeframes(pcm.tobytes())
            hypotheses = self._model.transcribe([str(path)])
            first = hypotheses[0] if hypotheses else ""
            text = first.text if hasattr(first, "text") else str(first)
            return text.strip(), "vi", 0.0
        finally:
            if path is not None:
                path.unlink(missing_ok=True)
