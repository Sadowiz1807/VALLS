"""Voice process: microphone capture -> NVIDIA Parakeet -> VOICE_FINAL text.

Sở hữu microphone stream và transcriber. Chỉ `VOICE_FINAL` gửi tiếp; không lưu audio.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional
from uuid import uuid4

import numpy as np

from .capture import MicrophoneCapture
from .transcriber import Transcriber


@dataclass
class VoiceEvent:
    event: str
    text: str = ""
    language: str = "vi"
    source: str = "microphone"
    confidence: float = 0.0
    request_id: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.request_id:
            self.request_id = uuid4().hex[:12]
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class VoiceProcess:
    def __init__(
        self,
        model_name: str = "nvidia/parakeet-ctc-0.6b-vi",
        sample_rate: int = 16000,
        min_rms: float = 0.02,
        on_event: Optional[Callable[[VoiceEvent], None]] = None,
    ):
        self.capture = MicrophoneCapture(sample_rate=sample_rate)
        self.transcriber = Transcriber(model_name=model_name, sample_rate=sample_rate, min_rms=min_rms)
        self.on_event = on_event
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()

    def _emit(self, event: VoiceEvent):
        if self.on_event:
            self.on_event(event)

    def _run(self):
        try:
            self.transcriber.load()
            self.capture.start()
        except Exception as exc:  # không có mic / model lỗi
            self._emit(VoiceEvent(event="VOICE_ERROR", text=str(exc)))
            return
        self._emit(VoiceEvent(event="VOICE_STARTED"))
        buffer = []
        try:
            while not self._stop.is_set():
                chunk = self.capture.poll(timeout=0.1)
                if chunk is None:
                    continue
                buffer.append(chunk)
        finally:
            self.capture.stop()
        if buffer:
            audio = np.concatenate(buffer) if len(buffer) > 1 else buffer[0]
            text, lang, conf = self.transcriber.transcribe(audio)
            self._emit(VoiceEvent(
                event="VOICE_FINAL" if text else "VOICE_CANCELLED",
                text=text,
                language=lang,
                confidence=conf,
            ))


if __name__ == "__main__":
    def handler(event: VoiceEvent):
        if event.event == "VOICE_FINAL":
            print(f"[FINAL] {event.text} (conf={event.confidence:.2f})")
        else:
            print(f"[{event.event}]")

    process = VoiceProcess(on_event=handler)
    print("Listening... press Ctrl+C to stop")
    try:
        process.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        process.stop()
        print("Stopped.")
