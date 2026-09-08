"""Voice process: microphone capture -> VAD -> NVIDIA Parakeet -> VOICE_FINAL text."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional
from uuid import uuid4

from .capture import MicrophoneCapture
from .transcriber import Transcriber
from .vad import VadConfig, UtteranceSegmenter


@dataclass
class VoiceEvent:
    event: str
    text: str = ""
    language: str = "vi"
    source: str = "microphone"
    confidence: float = 0.0
    request_id: str = ""
    utterance_id: str = ""
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
        vad_config: VadConfig | None = None,
        capture_factory: Callable[[], MicrophoneCapture] | None = None,
        transcriber: Transcriber | None = None,
    ):
        self.capture_factory = capture_factory or (lambda: MicrophoneCapture(sample_rate=sample_rate))
        self.capture = self.capture_factory()
        self.transcriber = transcriber or Transcriber(
            model_name=model_name, sample_rate=sample_rate, min_rms=min_rms,
        )
        self.segmenter = UtteranceSegmenter(vad_config or VadConfig(sample_rate=sample_rate))
        self.on_event = on_event
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._utterance_id = ""

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()

    def _emit(self, event: VoiceEvent):
        if self.on_event:
            try:
                self.on_event(event)
            except Exception:
                pass

    def _new_utterance(self):
        self._utterance_id = uuid4().hex
        self._emit(VoiceEvent(event="VOICE_SPEECH_STARTED", utterance_id=self._utterance_id))

    def _finalize(self, audio, restart: bool):
        utterance_id = self._utterance_id or uuid4().hex
        self.capture.stop()
        try:
            text, lang, conf = self.transcriber.transcribe(audio)
            self._emit(VoiceEvent(
                event="VOICE_FINAL" if text else "VOICE_CANCELLED",
                text=text, language=lang, confidence=conf, utterance_id=utterance_id,
            ))
        except Exception as exc:
            self._emit(VoiceEvent(event="VOICE_ERROR", text=str(exc), utterance_id=utterance_id))
        finally:
            self._utterance_id = ""
        if restart and not self._stop.is_set():
            self.capture = self.capture_factory()
            self.capture.start()

    def _run(self):
        try:
            self.transcriber.load()
            self.capture.start()
        except Exception as exc:
            self._emit(VoiceEvent(event="VOICE_ERROR", text=str(exc)))
            return
        self._emit(VoiceEvent(event="VOICE_STARTED"))
        try:
            while not self._stop.is_set():
                chunk = self.capture.poll(timeout=0.1)
                if chunk is None:
                    continue
                before = self.segmenter.state
                segments = self.segmenter.feed(chunk)
                if before == "IDLE" and self.segmenter.state == "LISTENING":
                    self._new_utterance()
                for audio in segments:
                    self._finalize(audio, restart=True)
        finally:
            self.capture.stop()
        audio = self.segmenter.flush()
        if audio is not None:
            self._finalize(audio, restart=False)


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
