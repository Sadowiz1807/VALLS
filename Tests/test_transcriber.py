from pathlib import Path
from types import SimpleNamespace
import wave

import numpy as np

from Voice import VoiceProcess
from Voice.transcriber import Transcriber


def test_default_voice_uses_parakeet():
    transcriber = Transcriber()
    voice = VoiceProcess()

    assert transcriber.model_name == "nvidia/parakeet-ctc-0.6b-vi"
    assert voice.transcriber.model_name == transcriber.model_name
    assert transcriber.sample_rate == voice.capture.sample_rate == 16000


def test_transcribe_writes_mono_16khz_wav_and_cleans_it():
    observed = {}

    class Model:
        def transcribe(self, paths):
            path = Path(paths[0])
            observed["path"] = path
            with wave.open(str(path), "rb") as stream:
                observed["channels"] = stream.getnchannels()
                observed["sample_rate"] = stream.getframerate()
                observed["sample_width"] = stream.getsampwidth()
            return [SimpleNamespace(text="Mở Spotify")]

    transcriber = Transcriber(min_rms=0.0)
    transcriber._model = Model()

    text, language, confidence = transcriber.transcribe(np.ones(1600, dtype=np.float32) * 0.1)

    assert (text, language, confidence) == ("Mở Spotify", "vi", 0.0)
    assert observed == {
        "path": observed["path"],
        "channels": 1,
        "sample_rate": 16000,
        "sample_width": 2,
    }
    assert not observed["path"].exists()


def test_no_speech_skips_model():
    transcriber = Transcriber(min_rms=0.02)
    transcriber._model = SimpleNamespace(
        transcribe=lambda _paths: (_ for _ in ()).throw(AssertionError("model must not run"))
    )

    assert transcriber.transcribe(np.zeros(16000, dtype=np.float32)) == ("", "vi", 0.0)


def test_voice_transcribes_only_after_explicit_stop():
    events = []
    voice = VoiceProcess(on_event=lambda event: events.append(event))

    class Capture:
        sample_rate = 16000

        def start(self):
            pass

        def poll(self, timeout=0.1):
            if voice._stop.is_set():
                return None
            return np.ones(160, dtype=np.float32) * 0.1

        def stop(self):
            pass

    class StubTranscriber:
        model_name = "nvidia/parakeet-ctc-0.6b-vi"

        def load(self):
            return self

        def transcribe(self, audio):
            assert len(audio) > 0
            return "Mở Spotify", "vi", 0.0

    voice.capture = Capture()
    voice.transcriber = StubTranscriber()
    voice.start()

    import time
    time.sleep(0.02)
    assert [event.event for event in events] == ["VOICE_STARTED"]
    voice.stop()

    assert [event.event for event in events] == ["VOICE_STARTED", "VOICE_FINAL"]
    assert events[-1].text == "Mở Spotify"
