import importlib.util
import io
import unittest
import wave

from somni_studio.settings import ListenSettings
from somni_studio.speech.listen import BadAudio, Transcriber, read_wav, split_wake_word

HAVE_WHISPER = importlib.util.find_spec("faster_whisper") is not None
HAVE_KOKORO = importlib.util.find_spec("kokoro_onnx") is not None


def wav_bytes(samples, rate=16000) -> bytes:
    import numpy as np
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


class WakeWordTest(unittest.TestCase):
    def test_heard(self):
        for text, command in [("Hey, Somni, turn on the kitchen lights.", "Turn on the kitchen lights."),
                              ("Somny what time is it", "What time is it"), ("okay somnee, go to sleep", "Go to sleep"),
                              ("Somni.", "")]:
            with self.subTest(text=text):
                result = split_wake_word(text, "somni")
                self.assertTrue(result.heard)
                self.assertEqual(result.command, command)

    def test_not_heard(self):
        for text in ("Turn on the lights", "Hey Sammy turn it on", "Sunny day today",
                     "I told Tom about somni yesterday", ""):
            with self.subTest(text=text):
                self.assertFalse(split_wake_word(text, "somni").heard)

    def test_other_wake_words(self):
        self.assertTrue(split_wake_word("Jarvis, lights off", "jarvis").heard)


class WavTest(unittest.TestCase):
    def test_read_and_resample(self):
        import numpy as np
        samples = read_wav(wav_bytes(np.zeros(48000), rate=48000))
        self.assertEqual(len(samples), 16000)
        self.assertEqual(samples.dtype, np.float32)

    def test_bad_audio(self):
        with self.assertRaises(BadAudio):
            read_wav(b"definitely not a wav")


class MissingModelTest(unittest.TestCase):
    def test_reports_what_is_missing_without_importing_it(self):
        import sys
        t = Transcriber(ListenSettings(model_dir="does/not/exist"))
        self.assertFalse(t.status()["ready"])
        self.assertNotIn("ctranslate2", sys.modules, "importing it here could abort the process (OpenMP)")


@unittest.skipUnless(HAVE_WHISPER and HAVE_KOKORO, "needs faster-whisper and kokoro-onnx")
class RoundTripTest(unittest.TestCase):
    """Speak with Kokoro, hear with Whisper (in its worker process)."""

    def test_spoken_command_is_transcribed(self):
        import numpy as np

        from somni_studio.settings import VoiceSettings
        from somni_studio.speech.service import KokoroVoice, VoiceStyle
        voice = KokoroVoice(VoiceSettings().model_path, 2)
        listener = Transcriber(ListenSettings(threads=2))
        if voice.missing() or listener.missing():
            self.skipTest(voice.missing() or listener.missing())
        wav = voice.synth("Hey Somni, turn on the kitchen lights.", VoiceStyle("am_michael"))
        with wave.open(io.BytesIO(wav)) as w:
            samples = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
            rate = w.getframerate()
        try:
            text = listener.transcribe(wav_bytes(samples, rate))
        finally:
            listener.shutdown()
        heard = split_wake_word(text, "somni")
        self.assertTrue(heard.heard, text)
        self.assertIn("kitchen light", heard.command.lower())


if __name__ == "__main__":
    unittest.main()
