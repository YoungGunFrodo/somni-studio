"""How Somni sounds: blended voices, the assistant effect, pacing."""
import io
import unittest
import wave

import numpy as np

from somni_studio.settings import SettingsError, VoiceSettings
from somni_studio.speech import effects
from somni_studio.speech.service import KokoroVoice, VoiceStyle

RATE = 24000


class EffectTest(unittest.TestCase):
    def test_assistant_effect(self):
        t = np.arange(RATE) / RATE
        tone = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        out = effects.assistant(tone, RATE)
        self.assertEqual(out.size, tone.size + int(RATE * effects.TAIL_MS / 1000), "a short room tail")
        self.assertAlmostEqual(float(np.max(np.abs(out))), effects.PEAK, places=3)
        self.assertTrue(np.isfinite(out).all())
        self.assertGreater(float(np.abs(out[tone.size:]).max()), 0.001, "the room rings on briefly")
        self.assertIs(effects.apply("none", tone, RATE).dtype, np.dtype(np.float32))
        self.assertEqual(effects.assistant(np.zeros(0, np.float32), RATE).size, 0)


class FakeKokoro:
    def __init__(self):
        self.voices_used = []

    def get_voice_style(self, name):
        return np.full((4,), {"bm_lewis": 1.0, "bm_george": 3.0}[name], dtype=np.float32)

    def create(self, text, voice, speed, lang):
        self.voices_used.append(voice)
        return np.full(RATE // 10, 0.1, dtype=np.float32), RATE


class SynthTest(unittest.TestCase):
    def synth(self, style):
        voice = KokoroVoice.__new__(KokoroVoice)
        fake = FakeKokoro()
        voice._load = lambda: fake
        wav = voice.synth("Good evening, sir.", style)
        with wave.open(io.BytesIO(wav)) as w:
            return fake.voices_used[0], w.getnframes()

    def test_blend_effect_and_pause(self):
        used, frames = self.synth(VoiceStyle("bm_lewis"))
        self.assertEqual((used, frames), ("bm_lewis", RATE // 10))
        used, frames = self.synth(VoiceStyle("bm_lewis", blend="bm_george", blend_amount=0.25, pause=0.5))
        np.testing.assert_allclose(used, np.full(4, 1.5), err_msg="75% Lewis + 25% George")
        self.assertEqual(frames, RATE // 10 + RATE // 2)
        _, frames = self.synth(VoiceStyle("bm_lewis", effect="assistant"))
        self.assertGreater(frames, RATE // 10)

    def test_style_from_settings_with_overrides(self):
        settings = VoiceSettings(voice="bm_lewis", effect="assistant", pause=0.2)
        style = VoiceStyle.of(settings, speed=0.9, blend=None)
        self.assertEqual((style.voice, style.speed, style.effect, style.pause, style.blend),
                         ("bm_lewis", 0.9, "assistant", 0.2, ""))

    def test_settings_are_checked(self):
        for bad in ({"effect": "robot"}, {"blend_amount": 0.9}, {"pause": 3.0}):
            with self.subTest(bad=bad), self.assertRaises(SettingsError):
                VoiceSettings(**bad).validate()


if __name__ == "__main__":
    unittest.main()
