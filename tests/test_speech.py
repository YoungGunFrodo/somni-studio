import importlib.util
import threading
import unittest

from somni_studio.settings import VoiceSettings
from somni_studio.speech.service import SpeechService
from somni_studio.speech.text import SentenceSplitter, speakable

HAVE_KOKORO = importlib.util.find_spec("kokoro_onnx") is not None


def stream(splitter, text, step=3):
    out = []
    for i in range(0, len(text), step):
        out += splitter.feed(text[i:i + step])
    return out


class SplitterTest(unittest.TestCase):
    def test_sentences_arrive_while_streaming(self):
        s = SentenceSplitter()
        said = stream(s, "Nikola Tesla was born in 1856. He worked on alternating current. And more")
        self.assertEqual(said, ["Nikola Tesla was born in 1856.", "He worked on alternating current."])
        self.assertEqual(s.finish(), ["And more"])
        self.assertEqual(s.finish(), [], "nothing is spoken twice")

    def test_citations_and_markdown_are_not_read_aloud(self):
        self.assertEqual(speakable("It is **Normandy**. [memory: normans/named]"), "It is Normandy.")
        self.assertEqual(speakable("Born in 1856 [nikola-tesla/born]"), "Born in 1856")

    def test_short_fragments_join_the_next_sentence(self):
        s = SentenceSplitter()
        said = stream(s, "Yes. The Normans named Normandy. ")
        self.assertEqual(said, ["Yes. The Normans named Normandy."])

    def test_long_clauses_are_cut_at_a_comma(self):
        s = SentenceSplitter()
        text = "word, " * 80
        said = s.feed(text)
        self.assertTrue(said and all(len(x) <= 262 for x in said))

    def test_replace_and_finish_with_final_text(self):
        s = SentenceSplitter()
        self.assertEqual(s.feed("First sentence here. Sec"), ["First sentence here."])
        self.assertEqual(s.feed("First sentence here. Second one.", replace=True), [])
        self.assertEqual(s.finish("First sentence here. Second one."), ["Second one."])

    def test_unstreamed_answer(self):
        s = SentenceSplitter()
        self.assertEqual(s.finish("A whole answer at once. Two sentences."),
                         ["A whole answer at once.", "Two sentences."])


class ServiceWithoutModelTest(unittest.TestCase):
    def test_missing_model_disables_speech_cleanly(self):
        svc = SpeechService(VoiceSettings(model_dir="does/not/exist"), lambda e: None)
        try:
            status = svc.status()
            self.assertFalse(status["ready"])
            self.assertIn("missing" if HAVE_KOKORO else "not installed", status["problem"])
            self.assertIsNone(svc.session("r1"), "no session, so answers are just not spoken")
            self.assertTrue(svc.voices(), "voice names are still listed")
        finally:
            svc.shutdown()


@unittest.skipUnless(HAVE_KOKORO, "kokoro-onnx is not installed")
class ServiceTest(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.got = threading.Event()
        self.svc = SpeechService(VoiceSettings(threads=2), self._publish)
        if not self.svc.status()["ready"]:
            self.svc.shutdown()
            self.skipTest(self.svc.status()["problem"])

    def _publish(self, event):
        self.events.append(event)
        if len(self.events) >= 2:
            self.got.set()

    def tearDown(self):
        self.svc.shutdown()

    def test_preview_is_a_wav(self):
        clip = self.svc.preview("Hello there.")
        audio = self.svc.clip(clip)
        self.assertTrue(audio.startswith(b"RIFF") and b"WAVE" in audio[:16])

    def test_session_speaks_sentences_in_order(self):
        session = self.svc.session("req")
        for piece in ("The Normans ", "named Normandy. ", "Tesla was born in 1856."):
            session.feed(piece)
        session.finish("The Normans named Normandy. Tesla was born in 1856.")
        self.assertTrue(self.got.wait(60))
        self.assertEqual([(e["seq"], e["text"]) for e in self.events],
                         [(0, "The Normans named Normandy."), (1, "Tesla was born in 1856.")])
        self.assertTrue(all(self.svc.clip(e["clip"]) for e in self.events))


if __name__ == "__main__":
    unittest.main()
