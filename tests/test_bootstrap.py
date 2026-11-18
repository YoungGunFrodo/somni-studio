import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from somni_studio.somni.bootstrap import ModelLocked, init_home, local_models, model_locked_by, set_model
from somni_studio.somni.home import HomePaths, base_model

from .helpers import SOMNI_REPO, HomeTestCase, require_somni


class InitHomeTest(unittest.TestCase):
    def setUp(self):
        require_somni(self)
        from somni_studio.somni.runtime import ensure_sleepai
        ensure_sleepai(SOMNI_REPO)
        self._tmp = tempfile.TemporaryDirectory()
        self.paths = HomePaths(Path(self._tmp.name) / "home")

    def tearDown(self):
        self._tmp.cleanup()

    def test_creates_a_blank_home(self):
        self.assertTrue(init_home(self.paths, "org/some-model"))
        cfg = json.loads(self.paths.config.read_text(encoding="utf-8"))
        self.assertEqual(cfg["base_model"], "org/some-model")
        self.assertEqual(cfg["trainer"], "experts")
        self.assertTrue(cfg["expert_route"])
        self.assertTrue(cfg["sleep"]["tags"])
        db = sqlite3.connect(self.paths.memory_db)
        self.assertEqual(db.execute("SELECT COUNT(*) FROM memories").fetchone()[0], 0, "a new home starts empty")
        db.close()

    def test_never_touches_an_existing_folder(self):
        self.paths.home.mkdir(parents=True)
        (self.paths.home / "something.txt").write_text("keep me", encoding="utf-8")
        self.assertFalse(init_home(self.paths))
        self.assertFalse(self.paths.config.exists())

    def test_set_model(self):
        init_home(self.paths, "org/a")
        set_model(self.paths, "org/b")
        self.assertEqual(base_model(self.paths), "org/b")
        with self.assertRaises(ValueError):
            set_model(self.paths, "  ")


class ModelLockTest(HomeTestCase):
    def test_experts_lock_the_model(self):
        paths = HomePaths(self.home)              # the fixture home has an expert checkpoint
        self.assertIn("learned experts for", model_locked_by(paths))
        with self.assertRaises(ModelLocked):
            set_model(paths, "org/other")
        set_model(paths, base_model(paths))      # choosing the same model is fine


class LocalModelsTest(unittest.TestCase):
    def test_finds_causal_lms_in_the_hub_cache(self):
        with tempfile.TemporaryDirectory() as d:
            hub = Path(d)
            def model(name, arch, weights=b"x" * 2048):
                folder = hub / f"models--{name.replace('/', '--')}"
                snap = folder / "snapshots" / "abc"
                snap.mkdir(parents=True)
                (snap / "config.json").write_text(json.dumps({"architectures": [arch], "num_hidden_layers": 4}))
                (folder / "blobs").mkdir()
                (folder / "blobs" / "w.safetensors").write_bytes(weights)
            model("org/chat-model", "QwenForCausalLM")
            model("org/embedder", "BertModel")
            model("org/empty", "LlamaForCausalLM", weights=b"")
            with mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(hub)}), \
                    mock.patch("pathlib.Path.home", return_value=hub / "nohome"):
                found = local_models(min_bytes=1000)
        self.assertEqual([m.id for m in found], ["org/chat-model"])
        self.assertEqual(found[0].layers, 4)


if __name__ == "__main__":
    unittest.main()
