import importlib.util
import unittest

from somni_studio.settings import SettingsError, StudioSettings

from .helpers import SOMNI_REPO, require_somni

HAVE_TORCH = importlib.util.find_spec("torch") is not None


class DtypeSettingTest(unittest.TestCase):
    def test_default_is_auto_and_choices_are_validated(self):
        self.assertEqual(StudioSettings().provider.dtype, "auto")
        self.assertEqual(StudioSettings().merged({"provider": {"dtype": "float16"}}).provider.dtype, "float16")
        with self.assertRaisesRegex(SettingsError, "provider.dtype"):
            StudioSettings().merged({"provider": {"dtype": "fp8"}})


@unittest.skipUnless(HAVE_TORCH, "torch is not installed")
class PrecisionTest(unittest.TestCase):
    def test_explicit_choices(self):
        import torch

        from somni_studio.engine.precision import resolve_dtype
        self.assertIs(resolve_dtype("float16"), torch.float16)
        self.assertIs(resolve_dtype("bfloat16"), torch.bfloat16)
        self.assertIn(resolve_dtype("auto"), (torch.float16, torch.bfloat16, torch.float32))

    def test_somni_loader_dtype_is_patched_and_restored(self):
        require_somni(self)
        import torch

        from somni_studio.engine.precision import somni_load_dtype
        from somni_studio.somni.runtime import ensure_sleepai
        ensure_sleepai(SOMNI_REPO)
        from sleepai.models import peft_trainer
        original = peft_trainer.pick_device
        device, _ = original()
        with somni_load_dtype(torch.float16):
            self.assertEqual(peft_trainer.pick_device(), (device, torch.float16))
        self.assertIs(peft_trainer.pick_device, original)
        with self.assertRaises(RuntimeError), somni_load_dtype(torch.float16):
            raise RuntimeError("load failed")
        self.assertIs(peft_trainer.pick_device, original, "restored even when loading fails")


if __name__ == "__main__":
    unittest.main()
