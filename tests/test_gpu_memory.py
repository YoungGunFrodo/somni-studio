"""Graphics memory: an answer that runs out fails clearly instead of spilling into RAM (and is cleaned up)."""
import unittest
from unittest import mock

from somni_studio.engine import gpu_memory
from somni_studio.engine.agent import LocalBackend


class FakeTokens(dict):
    def __init__(self):
        super().__init__(input_ids=mock.Mock(shape=(1, 10)))

    def to(self, _device):
        return self

    @property
    def input_ids(self):
        return self["input_ids"]


class GpuMemoryTest(unittest.TestCase):
    def test_out_of_memory_becomes_a_clear_answer(self):
        tok = mock.Mock(return_value=FakeTokens(), pad_token_id=0)
        model = mock.Mock(device="cuda", _expert_ctx=None)
        model.generate.side_effect = RuntimeError("HIP out of memory. Tried to allocate 2.00 GiB")
        with mock.patch.object(gpu_memory, "release") as release, \
                self.assertRaisesRegex(gpu_memory.OutOfGraphicsMemory, "graphics memory"):
            LocalBackend(model, tok)._generate("hi", 8, stop_unless_tool=False)
        release.assert_called_once()

    def test_other_errors_pass_through(self):
        tok = mock.Mock(return_value=FakeTokens(), pad_token_id=0)
        model = mock.Mock(device="cuda", _expert_ctx=None)
        model.generate.side_effect = RuntimeError("shape mismatch")
        with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            LocalBackend(model, tok)._generate("hi", 8, stop_unless_tool=False)

    def test_no_gpu_is_fine(self):
        with mock.patch.object(gpu_memory, "_torch", return_value=None):
            gpu_memory.cap()
            gpu_memory.release()
            self.assertEqual(gpu_memory.usage(), {})


if __name__ == "__main__":
    unittest.main()
