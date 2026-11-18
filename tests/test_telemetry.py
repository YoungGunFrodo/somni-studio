"""Telemetry on a tiny random Qwen3 with real Somni experts, routed by Somni's own HFStudent._route.
Skipped without torch/transformers or a cached Qwen3 tokenizer."""
import importlib.util
import os
import unittest

from somni_studio.events import EventType

from .helpers import SOMNI_REPO, require_somni

HAVE_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "transformers"))


@unittest.skipUnless(HAVE_TORCH, "torch and transformers are not installed")
class TelemetryTest(unittest.TestCase):
    def setUp(self):
        require_somni(self)
        from somni_studio.somni.runtime import ensure_sleepai
        ensure_sleepai(SOMNI_REPO)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoTokenizer
        try:
            self.tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
        except Exception as exc:          # not cached on this machine
            self.skipTest(f"Qwen3 tokenizer not available offline: {exc}")

    def _tiny_model(self, layers=3):
        import torch
        from sleepai.models.experts import DEFAULT_TARGETS, add_expert, wrap
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(0)
        cfg = Qwen3Config(vocab_size=len(self.tok), hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256)
        model = Qwen3ForCausalLM(cfg).eval()
        ctx = {}
        model._expert_ctx = ctx
        wrapped = wrap(model, DEFAULT_TARGETS, ctx)
        add_expert(wrapped, rank=4, alpha=8.0, gate_bias=0.0)
        add_expert(wrapped, rank=4, alpha=8.0, gate_bias=0.0)
        ctx["thresholds"] = [None, 0.0]            # expert 0 always on, expert 1 routed (always passes)
        return model, ctx, len(DEFAULT_TARGETS)

    def test_route_activity_and_tokens(self):
        import torch
        from sleepai.models.hf import HFStudent

        from somni_studio.engine.telemetry import Telemetry
        model, ctx, per_layer = self._tiny_model(layers=3)
        events = []
        tel = Telemetry(model, self.tok, modules_per_layer=per_layer)
        tel.attach()
        tel.begin(lambda kind, **payload: events.append((kind, payload)))
        enc = self.tok(["Who gave their name to Normandy?"], return_tensors="pt")
        HFStudent("unused")._route(model, ctx, enc)            # Somni's real routing pass
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=5, min_new_tokens=5, do_sample=False,
                                 pad_token_id=self.tok.pad_token_id)
        ctx["route"] = None
        tel.end()
        tel.detach()

        routes = [p for k, p in events if k is EventType.ROUTE]
        self.assertEqual(len(routes), 1)
        experts = routes[0]["experts"]
        self.assertEqual([e["index"] for e in experts], [0, 1])
        self.assertEqual([e["on"] for e in experts], [True, True])
        self.assertIsNone(experts[0]["threshold"])
        self.assertEqual([x["node_id"] for x in experts[1]["layers"]], ["expert:1:0", "expert:1:1", "expert:1:2"])

        phases = [p["phase"] for k, p in events if k is EventType.ACTIVITY]
        self.assertEqual(phases[:2], ["route", "read"])
        # Generation frames are throttled to 30/s: they arrive as "write", or, if the answer ends
        # inside one interval, as the final "done" flush. Either way the writing must be reported.
        self.assertTrue(set(phases[2:]) & {"write", "done"}, phases)
        self.assertLessEqual(set(phases), {"route", "read", "write", "done"})
        frame = next(p for k, p in events if k is EventType.ACTIVITY)
        self.assertEqual(len(frame["layers"]), 3)
        self.assertTrue(all(x["value"] >= 0 for x in frame["layers"]))

        text = ""
        for k, p in events:
            if k is EventType.TOKEN:
                text = p["text"] if p.get("replace") else text + p["text"]
        expected = self.tok.decode(gen[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
        self.assertEqual(text, expected, "streamed tokens must equal what generate() produced")

    def test_hooks_do_not_change_the_output(self):
        import torch

        from somni_studio.engine.telemetry import Telemetry
        model, ctx, per_layer = self._tiny_model(layers=2)
        enc = self.tok(["hello"], return_tensors="pt")
        with torch.no_grad():
            plain = model(**enc).logits
        tel = Telemetry(model, self.tok, modules_per_layer=per_layer)
        tel.attach()
        tel.begin(lambda *a, **k: None)
        with torch.no_grad():
            hooked = model(**enc).logits
        tel.end()
        tel.detach()
        self.assertTrue(torch.equal(plain, hooked))
        self.assertFalse(any(layer._forward_hooks for layer in model.model.layers), "detach removes every hook")


if __name__ == "__main__":
    unittest.main()
