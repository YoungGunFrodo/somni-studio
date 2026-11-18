"""Forward hooks that turn one generation into UI events, without changing what the model computes.

Three kinds of hook, all read-only:

  * every transformer block: how much the block changed the hidden state at the newest position,
    ||h_out - h_in|| / ||h_in||  ->  per-layer `activity` (the brain tower lights up per token)
  * the decoder as a whole: when Somni's router has just scored the prompt (its stacked experts
    leave per-module scores in the model's expert context), capture them  ->  `route`
    (each expert's score at every layer, its threshold, and whether it switched on)
  * the output head: the greedy next token  ->  `token` (Somni decodes greedily, so the argmax
    is exactly the token generate() emits)

Per-layer numbers stay on the GPU until the output head fires, then are read in one transfer per
token, so the hooks add one small sync per token instead of one per layer.
"""
from __future__ import annotations

import time
from typing import Any

from ..events import EventType
from ..graph.brain_graph import expert_id, layer_id
from .types import Emitter

ACTIVITY_INTERVAL = 1 / 30          # at most 30 activity frames a second reach the browser


def _decoder_layers(model):
    decoder = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    layers = getattr(decoder, "layers", None)
    if layers is None:
        raise RuntimeError("cannot find the transformer blocks of this model")
    return decoder, list(layers)


class Telemetry:
    def __init__(self, model, tokenizer, modules_per_layer: int) -> None:
        self.model, self.tok = model, tokenizer
        self.modules_per_layer = max(1, modules_per_layer)
        self.decoder, self.layers = _decoder_layers(model)
        self.ectx: dict | None = getattr(model, "_expert_ctx", None)
        self._handles: list = []
        self._emit: Emitter | None = None
        self._acts: list = [None] * len(self.layers)
        self._peak: list[float] = [0.0] * len(self.layers)
        self._last_sent = 0.0
        self._ids: list[int] = []
        self._text = ""
        self._passes = 0
        self.mute_tokens = False
        self._hold = None

    # ------------------------------------------------------------ lifecycle
    def attach(self) -> None:
        if self._handles:
            return
        for i, block in enumerate(self.layers):
            self._handles.append(block.register_forward_hook(self._block_hook(i), with_kwargs=True))
        self._handles.append(self.decoder.register_forward_hook(self._decoder_hook))
        head = self.model.get_output_embeddings()
        if head is not None:
            self._handles.append(head.register_forward_hook(self._head_hook))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def begin(self, emit: Emitter) -> None:
        self._emit = emit
        self._acts = [None] * len(self.layers)
        self._peak = [0.0] * len(self.layers)
        self._ids, self._text, self._passes, self._last_sent = [], "", 0, 0.0
        self.mute_tokens = False
        self._hold = None

    def restart_text(self, hold=None) -> None:
        """A new reply starts (after tool calls). While `hold(raw_text)` is true nothing is sent:
        the reply might still turn out to be another tool call, which is not shown as text."""
        self._ids, self._text, self._hold = [], "", hold

    def end(self) -> None:
        self._flush_activity(force=True, phase="done")
        self._emit = None

    # ------------------------------------------------------------ hooks
    def _block_hook(self, i: int):
        def hook(_module, args, kwargs, output):
            if self._emit is None:
                return
            h_in = args[0] if args else kwargs.get("hidden_states")
            h_out = output[0] if isinstance(output, tuple) else output
            if h_in is None or not hasattr(h_out, "shape"):
                return
            a, b = h_in[0, -1].float(), h_out[0, -1].float()
            self._acts[i] = (b - a).norm() / a.norm().clamp_min(1e-6)
        return hook

    def _decoder_hook(self, _module, _args, _output) -> None:
        if self._emit is None:
            return
        scores = self.ectx.get("scores") if self.ectx is not None else None
        if isinstance(scores, dict) and scores:          # Somni's routing pass over the prompt
            self._emit_route(scores)
            self._flush_activity(force=True, phase="route")
            return
        self._passes += 1
        if self._passes == 1:                            # prefill: the model reads the prompt
            self._flush_activity(force=True, phase="read")

    def _head_hook(self, _module, _args, logits) -> None:
        if self._emit is None or not hasattr(logits, "shape"):
            return
        self._flush_activity(force=False, phase="write")
        if self.mute_tokens:                 # the model is deciding on a tool: not part of the answer
            return
        self._ids.append(int(logits[0, -1].argmax()))
        if self._hold is not None:
            if self._hold(self.tok.decode(self._ids, skip_special_tokens=False)):
                return
            self._hold = None                # decided: it is a reply, stream it from the start
        text = self.tok.decode(self._ids, skip_special_tokens=True)
        if len(text) > len(self._text) and text.startswith(self._text):
            self._emit(EventType.TOKEN, text=text[len(self._text):])
            self._text = text
        elif text != self._text:            # a merge changed earlier characters: resend all
            self._emit(EventType.TOKEN, text=text, replace=True)
            self._text = text

    # ------------------------------------------------------------ events
    def _flush_activity(self, force: bool, phase: str) -> None:
        ready = [(i, a) for i, a in enumerate(self._acts) if a is not None]
        if ready:
            import torch
            values = torch.stack([a for _, a in ready]).tolist()
            for (i, _), v in zip(ready, values, strict=True):
                self._peak[i] = max(self._peak[i], float(v))
            self._acts = [None] * len(self.layers)
        now = time.monotonic()
        if not force and now - self._last_sent < ACTIVITY_INTERVAL:
            return
        if self._emit is not None and any(self._peak):
            self._emit(EventType.ACTIVITY, phase=phase,
                       layers=[{"node_id": layer_id(i), "value": round(v, 4)} for i, v in enumerate(self._peak)])
        self._peak = [0.0] * len(self.layers)
        self._last_sent = now

    def _emit_route(self, scores: dict[int, list]) -> None:
        import torch
        thresholds = list((self.ectx or {}).get("thresholds") or [])
        experts: list[dict[str, Any]] = []
        for k in sorted(scores):
            calls = scores[k]
            if not calls:
                continue
            per_call = torch.stack(calls)[:, 0].float()          # one score per wrapped linear layer
            overall = float(per_call.mean())
            m = self.modules_per_layer
            n_layers = len(per_call) // m if len(per_call) % m == 0 else 0
            per_layer = per_call.view(n_layers, m).mean(1).tolist() if n_layers else []
            threshold = thresholds[k] if k < len(thresholds) else None
            experts.append({
                "index": k,
                "score": round(overall, 4),
                "threshold": None if threshold is None else round(float(threshold), 4),
                "on": threshold is None or overall >= float(threshold),
                "layers": [{"node_id": expert_id(k, i), "value": round(v, 4)} for i, v in enumerate(per_layer)],
            })
        if experts and self._emit is not None:
            self._emit(EventType.ROUTE, experts=experts)
