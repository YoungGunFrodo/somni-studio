"""Is this model a good brain for Somni on this computer? Speed, tool calling, and learning.

    python scripts/model_check.py Qwen/Qwen3-4B-Instruct-2507
    python scripts/model_check.py Qwen/Qwen3-4B-Instruct-2507

Measures, with the studio's own code paths:
  1. speed      tokens per second of an answer, and time to the first word (fp16 on GPUs without
                fast bf16, like the studio)
  2. tools      8 household requests against a fake smart home: did it call the right tool (or
                rightly no tool)?
  3. learning   how many of Somni's expert targets the model has, and one real training step of a
                Somni expert (forward + backward): does it run, how long, how much VRAM

Needs the Somni checkout (studio.json somni.repo) and a GPU environment with torch.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEVICES = {"light.kitchen": "off", "light.bedroom": "on", "switch.fan": "off"}
QUESTIONS = [                                              # (question, expected tool or None)
    ("When was Nikola Tesla born?", None),
    ("Is the fan on?", "home_states"),
    ("Is the bedroom light on?", "home_states"),
    ("Turn off the bedroom light.", "home_control"),
    ("Remember that my sister's name is Ana.", "memory_remember"),
    ("Write me a short poem about rain.", None),
    ("What time is it?", "clock_now"),
    ("Go to sleep now.", "sleep_start"),
]


class FakeHomeAssistant(BaseHTTPRequestHandler):
    def _send(self, body):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def do_GET(self):  # noqa: N802 - http.server API
        if self.path == "/api/states":
            self._send([{"entity_id": e, "state": s, "attributes": {"friendly_name": e.split(".")[1].title() + " "
                         + e.split(".")[0]}} for e, s in DEVICES.items()])
        elif self.path.startswith("/api/states/"):
            e = self.path.rsplit("/", 1)[1]
            self._send({"entity_id": e, "state": DEVICES.get(e, "unknown"), "attributes": {"friendly_name": e}})
        else:
            self._send({"message": "API running."})

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        DEVICES[body["entity_id"]] = {"turn_on": "on", "turn_off": "off"}.get(self.path.rsplit("/", 1)[1], "on")
        self._send([])

    def log_message(self, *args):
        pass


def check_speed(model, tok) -> dict:
    import torch
    messages = [{"role": "user", "content": "Tell me about the city of Paris in a few sentences."}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        model.generate(**enc, max_new_tokens=8, do_sample=False)            # warm-up
        torch.cuda.synchronize()
        t = time.perf_counter()
        model.generate(**enc, max_new_tokens=1, do_sample=False)
        torch.cuda.synchronize()
        first = time.perf_counter() - t
        t = time.perf_counter()
        out = model.generate(**enc, max_new_tokens=96, min_new_tokens=96, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t
    n = out.shape[1] - enc.input_ids.shape[1]
    return {"tokens_per_s": round(n / elapsed, 1), "first_token_s": round(first, 2)}


def check_tools(model, tok) -> dict:
    from somni_studio.engine.agent import LocalBackend, ToolAgent
    from somni_studio.tools.catalog import ToolContext, available_tools
    server = HTTPServer(("127.0.0.1", 0), FakeHomeAssistant)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        ctx = ToolContext(integration_config=lambda i: {"url": url, "token": "t"},
                          integration_ready=lambda i: i == "home_assistant",
                          allowed_domains=frozenset({"light", "switch"}),
                          remember=lambda text: "m1", request_sleep=lambda: None)
        tools = available_tools(ctx)
        results, right = [], 0
        for question, expected in QUESTIONS:
            calls = []
            ctx.question = question
            def record(kind, calls=calls, **p):
                if kind.value == "tool.call":
                    calls.append(p["name"])
            agent = ToolAgent(LocalBackend(model, tok, None, 160), ctx, record, 3)
            t = time.perf_counter()
            agent.run(question, [], tools)
            ok = (calls[:1] == [expected]) if expected else not calls
            right += ok
            results.append({"question": question, "expected": expected, "called": calls,
                            "ok": ok, "seconds": round(time.perf_counter() - t, 1)})
        return {"right": right, "of": len(QUESTIONS), "details": results, "devices_after": dict(DEVICES)}
    finally:
        server.shutdown()
        server.server_close()


def check_learning(model, tok) -> dict:
    import torch
    from sleepai.models.experts import DEFAULT_TARGETS, add_expert, wrap
    linear_names = {name.rsplit(".", 1)[-1] for name, m in model.named_modules() if m.__class__.__name__ == "Linear"}
    ctx: dict = {}
    wrapped = wrap(model, DEFAULT_TARGETS, ctx)
    params = add_expert(wrapped, rank=16, alpha=32.0, gate_bias=0.0)
    for p in params:
        p.data = p.data.float() if p.dtype != torch.float32 else p.data
    per_target = {t: sum(1 for name, _ in wrapped if name.endswith("." + t)) for t in DEFAULT_TARGETS}
    text = "The user's sister is called Ana. The assistant's name is Somni."
    batch = tok([text] * 2, return_tensors="pt").to(model.device)
    torch.cuda.reset_peak_memory_stats()
    model.train()
    t = time.perf_counter()
    try:
        out = model(**batch, labels=batch.input_ids)
        out.loss.backward()
        torch.cuda.synchronize()
        ok, error = True, ""
    except Exception as exc:                  # a model Somni cannot train is exactly what this finds
        ok, error = False, f"{type(exc).__name__}: {exc}"[:300]
    finally:
        model.eval()
    return {"trains": ok, "error": error, "step_s": round(time.perf_counter() - t, 2),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "expert_modules": len(wrapped), "per_target": per_target,
            "model_linear_names": sorted(linear_names)[:40]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model")
    parser.add_argument("--skip-learning", action="store_true")
    args = parser.parse_args()

    from somni_studio.settings import load_settings, settings_path
    from somni_studio.somni.runtime import ensure_sleepai
    ensure_sleepai(load_settings(settings_path()).somni.repo_path)
    import torch
    from sleepai.models import hf

    from somni_studio.engine.precision import resolve_dtype, somni_load_dtype
    dtype = resolve_dtype("auto")
    t = time.perf_counter()
    with somni_load_dtype(dtype):
        model, tok = hf._load(args.model, None)          # exactly how the studio loads it
    report = {"model": args.model, "dtype": str(dtype).replace("torch.", ""), "class": type(model).__name__,
              "load_s": round(time.perf_counter() - t, 1),
              "weights_gb": round(sum(p.numel() * p.element_size() for p in model.parameters()) / 2**30, 2)}
    print(json.dumps(report), flush=True)
    report["speed"] = check_speed(model, tok)
    print(json.dumps({"speed": report["speed"]}), flush=True)
    report["tools"] = check_tools(model, tok)
    print(json.dumps({"tools": {k: v for k, v in report["tools"].items() if k != "details"}}), flush=True)
    for d in report["tools"]["details"]:
        print(f"  {'ok ' if d['ok'] else 'MISS'} {d['seconds']:5.1f}s  {d['question']}  -> {d['called'] or '-'}")
    if not args.skip_learning:
        report["learning"] = check_learning(model, tok)
        print(json.dumps({"learning": report["learning"]}), flush=True)
    out = ROOT / ".cache" / f"model-check-{Path(args.model).name.replace('/', '_')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
