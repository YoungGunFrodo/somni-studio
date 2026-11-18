// Model tab: which model the assistant runs, chosen from what is installed on this computer, or an
// OpenAI-compatible server (LM Studio, llama-server) instead.

import { cloneTemplate, fill, fmt } from "../../core/dom.js";
import { api } from "../../net/api.js";
import { busy, button, card, field, input, para, row, select, statusLine } from "./widgets.js";

const DTYPES = [
  { value: "auto", label: "auto (fastest for this GPU)" },
  { value: "float16", label: "float16" },
  { value: "bfloat16", label: "bfloat16" },
  { value: "float32", label: "float32 (slow, most memory)" },
];

export class ModelTab {
  title = "Model";

  constructor({ hud, saveSettings, refresh }) {
    this.hud = hud;
    this.saveSettings = saveSettings;
    this.refresh = refresh;
  }

  async render(body) {
    const data = await api.model();
    const provider = data.provider;
    const usingServer = provider.kind === "openai";

    // ---- what runs now
    const now = card("Running", usingServer ? "server" : "on this computer");
    const name = document.createElement("div");
    name.className = "big";
    name.textContent = usingServer ? provider.model || "(server default model)" : data.current;
    now.body.appendChild(name);
    const engine = data.engine;
    now.body.appendChild(statusLine(engine.state === "ready" || engine.state === "busy" ? "ready" : engine.state,
      `${engine.state}${engine.dtype ? ` · ${engine.dtype}` : ""}${engine.detail ? ` · ${engine.detail}` : ""}`));
    if (!usingServer && !data.installed) {
      now.body.appendChild(para("Not downloaded yet: it will be downloaded from Hugging Face the first time it loads."));
    }
    body.appendChild(now.root);

    // ---- local models
    const local = card("Models on this computer", `${data.local.length} found`);
    if (data.relearn_pending) {
      local.body.appendChild(para(`The next sleep relearns every memory for ${data.relearn_pending.to}.`));
    } else if (data.locked) {
      local.body.appendChild(para(data.locked));
    }
    else if (data.read_only) local.body.appendChild(para("This Somni home is read-only, so its model cannot be changed here."));
    const list = document.createElement("div");
    list.className = "choice-list";
    const fixed = Boolean(data.read_only);
    for (const m of data.local) {
      const where = m.load === m.id ? "" : " · local folder";
      const rowEl = fill(cloneTemplate("tpl-model-row"), {
        title: m.id,
        sub: `${m.size_gb} GB · ${m.layers ?? "?"} layers · ${m.architecture.replace("ForCausalLM", "")}${where}`,
        tag: m.load === data.current ? "in use" : "",
      });
      const radio = rowEl.querySelector("input");
      radio.value = m.load;
      radio.checked = m.load === data.current;
      rowEl.title = m.load;
      radio.disabled = fixed;
      if (fixed) rowEl.classList.add("disabled");
      list.appendChild(rowEl);
    }
    if (!data.local.length) local.body.appendChild(para("No language models found in the Hugging Face cache."));
    const dirs = input("text", provider.model_dirs, { placeholder: "e.g. D:/models;E:/hf (folders separated by ;)" });
    const scan = button("Look there too");
    scan.addEventListener("click", () => busy(scan, this.hud, async () => {
      await this.saveSettings({ provider: { model_dirs: dirs.value.trim() } });
      await this.refresh();
    }, "Looking…"));
    local.body.appendChild(list);
    const custom = input("text", "", { placeholder: "or a Hugging Face id / local folder, e.g. Qwen/Qwen3-4B" });
    custom.disabled = fixed;
    const use = button(data.locked ? "Switch and relearn" : "Use this model", { primary: true });
    use.disabled = fixed;
    use.addEventListener("click", () => busy(use, this.hud, async () => {
      const chosen = custom.value.trim() || list.querySelector("input:checked")?.value;
      if (!chosen) throw new Error("choose a model");
      if (usingServer) await this.saveSettings({ provider: { kind: "somni" } });
      const res = await api.setModel(chosen, Boolean(data.locked));
      if (res.relearn) this.hud.toast("Old experts archived. The next sleep teaches the new model all your memories.");
      this.hud.toast(`model set to ${res.current}; it loads with the next question`, "good");
      await this.refresh();
    }, "Saving…"));
    local.body.appendChild(field({ label: "Other model", control: custom,
      help: "Memories always stay. Experts are made for one model, so after a switch the next sleep relearns." }));
    local.body.appendChild(row(use));
    local.body.appendChild(field({ label: "More model folders", control: dirs,
      help: "A Hugging Face cache (like a custom HF_HOME) or a folder of model folders." }));
    local.body.appendChild(row(scan));
    body.appendChild(local.root);

    // ---- server
    const server = card("Or use a server", "LM Studio, llama-server, any OpenAI-compatible API");
    const url = input("url", provider.url);
    const modelSelect = select([{ value: provider.model, label: provider.model || "(server default)" }], provider.model);
    const status = statusLine("checking", "checking server…");
    server.body.append(field({ label: "Server address", control: url }), field({ label: "Model", control: modelSelect }),
      status, para("With a server, memory recall and learning still work; the brain view stays dark because a server does not show its layers."));
    const useServer = button(usingServer ? "Save server" : "Use server", { primary: !usingServer });
    const useLocal = usingServer ? button("Back to the local model") : null;
    useServer.addEventListener("click", () => busy(useServer, this.hud, async () => {
      await this.saveSettings({ provider: { kind: "openai", url: url.value.trim(), model: modelSelect.value } });
      this.hud.toast("using the server", "good");
      await this.refresh();
    }));
    useLocal?.addEventListener("click", () => busy(useLocal, this.hud, async () => {
      await this.saveSettings({ provider: { kind: "somni" } });
      await this.refresh();
    }));
    server.body.appendChild(row(useServer, useLocal));
    body.appendChild(server.root);
    api.serverModels().then((res) => {
      status.dataset.state = res.reachable ? "connected" : "error";
      status.querySelector(".status-text").textContent = res.reachable
        ? `${res.models.length} models available` : `not reachable (${res.error})`;
      for (const id of res.models) {
        if (id === provider.model) continue;
        const opt = document.createElement("option");
        opt.value = opt.textContent = id;
        modelSelect.appendChild(opt);
      }
    }).catch(() => {});

    // ---- advanced
    const adv = card("Advanced");
    const dtype = select(DTYPES, provider.dtype);
    const maxTokens = input("number", String(provider.max_new_tokens), { min: "16", max: "4096", step: "16" });
    adv.body.append(
      field({ label: "Precision", control: dtype,
        help: "auto uses float16 on GPUs without fast bfloat16 (e.g. Radeon RX 6000): several times faster there." }),
      field({ label: "Longest answer (tokens)", control: maxTokens }));
    const save = button("Save");
    save.addEventListener("click", () => busy(save, this.hud, async () => {
      const n = Number.parseInt(maxTokens.value, 10);
      if (!Number.isFinite(n)) throw new Error("the answer length must be a number");
      await this.saveSettings({ provider: { dtype: dtype.value, max_new_tokens: n } });
      this.hud.toast(`saved · ${fmt.int(n)} tokens · ${dtype.value}`, "good");
    }));
    adv.body.appendChild(row(save));
    body.appendChild(adv.root);
  }
}
