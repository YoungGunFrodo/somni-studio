// REST client. Every call resolves to parsed JSON or throws ApiError with the server's message.

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request(method, path, body) {
  const init = { method, headers: { Accept: "application/json" } };
  if (body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  let response;
  try {
    response = await fetch(path, init);
  } catch {
    throw new ApiError(0, "the studio server is not reachable");
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : data.detail;
    throw new ApiError(response.status, detail || `${method} ${path} failed (${response.status})`);
  }
  return data;
}

export const api = {
  graph: () => request("GET", "/api/graph"),
  node: (id) => request("GET", `/api/nodes/${encodeURIComponent(id)}`),
  file: (episodeId) => request("GET", `/api/files/${episodeId}`),
  openFile: (episodeId) => request("POST", `/api/files/${episodeId}/open`),
  settings: () => request("GET", "/api/settings"),
  saveSettings: (patch) => request("PUT", "/api/settings", patch),
  engine: () => request("GET", "/api/engine"),
  loadEngine: () => request("POST", "/api/engine/load"),
  unloadEngine: () => request("POST", "/api/engine/unload"),
  ask: (question) => request("POST", "/api/chat", { question }),
  conversation: (limit = 40) => request("GET", `/api/conversation?limit=${limit}`),

  model: () => request("GET", "/api/model"),
  serverModels: () => request("GET", "/api/model/server"),
  setModel: (model, relearn = false) => request("PUT", "/api/model", { model, relearn }),

  sleep: () => request("GET", "/api/sleep"),
  startSleep: () => request("POST", "/api/sleep"),
  wake: () => request("POST", "/api/sleep/wake"),
  stopSleep: () => request("POST", "/api/sleep/stop"),
  sleepReport: () => request("GET", "/api/sleep/report"),
  review: (checkpoint, decision) => request("POST", `/api/review/${encodeURIComponent(checkpoint)}/${decision}`),

  connections: () => request("GET", "/api/connections"),
  devices: () => request("GET", "/api/devices"),
  scanDevices: () => request("POST", "/api/devices/scan"),
  connectDevice: (target) => request("POST", "/api/devices/connect", { target }),
  controlDevice: (id, body) => request("POST", `/api/devices/${encodeURIComponent(id)}/control`, body),
  forgetDevice: (linkId) => request("DELETE", `/api/devices/links/${encodeURIComponent(linkId)}`),
  saveConnection: (id, values) => request("PUT", `/api/connections/${encodeURIComponent(id)}`, { values }),
  testConnection: (id) => request("POST", `/api/connections/${encodeURIComponent(id)}/test`),
  removeConnection: (id) => request("DELETE", `/api/connections/${encodeURIComponent(id)}`),
  tools: () => request("GET", "/api/tools"),
  discoverHomeAssistant: () => request("POST", "/api/connections/home_assistant/discover"),
  homeAssistantVm: () => request("GET", "/api/connections/home_assistant/vm"),
  homeAssistantVmControl: (action) => request("POST", `/api/connections/home_assistant/vm/${action}`),

  speech: () => request("GET", "/api/speech"),
  previewSpeech: (text, style) => request("POST", "/api/speech/preview", { text, ...style }),
  clipUrl: (clip) => `/api/speech/clips/${clip}.wav`,

  listenStatus: () => request("GET", "/api/listen"),
  gpu: () => request("GET", "/api/gpu"),
  gpuResume: (ignoreProgram = false) => request("POST", "/api/gpu/resume", { ignore_program: ignoreProgram }),
  async listen(wav, mode) {
    let response;
    try {
      response = await fetch(`/api/listen?mode=${encodeURIComponent(mode)}`, {
        method: "POST", headers: { "Content-Type": "audio/wav", Accept: "application/json" }, body: wav });
    } catch {
      throw new ApiError(0, "the studio server is not reachable");
    }
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new ApiError(response.status, data.detail || `listening failed (${response.status})`);
    return data;
  },
};
