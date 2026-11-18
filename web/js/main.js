// Somni Studio: wiring. Each module owns one concern; this file only connects them.

import * as THREE from "three";
import { $ } from "./core/dom.js";
import { Dreams } from "./live/dreams.js";
import { LiveLights } from "./live/liveLights.js";
import { Speaker } from "./live/speaker.js";
import { api } from "./net/api.js";
import { LiveSocket } from "./net/socket.js";
import { Effects } from "./scene/effects.js";
import { nodeColour } from "./scene/palette.js";
import { GraphView } from "./scene/graphView.js";
import { Labels } from "./scene/labels.js";
import { FLY_MS, Stage } from "./scene/stage.js";
import { Strands } from "./scene/strands.js";
import { GraphStore } from "./state/graphStore.js";
import { Chat } from "./ui/chat.js";
import { Drawer } from "./ui/drawer.js";
import { Hud } from "./ui/hud.js";
import { Inspector } from "./ui/inspector.js";
import { Search } from "./ui/search.js";
import { SettingsDialog } from "./ui/settings.js";
import { ConnectionsTab } from "./ui/tabs/connectionsTab.js";
import { ModelTab } from "./ui/tabs/modelTab.js";
import { SleepTab } from "./ui/tabs/sleepTab.js";
import { ToolsTab } from "./ui/tabs/toolsTab.js";
import { VoiceTab } from "./ui/tabs/voiceTab.js";
import { GpuBanner } from "./ui/gpuBanner.js";
import { Tooltip } from "./ui/tooltip.js";
import { VoiceInput } from "./ui/voiceInput.js";

const CLICK_SLOP_PX = 5;
const FLY_DISTANCE = { memory: 75, file: 85, tag: 110, expert: 95, module: 85, layer: 115, embed: 120, head: 120, hub: 140 };

// ---------------------------------------------------------------- scene
const host = $("#canvas-host");
const stage = new Stage(host);
const view = new GraphView(stage);
const effects = new Effects(stage);
const strands = new Strands(stage);
const labels = new Labels(stage, host);
const store = new GraphStore();
const socket = new LiveSocket();
const tooltip = new Tooltip(host);

// ---------------------------------------------------------------- selection
const selection = { id: null, hovered: null, search: null };

function neighbourhood(id) {
  const ids = new Set([id]);
  for (const link of store.links.values()) {
    if (link.source === id) ids.add(link.target);
    else if (link.target === id) ids.add(link.source);
  }
  return ids;
}

function fly(id) {
  const node = store.get(id);
  const pos = view.positionOf(id);
  if (!node || !pos) return;
  const distance = node.kind === "tag" && node.attrs?.root ? 180 : FLY_DISTANCE[node.kind] ?? 90;
  stage.flyTo(pos, distance);
}

function select(id, { fly: andFly = false } = {}) {
  if (!store.get(id)) return;
  selection.id = id;
  view.setSelected(id);
  view.setHighlight(neighbourhood(id));
  inspector.show(id);
  if (andFly) fly(id);
}

function deselect() {
  selection.id = null;
  view.setSelected(null);
  view.setHighlight(selection.search);
  inspector.hide();
}

// ---------------------------------------------------------------- UI
const hud = new Hud({
  onEngineToggle: async (state) => {
    try {
      if (state === "ready") await api.unloadEngine();
      else await api.loadEngine();
    } catch (err) {
      hud.toast(err.message, "error");
    }
  },
  onSleepToggle: async (asleep) => {
    try {
      if (asleep) await api.wake();
      else {
        await api.startSleep();
        drawer.open("sleep");
      }
    } catch (err) {
      hud.toast(err.message, "error");
    }
  },
});

async function saveSettings(patch) {
  return api.saveSettings(patch);
}

let voiceInput = null;            // created below, once the chat exists
const speaker = new Speaker({
  button: $("#voice-toggle"),
  onToggle: (on) => saveSettings({ voice: { enabled: on } })
    .then(() => speaker.setEnabled(on))
    .catch((err) => hud.toast(err.message, "error")),
  onPlaying: (on) => {
    if (voiceInput) voiceInput.assistantSpeaking = on;
  },
});

const drawer = new Drawer({ onChange: () => updateInsets() });
const refreshDrawer = () => drawer.refresh();
drawer.register("model", new ModelTab({ hud, saveSettings, refresh: refreshDrawer }));
drawer.register("sleep", new SleepTab({ hud, refresh: refreshDrawer }));
drawer.register("voice", new VoiceTab({ hud, saveSettings, speaker,
  onListenChanged: (status) => voiceInput.configure(status) }));
drawer.register("connections", new ConnectionsTab({ hud, refresh: refreshDrawer }));
drawer.register("tools", new ToolsTab({ refresh: refreshDrawer }));
const inspector = new Inspector({ store, hud, onSelect: select, onFocus: fly, onClose: deselect });
const chat = new Chat({
  socket,
  hud,
  toolLabel: (name) => store.get(`tool:${name}`)?.label ?? name.replace("_", " "),
  onChipHover: (id, on) => {
    if (on) view.pulse(id, 1.2);
    view.setHover(on ? id : selection.hovered);
  },
  onChipClick: (id) => select(id, { fly: true }),
});
const search = new Search({
  store,
  onHighlight: (ids) => {
    selection.search = ids;
    if (!selection.id) view.setHighlight(ids);
  },
  onPick: (id) => select(id, { fly: true }),
});
new SettingsDialog({ hud, onSaved: () => {} });     // the server pushes graph.snapshot after a save
new LiveLights({ view, effects, strands, store }).bind(socket);
const dreams = new Dreams({ view, effects, strands, store });
dreams.bind(socket);
voiceInput = new VoiceInput({ hud, speaker, send: (text) => chat.sendText(text) });

// ---------------------------------------------------------------- graph sync
let framed = false;

store.on("changed", ({ reason, added }) => {
  view.setGraph(store.nodes, store.links);
  strands.rebuild(store.nodes);
  labels.rebuild(store.nodes, store.meta);
  hud.setGraphMeta(store.meta);
  if (!framed && store.nodes.size) {
    stage.frameBox(view.bounds());
    framed = true;
  }
  if (reason === "diff") announceBirths(added);
  if (selection.id) {
    if (store.get(selection.id)) select(selection.id);
    else deselect();
  }
});

/** New nodes (a memory just made, an expert grown in sleep) form like stars, in their own colour.
 * If one appears off screen and the camera is not in the user's hands, the view eases out to show it. */
const WHITE = new THREE.Color(0xffffff);
const HANDS_OFF_S = 8;
function announceBirths(added) {
  const born = added.map((id) => store.get(id)).filter(Boolean);
  if (!born.length) return;
  const hidden = born.some((n) => !stage.inView(view.positionOf(n.id)));
  const reframe = hidden && stage.idleSeconds > HANDS_OFF_S && !selection.id;
  if (reframe) stage.frameBox(view.bounds());
  const form = () => {
    const colour = new THREE.Color();
    for (const node of born) {
      view.grow(node.id);
      nodeColour(node, colour);
      effects.birth(view.positionOf(node.id), node.size, colour.clone().lerp(WHITE, 0.25));
    }
  };
  if (reframe) {
    for (const node of born) view.grow(node.id);     // hidden until the camera arrives
    setTimeout(form, FLY_MS * 0.7);
  } else {
    form();
  }
}

let loading = null;
async function loadGraph() {
  if (loading) return loading;
  loading = api.graph()
    .then((snapshot) => store.loadSnapshot(snapshot))
    .catch((err) => hud.toast(`could not load the graph: ${err.message}`, "error"))
    .finally(() => (loading = null));
  return loading;
}

socket.on("connection", ({ connected }) => hud.setConnected(connected));
socket.on("hello", (e) => {
  hud.setEngine(e.engine);
  hud.setSleep(e.sleep?.state ?? "idle");
  speaker.setEnabled(e.speech?.enabled ?? true);
  if (e.listen) voiceInput.configure(e.listen);
  gpuBanner.show(e.gpu);
  dreams.resume(e.sleep);
  if (e.revision !== store.revision) loadGraph();
});
socket.on("graph.snapshot", () => loadGraph());
socket.on("resync", () => loadGraph());
socket.on("graph.diff", (diff) => {
  if (!store.applyDiff(diff)) loadGraph();
});
socket.on("engine.status", (status) => hud.setEngine(status));
const gpuBanner = new GpuBanner({ hud });
socket.on("gpu.status", (status) => gpuBanner.show(status));
socket.on("chat.start", (e) => speaker.begin(e.request_id));
socket.on("speech", (e) => speaker.handle(e, api.clipUrl));
socket.on("sleep.status", (e) => {
  hud.setSleep(e.state);
  if (e.state === "finished") hud.toast("Sleep finished: open Sleep to review what it learned", "good");
  if (e.state === "failed") hud.toast(e.summary || "the sleep did not finish", "error");
  drawer.dispatch("sleep.status", e);
});
for (const type of ["sleep.log", "sleep.phase", "connection"]) socket.on(type, (e) => drawer.dispatch(type, e));
socket.connect();
api.engine().then((s) => hud.setEngine(s)).catch(() => {});

// ---------------------------------------------------------------- pointer
const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();
const pointer = { x: 0, y: 0, down: null, pending: false };

function pickAtPointer() {
  pointer.pending = false;
  const box = host.getBoundingClientRect();
  ndc.set(((pointer.x - box.left) / box.width) * 2 - 1, -((pointer.y - box.top) / box.height) * 2 + 1);
  raycaster.setFromCamera(ndc, stage.camera);
  const id = view.pick(raycaster);
  if (id !== selection.hovered) {
    selection.hovered = id;
    view.setHover(id);
    host.classList.toggle("hovering", Boolean(id));
  }
  tooltip.show(id ? store.get(id) : null, pointer.x, pointer.y);
}

host.addEventListener("pointermove", (e) => {
  pointer.x = e.clientX;
  pointer.y = e.clientY;
  if (pointer.down) return;                       // no hover work while orbiting
  if (!pointer.pending) {
    pointer.pending = true;
    requestAnimationFrame(pickAtPointer);
  }
});
host.addEventListener("pointerdown", (e) => (pointer.down = { x: e.clientX, y: e.clientY }));
host.addEventListener("pointerup", (e) => {
  const down = pointer.down;
  pointer.down = null;
  if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > CLICK_SLOP_PX || e.button !== 0) return;
  pointer.x = e.clientX;
  pointer.y = e.clientY;
  pickAtPointer();
  if (selection.hovered) select(selection.hovered);
  else if (selection.id) deselect();
});
host.addEventListener("dblclick", () => selection.hovered && select(selection.hovered, { fly: true }));
host.addEventListener("pointerleave", () => {
  tooltip.hide();
  selection.hovered = null;
  view.setHover(null);
});

// ---------------------------------------------------------------- panels vs. scene
const PANEL_GAP_PX = 12;
const NARROW_PX = 900;

function updateInsets() {
  if (window.innerWidth <= NARROW_PX) {
    stage.setInsets(0, 0);
    return;
  }
  const hostBox = host.getBoundingClientRect();
  const chat = $("#chat");
  const leftPanel = drawer.isOpen ? $("#drawer") : chat.classList.contains("collapsed") ? $(".toolbox") : chat;
  const left = leftPanel.getBoundingClientRect().right - hostBox.left;
  const right = inspector.open ? hostBox.right - $("#inspector").getBoundingClientRect().left : 0;
  stage.setInsets(left + PANEL_GAP_PX, right ? right + PANEL_GAP_PX : 0);
}
new ResizeObserver(updateInsets).observe($("#chat"));
new MutationObserver(updateInsets).observe($("#inspector"), { attributes: true, attributeFilter: ["hidden"] });
window.addEventListener("resize", updateInsets);
updateInsets();

// ---------------------------------------------------------------- toolbox & keys
$("#toggle-labels").addEventListener("change", (e) => (labels.visible = e.target.checked));
$("#toggle-links").addEventListener("change", (e) => view.setLinksVisible(e.target.checked));
$("#toggle-bloom").addEventListener("change", (e) => (stage.bloomEnabled = e.target.checked));
$("#toggle-spin").addEventListener("change", (e) => (stage.autoRotate = e.target.checked));
$("#reset-view").addEventListener("click", () => stage.frameBox(view.bounds()));

document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape" || document.querySelector("dialog[open]")) return;
  if (document.activeElement === $("#search")) return;
  if (selection.id) deselect();
  else if (drawer.isOpen) drawer.close();
  else search.clear();
});
