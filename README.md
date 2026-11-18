# Somni Studio

A home for a local AI assistant that **learns while it sleeps**, with a live 3D view of its brain and memory.

Somni Studio is the interface for [Somni](https://github.com/YoungGunFrodo/somni) (Sleep-Optimized Memory for
Neural Intelligence): a framework that gives a local language model a human-like memory process. During the
day it remembers conversations; while it sleeps, it turns them into memories and trains a new "expert" on
them, without overwriting anything it already knew. Somni Studio lets you talk to it, watch it think, put it
to sleep, decide what it keeps, and connect it to your home.

## What you see

- **The brain** (left): the model as a tower of transformer blocks. Every night of sleep adds an **expert**,
  drawn as a strand winding around the tower. While the model answers, each block glows with how much it is
  working, and the experts that switched on for this question light up and stream.
- **The memory** (right): every memory the assistant has formed, clustered by subject tags (broad to
  specific), with the full source text (memory files) beside the memories made from it. Hover to see a name,
  click to open it.
- **Recall** (the bridge): when you ask something, the memories it recalls glow and travel through the recall
  hub into the model's input. The memory the answer cites glows gold.
- **Tools** (top of the brain): when it uses a tool, a beam goes from the output to that tool and back.
- **Sleep**: memories shimmer while new ones form; the next expert grows up the tower as a violet strand
  with the real training steps; while it dreams, the memories it quizzes itself on beam into the brain
  (amber when it got one wrong and practises it); a scan sweeps the tower while the tests run.
- **Live**: when a sleep adds memories or an expert, they appear in place. No reload.

## What you can do

| | |
| --- | --- |
| **Talk** | Type, click the mic and speak, or go hands-free: "Somni, turn on the kitchen light." Answers are read aloud as they are written. |
| **Act** | It uses tools when you ask: the time, the weather, your devices, the web, your email, "remember that…", "go to sleep". |
| **Look it up** | Like any assistant: when it does not know something (and it is not about you), it searches instead of saying "I'm not sure". Weather comes from Open-Meteo for wherever you said you live, or any place you name. |
| **Devices** | Ask "find my devices" and it scans your network; "connect the kitchen display" and it is done. Lights and plugs (WiZ, LIFX, TP-Link Kasa, Philips Hue, Yeelight, Govee, Shelly), speakers and displays (Google Nest, Chromecast: volume, pause, and Somni speaking on them) and Roku. Directly on your network: no hub, no account, no cloud. |
| **Learn** | It remembers what you tell it about yourself ("you can call me sir") the moment you say it, keeps what it looked up for you, and recalls facts about you in every answer. A new memory forms in the graph like a star. |
| **Sleep** | Press *Sleep now* (or tell it to) to learn from recent conversations. Watch it happen in the brain: memories form, a new expert grows up the tower, and it dreams, quizzing itself on what it learned. |
| **Review** | Keep or reject what a night learned (Somni's morning review), with the test results. |
| **Voice** | Choose from 27 English voices, set the speed, preview. |
| **Connect** | Devices (above), Home Assistant (optional, for Zigbee, Z-Wave, thermostats and cloud-only brands; found on your network automatically), Google Home (through Home Assistant), web lookup and weather (Wikipedia and weather with no setup; for the whole web, a free Tavily or Brave key, or your own SearXNG), any API you have a key for (Custom API: say what it is for and Somni uses it; the key never reaches the model) and email (read-only). Each card shows how to set it up and what you can then say. |
| **Model** | Pick any model on your computer; switching later keeps every memory and the next sleep relearns them. `scripts/model_check.py` tests a model's speed, tool calling and learning before you switch. |
| **Gaming** | When a game (or anything else) needs the graphics card, Somni steps aside: it unloads its model and pauses sleep, then comes back when the GPU is free. |
| **Tools** | See everything the assistant can do and what each needs. |

Everything runs on your own computer. Conversations, memories, credentials and models stay in `data/`.

> **Status:** a working prototype. The assistant recalls, answers, learns, listens, speaks and uses tools.

## Set up

You need Python 3.10+ and a GPU that PyTorch supports (NVIDIA, or AMD with ROCm); the toy setup runs on CPU.

1. **Get Somni** next to this folder and set it up as its README describes (it installs PyTorch for your GPU):

   ```
   git clone https://github.com/YoungGunFrodo/somni ../somni
   ```

2. **Install the studio** into the same Python environment:

   ```
   pip install -e .[voice,listen]
   python scripts/vendor.py        # three.js for the 3D view (hash-checked, once)
   python scripts/fetch_voice.py   # the Kokoro voice model, about 350 MB (hash-checked, once)
   python scripts/fetch_listen.py  # the Whisper base.en model, about 145 MB (pinned revision, once)
   ```

3. **Run it**:

   ```
   python -m somni_studio --open
   ```

   On Windows, `.\run.ps1` finds Somni's environment and does steps 2-3 for you.

The first start creates a new, empty assistant in `data/home`. Settings live in `studio.json`
(copy `studio.example.json`; most things are also in the UI). If your Somni checkout is somewhere else, set
`somni.repo`. If you keep models outside the default Hugging Face cache, add the folder in the Model tab.

### Good to know

- **Speed.** On GPUs without fast bfloat16 (for example Radeon RX 6000), the studio runs the model in
  float16, measured 4.7× faster with identical answers (`provider.dtype`: `auto`).
- **The first sleep takes longer**: Somni first records how the base model answers a few hundred practice
  questions, so later sleeps can make sure it never gets worse at them.
- **Sleep uses the whole GPU**, so chat pauses until it wakes. The model is unloaded first.
- **Speech runs on the CPU** (4 threads each for speaking and listening), so it never slows the model down.
  Listening runs in its own process: Whisper's runtime and PyTorch's cannot share one.
- **Tool safety**: the assistant only switches device types listed in `tools.allowed_domains` (no locks,
  alarms or garage doors by default); it changes a device only when you ask for an action ("is the fan
  on?" never switches it); it saves to memory or goes to sleep only when your own words ask for it; web
  pages are read only from public addresses, never your own network; email access is read-only.
- **Devices stay on your network.** Somni talks to them on your LAN only and never contacts their cloud.
  Some need one switch first: Yeelight and Govee have a "LAN Control" setting in their app, a Hue bridge
  needs its button pressed once. Tapo and newer Kasa firmware need a TP-Link login, and Matter devices
  need commissioning: neither is supported yet. If the router gives a device a new address, Somni finds
  it again. A device you mention is read before answering, so "what volume is the kitchen display at?"
  is never a guess.
- **Music** is internet radio from Radio Browser (radio-browser.info: free, open, no account): ask for a genre,
  a mood or a station ("play some jazz in the living room", "put on BBC Radio 2") and a Nest Hub or Chromecast
  streams it. Only the words you asked for go to the directory. Spotify and YouTube Music need their own apps.
- **Speaking on a Nest Hub**: Somni's voice is served to that one device, from a one-file server at a
  random address that lives for two minutes.
- **Google Home** has no local API. Nest Hubs, Nest speakers and Chromecasts connect directly; lights that
  only work through the Google Home app need Home Assistant (with its Google Assistant SDK integration).
- **Searching the whole web** needs a search service that allows programs: Tavily (made for AI assistants,
  free plan, app.tavily.com), Brave Search (free plan, api-dashboard.search.brave.com) or your own SearXNG. Without one, Somni searches Wikipedia, which
  covers facts but not news, prices or opening hours. Pages from the top results are read too, so answers
  come from the page, not a two-line snippet. Only the search words go to the search service.
- **Working things out.** When it needs something it does not know (where you are, a name, a fact), the
  assistant looks in its memories, then asks the computer (`computer_info`: its town from Windows' location
  service, time zone, date), then searches the web, and asks you only when none of those can tell it. If it
  gathers what a tool needed and then stops short, Somni sends it back once to finish the job.
- **Knowing itself.** Every answer gets a live note of what Somni can do right now and what is not connected
  (built from the real tool and connection state, never stored), and a `system_status` tool reports its
  model, memories, learned experts, voice, listening, sleep and graphics card. Connecting or disconnecting
  anything (from chat or the Connect tab) becomes a memory, and so do facts its tools discover ("the user's
  home network has: ..."), so it learns its own setup in sleep too.
- **Memory.** One conversation is one episode, so tonight's sleep reads "what's my name?" next to "you can
  call me sir". Facts about you are saved the moment you say them (a newer one replaces an older one on the
  same topic: "call me captain" replaces "sir") and are recalled in every answer. What it looks up on the
  web is kept as *inferred*: used in answers, but not trained into the model until confirmed.

## How it is built

```
somni_studio/
  settings.py        typed, validated settings (a typo is an error, never a silent default)
  events.py          thread-safe event bus to every browser; a slow browser never slows the model
  somni/             a Somni home: read-only store access, the blank-home bootstrap, models on disk, sleep runner
  graph/             graph model and diffs, deterministic layout, brain and memory builders, live watcher
  engine/            one worker thread, providers, recall instrumentation, telemetry, precision, the tool agent
  speech/            Kokoro text-to-speech, Whisper speech-to-text (worker process), the wake word
  tools/             what the assistant can do: schemas, argument checks, guards
  devices/           direct device control: LAN discovery (SSDP, mDNS), drivers for lights and plugs, Cast, Roku
  integrations/      Home Assistant, Google Home, web lookup, email, credential store, tool catalogue
  api/               FastAPI: routes/ for REST, /ws for live events, the static UI
web/
  index.html         layout and every <template> the UI clones
  css/app.css        design tokens and components
  js/                ES modules: core/, net/, state/, scene/ (three.js), live/, ui/ (+ ui/tabs/), main.js
```

Design notes:

- **Chat goes through Somni's own recall** (its Orchestrator), so the studio answers exactly as Somni's wake
  does, and every conversation becomes an episode the next sleep learns from (`somni.read_only` turns that off
  for browsing other homes).
- **Positions are computed on the server** from hashes of node ids: the same memories always give the same
  picture, a new memory appears next to its subject without moving anything else, and layout cost grows
  linearly.
- **Rendering is instanced**: one draw call per kind of node, so tens of thousands of memories stay smooth.
- **Telemetry never changes the model's output** (tested) and costs about 0.5% of generation time (measured).
- **Speech starts while the answer is still being written**: sentences are voiced one by one as they finish.
- **Security**: nothing loads from a CDN at runtime; strict Content-Security-Policy; the WebSocket refuses
  other origins; credentials are stored locally, never sent back to the browser, and can only be changed from
  this computer.

## Test

```
python -m unittest discover -s tests -t .   # 210 tests; model tests use a tiny random Qwen3
python scripts/model_check.py <model>        # speed, tool calling (8 requests) and learning, for any model
python scripts/ui_smoke.py                   # drives the running UI in headless Edge/Chrome, saves screenshots
python scripts/ui_smoke.py --chat "Hello!"   # also loads the model, asks, and checks speech
python scripts/ui_smoke.py --chat "you can call me sir" --burst   # screenshots a new memory forming
python scripts/ui_smoke.py --voice-file say.wav   # hands-free through a fake microphone (48 kHz WAV)
```

## Roadmap

- More connections (calendar, sending email with confirmation, cameras).
- More device drivers (Tapo, Matter, Sonos); drivers as plugins, so a new brand is one small file.
- Remote access (Tailscale, Cloudflare Tunnel) and a phone-sized layout.

## License

MIT (see `LICENSE`). three.js is MIT; Kokoro and kokoro-onnx are Apache-2.0; Whisper and faster-whisper are MIT.
