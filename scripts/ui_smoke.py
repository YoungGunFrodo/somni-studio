"""UI smoke test: drives the running studio in headless Edge/Chrome over the DevTools protocol,
clicks through it like a person would, saves screenshots, and fails on any page error.

    python scripts/ui_smoke.py                 # browse: search, select, open a file, hover, settings
    python scripts/ui_smoke.py --chat "Who gave their name to Normandy?" --chat "..."   # also load the model and ask

Screenshots go to .cache/screens/. The studio must already be running (python -m somni_studio).
Needs the `websockets` package (installed with the studio).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parents[1]
SCREENS = ROOT / ".cache" / "screens"
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "msedge", "google-chrome", "chromium", "chrome",
]
DEBUG_PORT = 9333
VIEW = (1600, 900)

# Page probes (JavaScript expressions).
INSPECTOR_READY = ("!document.querySelector('#inspector').hidden"
                   " && !document.querySelector('#inspector-body').innerText.includes('loading')")
CLICK_FILE_LINK = """(() => {
    const b = [...document.querySelectorAll('#inspector-body .link-row')]
        .find(b => b.innerText.includes('open memory file'));
    if (!b) return null;
    b.click();
    return b.innerText;
})()"""
TOOLTIP_TEXT = "document.querySelector('#tooltip').hidden ? '' : document.querySelector('#tooltip').innerText"
SETTINGS_OPEN = "document.querySelector('#settings').open && document.querySelectorAll('[data-path]').length"
ENGINE_SETTLED = "['ready', 'error'].includes(document.querySelector('#engine-pill').dataset.state)"
ENGINE_BUSY = "document.querySelector('#engine-pill').dataset.state === 'busy'"
ANSWERS = "document.querySelectorAll('.msg-meta:not(:empty)').length"
MEMORY_COUNT = "Number(document.querySelector('[data-stat=memories]').textContent.replace(/[^0-9]/g, '')) || 0"
DRAWER_READY = ("!document.querySelector('#drawer').hidden && document.querySelector('#drawer-body').children.length"
                " && !document.querySelector('#drawer-body').innerText.includes('loading')")
CARD_TITLES = "[...document.querySelectorAll('#drawer-body .card h3')].map(h => h.textContent).join(', ')"
CLIPS_FETCHED = ("performance.getEntriesByType('resource')"
                 ".filter(e => e.name.includes('/api/speech/clips/')).length")


def find_browser() -> str:
    for b in BROWSERS:
        if Path(b).exists() or shutil.which(b):
            return b
    raise SystemExit("no Edge or Chrome found")


class Page:
    def __init__(self, ws):
        self.ws, self.n, self.errors, self.log = ws, 0, [], []

    def _event(self, msg: dict) -> None:
        method = msg.get("method")
        if method == "Runtime.exceptionThrown":
            d = msg["params"]["exceptionDetails"]
            where = f" ({d.get('url', '').rsplit('/', 2)[-1]}:{d.get('lineNumber', 0) + 1})" if d.get("url") else ""
            self.errors.append((d.get("exception", {}).get("description") or d.get("text")) + where)
        elif method == "Runtime.consoleAPICalled":
            text = " ".join(str(a.get("value", a.get("description", ""))) for a in msg["params"]["args"])
            (self.errors if msg["params"]["type"] == "error" else self.log).append(text)

    async def call(self, method: str, **params):
        self.n += 1
        my = self.n
        await self.ws.send(json.dumps({"id": my, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if "method" in msg:
                self._event(msg)
            elif msg.get("id") == my:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    async def js(self, expr: str):
        r = await self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"].get("exception", {}).get("description", "js failed"))
        return r.get("result", {}).get("value")

    async def wait(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:          # keep reading so events are collected
            await self.js("0")
            await asyncio.sleep(0.15)

    async def wait_for(self, expr: str, timeout: float, what: str):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            value = await self.js(expr)
            if value:
                return value
            await asyncio.sleep(0.25)
        raise TimeoutError(f"timed out waiting for {what}")

    async def shot(self, name: str) -> None:
        data = (await self.call("Page.captureScreenshot", format="png"))["data"]
        SCREENS.mkdir(parents=True, exist_ok=True)
        (SCREENS / name).write_bytes(base64.b64decode(data))
        print(f"  screenshot {SCREENS / name}")

    async def click(self, x: float, y: float) -> None:
        for kind in ("mouseMoved", "mousePressed", "mouseReleased"):
            await self.call("Input.dispatchMouseEvent", type=kind, x=x, y=y, button="left", clickCount=1)

    async def hover(self, x: float, y: float) -> None:
        await self.call("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y)

    async def type(self, text: str) -> None:
        await self.call("Input.insertText", text=text)

    async def key(self, key: str, code: int) -> None:
        for kind in ("keyDown", "keyUp"):
            await self.call("Input.dispatchKeyEvent", type=kind, key=key, code=key, windowsVirtualKeyCode=code)


# ---------------------------------------------------------------- scenarios
async def browse(p: Page, term: str) -> None:
    print("browse")
    await p.shot("01-overview.png")
    memories = await p.js(MEMORY_COUNT)
    if memories:
        await memory_walk(p, term)
    else:
        print("  the memory store is empty: skipping search and memory files")
    await hover_somewhere(p)
    await p.js("document.querySelector('#open-settings').click()")
    await p.wait_for(SETTINGS_OPEN, 5, "settings")
    await p.shot("05-settings.png")
    await p.js("document.querySelector('#settings').close()")


async def memory_walk(p: Page, term: str) -> None:
    await p.js("document.querySelector('#search').focus()")
    await p.type(term)
    await p.wait(0.5)
    results = await p.js("document.querySelectorAll('#search-results li').length")
    assert results, f"search found nothing for {term!r}"
    await p.key("Enter", 13)
    await p.wait_for(INSPECTOR_READY, 10, "the inspector")
    await p.wait(1.4)
    await p.shot("02-memory-selected.png")
    title = await p.js("document.querySelector('#inspector-title').textContent")
    print(f"  selected memory: {title}")
    opened = await p.js(CLICK_FILE_LINK)
    if not opened:
        print("  (this memory has no source file to open)")
        await p.key("Escape", 27)
        await p.js("document.querySelector('#search').blur()")
        await p.key("Escape", 27)
        return
    await p.wait_for("document.querySelector('.reader-text')?.innerText.length > 0", 10, "the memory file text")
    await p.wait(1.4)
    await p.shot("03-memory-file.png")
    opened_title = await p.js("document.querySelector('#inspector-title').textContent")
    print(f"  opened memory file: {opened_title}")
    await p.key("Escape", 27)
    await p.js("document.querySelector('#search').blur()")
    await p.key("Escape", 27)
    await p.wait(1.5)


async def hover_somewhere(p: Page) -> None:
    await p.js("document.querySelector('#reset-view').click()")      # selecting may have flown away
    await p.wait(1.6)
    hit = None
    for x in range(520, 900, 10):
        for y in range(250, 700, 45):
            await p.hover(x, y)
            await p.wait(0.05)
            text = await p.js(TOOLTIP_TEXT)
            if text:
                hit = (x, y, text.replace("\n", " | "))
                break
        if hit:
            break
    assert hit, "hovering the scene never showed a tooltip"
    print(f"  hover at {hit[:2]}: {hit[2]}")
    await p.shot("04-hover.png")


TABS = ("model", "sleep", "voice", "connections", "tools")


async def visit_tabs(p: Page) -> None:
    print("tabs")
    for i, name in enumerate(TABS):
        await p.js(f"document.querySelector('.rail-btn[data-tab={name}]').click()")
        await p.wait_for(DRAWER_READY, 15, f"the {name} tab")
        await p.wait(0.8)
        title = await p.js("document.querySelector('#drawer-title').textContent")
        cards = await p.js(CARD_TITLES)
        print(f"  {title}: {cards}")
        await p.shot(f"1{i}-tab-{name}.png")
    await p.js("document.querySelector('.rail-btn[data-tab=brain]').click()")
    await p.wait(0.5)


async def chat(p: Page, question: str, n: int, burst: bool = False) -> None:
    print(f"chat: {question}")
    await p.js("document.querySelector('#reset-view').click()")
    state = await p.js("document.querySelector('#engine-pill').dataset.state")
    if state != "ready":
        await p.js("document.querySelector('#engine-toggle').click()")
        print("  loading the model (first load can take a minute)…")
        await p.wait_for(ENGINE_SETTLED, 600, "the model")
        state = await p.js("document.querySelector('#engine-pill').dataset.state")
        assert state == "ready", await p.js("document.querySelector('#engine-text').textContent")
    await p.js("document.querySelector('#chat-input').focus()")
    memories = await p.js(MEMORY_COUNT)
    await p.type(question)
    await p.key("Enter", 13)
    if burst:                                   # catch a new memory forming in the graph
        try:
            await p.wait_for(f"{MEMORY_COUNT} !== {json.dumps(memories)}", 120, "a new memory")
            for k in range(14):
                await p.shot(f"08-birth-{n}-{k:02d}.png")
                await p.wait(0.2)
        except TimeoutError:
            print("  (no new memory from this message)")
    await p.wait_for(f"{ENGINE_BUSY} || {ANSWERS} >= {n}", 30, "the answer to start")
    await p.wait(0.6)
    await p.shot(f"06-answering-{n}.png")
    await p.wait_for(f"{ANSWERS} >= {n}", 300, "the answer")
    await p.wait(0.3)
    await p.shot(f"07-answered-{n}.png")
    answer = await p.js("[...document.querySelectorAll('.msg[data-role=assistant]')].pop().innerText")
    print("  answer:", answer.replace("\n", " | "))
    await p.wait(2.0)
    print(f"  spoken clips fetched so far: {await p.js(CLIPS_FETCHED)}")


async def voice(p: Page, n: int) -> None:
    """Hands-free with a fake microphone that plays --voice-file (Chrome/Edge fake capture)."""
    print("voice: hands-free")
    await p.js("document.querySelector('#handsfree-toggle').click()")
    await p.wait_for("document.querySelector('#handsfree-toggle').getAttribute('aria-pressed') === 'true'", 15,
                     "hands-free to switch on")
    await p.wait_for(f"document.querySelectorAll('.msg[data-role=user]').length >= {n}", 90, "a spoken question")
    await p.js("document.querySelector('#handsfree-toggle').click()")        # one question is enough
    heard = await p.js("[...document.querySelectorAll('.msg[data-role=user]')].pop().innerText")
    print("  heard:", heard.replace("\n", " | "))
    await p.wait_for(f"{ANSWERS} >= {n}", 300, "the answer")
    await p.wait(1.0)
    await p.shot("08-voice-answered.png")
    answer = await p.js("[...document.querySelectorAll('.msg[data-role=assistant]')].pop().innerText")
    print("  answer:", answer.replace("\n", " | "))
    tools = await p.js("[...document.querySelectorAll('.chip[data-source=tool]')].map(c => c.innerText).join(' / ')")
    print(f"  tool chips: {tools or '(none)'}")


async def run(url: str, questions: list[str], voice_file: str | None = None, search: str = "somni",
              burst: bool = False) -> int:
    browser = find_browser()
    profile = tempfile.mkdtemp(prefix="somni-ui-")
    mic = ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
           f"--use-file-for-fake-audio-capture={voice_file}"] if voice_file else []
    proc = subprocess.Popen([browser, "--headless=new", f"--remote-debugging-port={DEBUG_PORT}", *mic,
                             "--autoplay-policy=no-user-gesture-required",
                             "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                             f"--window-size={VIEW[0]},{VIEW[1]}", f"--user-data-dir={profile}", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{DEBUG_PORT}/json", timeout=2).read())
                break
            except OSError:
                time.sleep(0.2)
        else:
            raise SystemExit("the browser did not start")
        target = next(t for t in tabs if t["type"] == "page")
        async with websockets.connect(target["webSocketDebuggerUrl"], max_size=64_000_000) as ws:
            p = Page(ws)
            for domain in ("Runtime", "Page"):
                await p.call(f"{domain}.enable")
            await p.call("Emulation.setDeviceMetricsOverride", width=VIEW[0], height=VIEW[1], deviceScaleFactor=1,
                         mobile=False)
            await p.call("Page.navigate", url=url)
            try:
                await p.wait_for("document.querySelector('[data-stat=memories]')?.textContent !== '–'", 30,
                                 "the graph")
            except TimeoutError:
                print("PAGE ERRORS:", *(p.errors or ["(none reported)"]), sep="\n  ")
                raise
            await p.wait(2.5)
            await browse(p, search)
            await visit_tabs(p)
            for n, question in enumerate(questions, 1):
                await chat(p, question, n, burst)
            if voice_file:
                await voice(p, len(questions) + 1)
            if p.errors:
                print("PAGE ERRORS:", *p.errors, sep="\n  ")
                return 1
            print("ok: no page errors")
            return 0
    finally:
        proc.kill()
        shutil.rmtree(profile, ignore_errors=True)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):          # answers can contain any script (Windows consoles)
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8765/")
    parser.add_argument("--search", default="somni", help="a word to find in the memory graph")
    parser.add_argument("--voice-file", metavar="WAV",
                        help="also talk to it hands-free: a 48 kHz 16-bit WAV the fake microphone plays "
                             "(start it with the wake word)")
    parser.add_argument("--chat", metavar="QUESTION", action="append", default=[],
                        help="also load the model and ask this (repeatable)")
    parser.add_argument("--burst", action="store_true",
                        help="after each answer, 12 quick screenshots (to see a new memory form)")
    args = parser.parse_args()
    return asyncio.run(run(args.url, args.chat, args.voice_file, args.search, args.burst))


if __name__ == "__main__":
    sys.exit(main())
