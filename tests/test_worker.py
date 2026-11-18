import asyncio
import threading
import unittest

from somni_studio.engine.worker import MAX_PENDING, EngineBusy, EngineState, EngineWorker
from somni_studio.events import EventBus, EventType
from somni_studio.settings import StudioSettings

from .fakes import FakeProvider, factory


async def collect(q: asyncio.Queue, until: str, timeout: float = 5.0) -> list[dict]:
    events = []
    while True:
        event = await asyncio.wait_for(q.get(), timeout)
        events.append(event)
        if event["type"] == until:
            return events


class WorkerTest(unittest.TestCase):
    def setUp(self):
        FakeProvider.instances.clear()

    def run_with_worker(self, scenario, **fake):
        async def main():
            bus = EventBus()
            bus.attach(asyncio.get_running_loop())
            q = bus.subscribe()
            worker = EngineWorker(bus, StudioSettings(), factory(**fake))
            try:
                return await scenario(worker, q)
            finally:
                await asyncio.to_thread(worker.shutdown)
        return asyncio.run(main())

    def test_ask_streams_events_in_order(self):
        async def scenario(worker, q):
            rid = worker.ask("  Who named Normandy?  ")
            events = [e for e in await collect(q, EventType.CHAT_DONE.value) if e.get("request_id") == rid]
            kinds = [e["type"] for e in events]
            self.assertEqual(kinds, ["chat.start", "recall", "activity", "token", "token", "chat.done"])
            self.assertEqual(events[0]["question"], "Who named Normandy?")
            self.assertEqual(events[-1]["used_node"], "mem:m1")
            # "ready" follows chat.done as its own event (the UI listens for it, never polls).
            while True:
                event = await asyncio.wait_for(q.get(), 5)
                if event["type"] == EventType.ENGINE_STATUS.value and event["state"] == EngineState.READY.value:
                    break
            self.assertEqual(worker.status()["state"], EngineState.READY.value)
        self.run_with_worker(scenario)

    def test_load_failure_is_reported_and_retried_next_time(self):
        async def scenario(worker, q):
            rid = worker.ask("hi")
            events = await collect(q, EventType.CHAT_ERROR.value)
            self.assertIn("no GPU here", events[-1]["message"])
            self.assertEqual(events[-1]["request_id"], rid)
            self.assertEqual(worker.status()["state"], EngineState.ERROR.value)
            self.assertTrue(FakeProvider.instances[0].closed, "a provider that failed to load is closed")
        self.run_with_worker(scenario, fail_load=True)

    def test_answer_failure_keeps_the_model_loaded(self):
        async def scenario(worker, q):
            worker.ask("hi")
            await collect(q, EventType.CHAT_ERROR.value)
            await asyncio.sleep(0.05)
            self.assertEqual(worker.status()["state"], EngineState.READY.value)
        self.run_with_worker(scenario, fail_ask=True)

    def test_queue_limit(self):
        gate = threading.Event()

        async def scenario(worker, q):
            for _ in range(MAX_PENDING):
                worker.ask("q")
            with self.assertRaises(EngineBusy):
                worker.ask("one too many")
            gate.set()
        self.run_with_worker(scenario, gate=gate)

    def test_rejects_empty_and_huge_questions(self):
        async def scenario(worker, q):
            with self.assertRaises(ValueError):
                worker.ask("   ")
            with self.assertRaises(ValueError):
                worker.ask("x" * 5000)
        self.run_with_worker(scenario)

    def test_reconfigure_closes_the_provider_only_when_needed(self):
        async def scenario(worker, q):
            worker.ask("hi")
            await collect(q, EventType.CHAT_DONE.value)
            provider = FakeProvider.instances[0]
            self.assertIsNone(worker.reconfigure(worker._settings.merged({"graph": {"poll_seconds": 5.0}})))
            self.assertFalse(provider.closed)
            await asyncio.wrap_future(worker.reconfigure(worker._settings.merged({"provider": {"model": "other"}})))
            self.assertTrue(provider.closed)
            self.assertEqual(worker.status()["state"], EngineState.UNLOADED.value)
        self.run_with_worker(scenario)


if __name__ == "__main__":
    unittest.main()
