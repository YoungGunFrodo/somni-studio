"""More of Home Assistant: its own assistant, rooms, routines, discovered devices, its speakers."""
import unittest
from unittest import mock

from somni_studio.integrations import ha_features as f
from somni_studio.integrations.base import CheckFailed
from somni_studio.tools import catalog
from somni_studio.tools.catalog import ToolContext, run_tool

CONFIG = {"url": "http://ha.test:8123", "token": "t"}


def context(question: str) -> ToolContext:
    return ToolContext(integration_config=lambda _id: CONFIG, integration_ready=lambda i: i == "home_assistant",
                       allowed_domains=frozenset({"light", "switch", "media_player"}), question=question,
                       integration_details=lambda _id: {"services": []})


class RoutineTest(unittest.TestCase):
    def test_times_days_and_sun(self):
        body = f.routine_config("Lights out", "turn off all the lights", "10:30pm", "weekdays")
        self.assertEqual(body["triggers"], [{"trigger": "time", "at": "22:30:00"}])
        self.assertEqual(body["conditions"][0]["weekday"], ["mon", "tue", "wed", "thu", "fri"])
        self.assertEqual(body["actions"][0]["data"]["text"], "turn off all the lights")
        sun = f.routine_config("", "turn on the porch light", "sunset", "", -15)
        self.assertEqual(sun["triggers"][0], {"trigger": "sun", "event": "sunset", "offset": "-00:15:00"})
        self.assertEqual(sun["conditions"], [], "every day")
        self.assertEqual(f.routine_config("x", "y", "12am")["triggers"][0]["at"], "00:00:00")
        door = f.routine_config("x", "turn on the hall light", device="binary_sensor.door", becomes="on")
        self.assertEqual(door["triggers"][0], {"trigger": "state", "entity_id": "binary_sensor.door", "to": "on"})
        for bad in ({"at": "25:00"}, {"at": "soon"}, {}):
            with self.assertRaises(CheckFailed):
                f.routine_config("x", "y", **bad)

    def test_only_when_asked(self):
        with mock.patch.object(f, "save_routine", return_value={"created": "x", "id": "1"}) as save:
            refused = run_tool("home_routine", {"action": "create", "do": "turn off the lights", "at": "22:00"},
                               context("turn off the lights"))
            self.assertFalse(refused.ok)
            done = run_tool("home_routine", {"action": "create", "name": "Night", "do": "turn off the lights",
                                             "at": "22:00"}, context("turn off the lights every night at 10"))
        self.assertTrue(done.ok, done.error)
        self.assertEqual(save.call_count, 1)
        self.assertEqual(done.summary(), "routine 'x' made")


class AssistTest(unittest.TestCase):
    REPLY = {"response": {"response_type": "action_done", "speech": {"plain": {"speech": "Turned off 3 lights"}},
                          "data": {"success": [{"name": "Kitchen"}], "failed": []}}}

    def test_says_what_it_did(self):
        with mock.patch.object(f, "_call", return_value=self.REPLY) as call:
            out = f.assist(CONFIG, "turn off all the lights")
        self.assertEqual(call.call_args.args[1:3], ("POST", "/api/conversation/process"))
        self.assertEqual((out["home_assistant_said"], out["changed"]), ("Turned off 3 lights", ["Kitchen"]))

    def test_devices_that_did_not_change_are_a_failure(self):
        """Measured: Home Assistant said "Turned off the lights" while both bulbs refused the command."""
        reply = {"response": {"response_type": "action_done", "speech": {"plain": {"speech": "Turned off the lights"}},
                              "data": {"success": [{"name": "Office", "type": "area"}],
                                       "failed": [{"name": "Office light", "type": "entity"},
                                                  {"name": "Office light 2", "type": "entity"}]}}}
        with mock.patch.object(f, "_call", return_value=reply):
            out = f.assist(CONFIG, "turn off the office lights")
            self.assertEqual(out["result"], "failed")
            record = run_tool("home_assist", {"request": "turn off the office lights"}, context("turn the office off"))
        self.assertFalse(record.ok)
        self.assertIn("did NOT change: Office light, Office light 2", record.error)

    def test_a_question_never_becomes_a_change(self):
        with mock.patch.object(f, "_call", return_value=self.REPLY) as call:
            out = run_tool("home_assist", {"request": "turn off the kitchen lights"},
                           context("are the kitchen lights on?"))
            self.assertFalse(out.ok)
            self.assertFalse(call.called)
            asked = run_tool("home_assist", {"request": "are the kitchen lights on"}, context("are they on?"))
        self.assertTrue(asked.ok)


class RoomsTest(unittest.TestCase):
    def test_rooms_from_areas(self):
        f._areas_cache.clear()
        text = '[{"room": "Kitchen", "entities": ["light.k1", "light.k2"]}, {"room": "Bedroom", "entities": []}]'
        with mock.patch.object(f, "_call", return_value=text):
            self.assertEqual(f.rooms(CONFIG), {"light.k1": "Kitchen", "light.k2": "Kitchen"})
        self.assertEqual(f.room_names(CONFIG), ["Kitchen"], "cached")
        from somni_studio.integrations import home_assistant
        states = [{"entity_id": "light.k1", "attributes": {"friendly_name": "Ceiling"}, "state": "on"},
                  {"entity_id": "light.b1", "attributes": {"friendly_name": "Lamp"}, "state": "off"}]
        with mock.patch.object(home_assistant, "_request", return_value=states):
            rows = home_assistant.HomeAssistant().states(CONFIG, "kitchen")
        self.assertEqual([(r["entity_id"], r["room"]) for r in rows], [("light.k1", "Kitchen")])


class DiscoveredTest(unittest.TestCase):
    FLOWS = [{"flow_id": "a", "handler": "xbox", "context": {"source": "ssdp", "title_placeholders": {}}},
             {"flow_id": "b", "handler": "hue", "context": {"source": "zeroconf",
                                                            "title_placeholders": {"name": "Hue Bridge"}}},
             {"flow_id": "c", "handler": "demo", "context": {"source": "user"}}]

    def test_add_one_and_report_what_needs_the_user(self):
        replies = {"a": {"type": "create_entry"}, "b": {"type": "form", "data_schema": [{"name": "host"}]}}
        with mock.patch.object(f, "_ws", return_value=self.FLOWS), \
                mock.patch.object(f, "_call", side_effect=lambda c, m, path, body: replies[path.rsplit("/", 1)[1]]):
            self.assertEqual([d["name"] for d in f.discovered(CONFIG)], ["xbox", "Hue Bridge"])
            out = f.add_discovered(CONFIG, "all")
        self.assertEqual(out["added"], ["xbox"])
        self.assertIn("Hue Bridge: needs host", out["needs_user"])
        with mock.patch.object(f, "_ws", return_value=self.FLOWS), mock.patch.object(f, "missed_wiz", return_value=[]):
            refused = run_tool("home_discovered", {"action": "add", "name": "xbox"}, context("what's new?"))
        self.assertFalse(refused.ok)

    def test_wiz_bulbs_home_assistant_missed(self):
        from somni_studio.devices.base import Found
        bulbs = [Found("wiz", "192.168.1.22", "WiZ light 0001", "WiZ", "a8bb50000001"),
                 Found("wiz", "192.168.1.23", "WiZ light 0002", "WiZ", "a8bb50000002")]
        registry = [{"connections": [["mac", "a8:bb:50:00:00:01"]]}]
        with mock.patch("somni_studio.devices.lights.Wiz.discover", return_value=bulbs), \
                mock.patch.object(f, "_ws", return_value=registry):
            missed = f.missed_wiz(CONFIG)
        self.assertEqual([b["host"] for b in missed], ["192.168.1.23"], "the one already in Home Assistant is skipped")
        steps = iter([{"type": "form", "flow_id": "w", "data_schema": [{"name": "host"}]},
                      {"type": "create_entry", "title": "WiZ RGBW Tunable 0002"}])
        with mock.patch.object(f, "_call", side_effect=lambda *a: next(steps)) as call:
            self.assertEqual(f.add_by_address(CONFIG, "wiz", "192.168.1.23"), {"added": "WiZ RGBW Tunable 0002"})
        self.assertEqual(call.call_args.args[3], {"host": "192.168.1.23"})
        with mock.patch.object(f, "missed_wiz", return_value=missed), \
                mock.patch.object(f, "discovered", return_value=[]), \
                mock.patch.object(f, "add_by_address", return_value={"added": "WiZ RGBW Tunable 0002"}):
            out = run_tool("home_discovered", {"action": "add", "name": "all"}, context("add the new bulb"))
        self.assertTrue(out.ok, out.error)
        self.assertEqual(out.result["added"], ["WiZ RGBW Tunable 0002"])
        self.assertIn("home_organize", out.result["next"])


class OrganizeTest(unittest.TestCase):
    def test_name_and_room_on_the_device(self):
        sent = []

        def ws(config, message):
            sent.append(message)
            return {"config/entity_registry/get": {"device_id": "dev1"},
                    "config/area_registry/list": [{"area_id": "kitchen", "name": "Kitchen"}],
                    "config/area_registry/create": {"area_id": "office"}}.get(message["type"], {})
        with mock.patch.object(f, "_ws", ws):
            out = run_tool("home_organize", {"device": "light.wiz_1", "name": "Office light", "room": "office"},
                           context("this bulb is the office light"))
        self.assertTrue(out.ok, out.error)
        self.assertIn({"type": "config/area_registry/create", "name": "Office"}, sent, "a new room is made")
        self.assertIn({"type": "config/device_registry/update", "device_id": "dev1", "area_id": "office",
                       "name_by_user": "Office light"}, sent)
        self.assertEqual(out.summary(), "Office light in office")
        with mock.patch.object(f, "_ws", ws):
            self.assertFalse(run_tool("home_organize", {"device": "light.wiz_1", "name": "x"},
                                      context("is the lamp on?")).ok)

    def test_a_new_bulb_is_added_first(self):
        """Measured: asked to name 'the one not in Home Assistant', a 4B model passed 'WiZ bulb 0002'
        (not added yet) and then a made-up entity id. Both now add the bulb and name it."""
        from somni_studio.integrations import home_assistant
        sent = []
        added = {"done": False}

        def ws(config, message):
            sent.append(message)
            if message["type"] == "config/entity_registry/get":
                if message["entity_id"] != "light.wiz_rgbw_tunable_0002":
                    raise CheckFailed("Home Assistant said: Entity not found")
                return {"device_id": "dev2"}
            if message["type"] == "config/area_registry/list":
                return [{"area_id": "office", "name": "Office"}]
            return {}

        def states(self, config, search="", domain="", limit=40):
            if added["done"] and "0002" in search.lower():
                return [{"entity_id": "light.wiz_rgbw_tunable_0002", "name": "WiZ RGBW Tunable 0002"}]
            return []

        def add(config, handler, host):
            added["done"] = True
            return {"added": "WiZ RGBW Tunable 0002"}
        missed = [{"name": "WiZ bulb 0002 at 192.168.1.23", "host": "192.168.1.23", "kind": "wiz"}]
        for device in ("WiZ bulb 0002", "light.wiz_bulb_0002", "the new bulb"):
            added["done"] = False
            sent.clear()
            with mock.patch.object(f, "_ws", ws), mock.patch.object(f, "missed_wiz", return_value=missed), \
                    mock.patch.object(f, "add_by_address", side_effect=add) as adding, \
                    mock.patch.object(home_assistant.HomeAssistant, "states", states):
                out = run_tool("home_organize", {"device": device, "name": "Office light 2", "room": "Office"},
                               context("name it office light 2 and put it in the office room"))
            self.assertTrue(out.ok, f"{device}: {out.error}")
            adding.assert_called_once_with(CONFIG, "wiz", "192.168.1.23")
            self.assertTrue(out.result["added_to_home_assistant"])
            self.assertIn({"type": "config/device_registry/update", "device_id": "dev2", "area_id": "office",
                           "name_by_user": "Office light 2"}, sent)


class DevicesFindTest(unittest.TestCase):
    def test_bulbs_already_in_home_assistant_are_marked(self):
        from somni_studio.devices.base import Found
        scan = [Found("wiz", "192.168.1.22", "WiZ light 0001", "WiZ", "a8bb50000001"),
                Found("wiz", "192.168.1.23", "WiZ light 0002", "WiZ", "a8bb50000002")]
        ctx = context("scan for the new bulb")
        ctx.devices = mock.Mock(links=[], scan=mock.Mock(return_value=scan))
        with mock.patch.object(f, "known_macs", return_value={"a8bb50000001"}):
            out = run_tool("devices_find", {}, ctx)
        self.assertTrue(out.ok, out.error)
        status = {d["name"]: d["status"] for d in out.result["found"]}
        self.assertIn("already in Home Assistant", status["WiZ light 0001"])
        self.assertIn("NEW", status["WiZ light 0002"])
        self.assertNotIn("then", out.result, "only a scan was asked for")
        ctx.question = "name the new one office light 2 and put it in the office"
        with mock.patch.object(f, "known_macs", return_value={"a8bb50000001"}):
            out = run_tool("devices_find", {}, ctx)
        self.assertEqual(out.result["then"]["tool"], "home_organize")
        self.assertIn("WiZ light 0002", out.result["then"]["say"])

    def test_the_agent_takes_the_next_step(self):
        """A scan that says the next step, and a model that answers instead: it is nudged to do it."""
        from somni_studio.engine.agent import Step, ToolAgent
        from somni_studio.tools.catalog import BY_NAME
        steps = iter([Step(calls=[{"name": "devices_find", "arguments": {}}]),
                      Step(text="Which bulb do you mean?"),
                      Step(calls=[{"name": "home_organize", "arguments": {"device": "WiZ light 0002",
                                                                          "name": "Office light 2",
                                                                          "room": "Office"}}]),
                      Step(text="Done, sir.")])
        seen = []
        def step(messages, *_args, **_kwargs):
            seen.append(messages[-1]["content"])
            return next(steps)
        backend = mock.Mock(step=mock.Mock(side_effect=step))
        ctx = context("name the new bulb office light 2 and put it in the office")
        results = {"devices_find": {"summary": "x", "found": [], "count": 1,
                                    "then": {"tool": "home_organize", "say": "The new bulb is WiZ light 0002."}},
                   "home_organize": {"organized": "light.0002", "name": "Office light 2", "room": "Office"}}
        with mock.patch("somni_studio.engine.agent.run_tool",
                        side_effect=lambda name, args, c: catalog.CallRecord(name, args, True, results[name])):
            result = ToolAgent(backend, ctx, mock.Mock()).run(ctx.question, [], [BY_NAME["devices_find"],
                                                                               BY_NAME["home_organize"]])
        self.assertEqual(result.text, "Done, sir.")
        self.assertEqual([r.name for r in result.calls], ["devices_find", "home_organize"])
        nudge = seen[2]
        self.assertIn("WiZ light 0002", nudge)


class NotDoneTest(unittest.TestCase):
    def run_agent(self, question, steps, results):
        from somni_studio.engine.agent import ToolAgent
        from somni_studio.tools.catalog import BY_NAME
        steps = iter(steps)
        seen = []

        def step(messages, *_args, **_kwargs):
            seen.append(messages[-1]["content"])
            return next(steps)
        ctx = context(question)
        with mock.patch("somni_studio.engine.agent.run_tool",
                        side_effect=lambda name, args, c: catalog.CallRecord(name, args, True, results[name])):
            result = ToolAgent(mock.Mock(step=mock.Mock(side_effect=step)), ctx, mock.Mock()).run(
                question, [], [BY_NAME[n] for n in ("home_states", "home_assist")])
        return result, seen

    def test_looking_is_not_doing(self):
        from somni_studio.engine.agent import Step
        result, seen = self.run_agent("turn the office lights off", [
            Step(calls=[{"name": "home_states", "arguments": {"search": "office"}}]),
            Step(text="I've turned off both lights."),                       # the false claim
            Step(calls=[{"name": "home_assist", "arguments": {"request": "turn off the office lights"}}]),
            Step(text="Done, sir: both office lights are off.")],
            {"home_states": {"devices": [], "count": 2},
             "home_assist": {"asked": "turn off the office lights", "home_assistant_said": "Turned off 2 lights",
                             "result": "action_done"}})
        self.assertEqual([r.name for r in result.calls], ["home_states", "home_assist"])
        self.assertIn("nothing has been done yet", seen[2])
        self.assertEqual(result.text, "Done, sir: both office lights are off.")

    def test_a_question_is_answered_from_looking(self):
        from somni_studio.engine.agent import Step
        result, _ = self.run_agent("are the office lights on?", [
            Step(calls=[{"name": "home_states", "arguments": {"search": "office"}}]),
            Step(text="Yes, both are on.")], {"home_states": {"devices": [], "count": 2}})
        self.assertEqual(result.text, "Yes, both are on.")


class MusicTest(unittest.TestCase):
    def test_plays_on_a_home_assistant_speaker(self):
        speakers = [{"entity_id": "media_player.kitchen_speaker", "name": "Kitchen speaker", "state": "off"},
                    {"entity_id": "media_player.living_room_display", "name": "Living room display", "state": "off"}]
        station = {"url": "http://radio.test/jazz", "name": "Jazz FM", "content_type": "audio/mpeg"}
        with mock.patch.object(f, "speakers", return_value=speakers), \
                mock.patch("somni_studio.devices.radio.find_station", return_value=station), \
                mock.patch.object(f, "play_url") as play:
            out = run_tool("play_music", {"device": "kitchen speaker", "what": "jazz"}, context("play jazz"))
        self.assertTrue(out.ok, out.error)
        play.assert_called_once_with(CONFIG, "media_player.kitchen_speaker", "http://radio.test/jazz")
        self.assertTrue(catalog.BY_NAME["play_music"].available(context("play jazz")))


if __name__ == "__main__":
    unittest.main()
