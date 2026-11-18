"""Direct device control: protocols, the hub (scan, connect, follow, control), and the chat tools."""
import json
import struct
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from somni_studio.devices import hub as hub_module
from somni_studio.devices import lights, media, net, share
from somni_studio.devices.base import Command, Device, DeviceError, Driver, Found, NeedsUser
from somni_studio.devices.hub import DeviceHub
from somni_studio.tools.catalog import ToolContext, available_tools, run_tool


class ProtocolTest(unittest.TestCase):
    def test_kasa_cipher(self):
        plain = b'{"system":{"get_sysinfo":{}}}'
        secret = lights.Kasa.encrypt(plain)
        self.assertEqual(secret[0], 0x7B ^ 171)
        self.assertEqual(lights.Kasa.decrypt(secret), plain)

    def test_lifx_packets(self):
        packet = lights.Lifx.packet(lights.Lifx.GET, tagged=True)
        self.assertEqual(len(packet), 36)
        self.assertEqual(struct.unpack_from("<H", packet)[0], 36)
        mac = bytes.fromhex("d073d5112233")
        state = (struct.pack("<HHI", 88, 1024, 0) + struct.pack("<8s6sBB", mac + b"\0\0", b"\0" * 6, 0, 0)
                 + struct.pack("<QHH", 0, lights.Lifx.STATE, 0)
                 + struct.pack("<HHHHhH", 1, 2, 32768, 3500, 0, 65535) + b"Desk".ljust(32, b"\0") + b"\0" * 8)
        parsed = lights.Lifx.parse_state(state)
        self.assertEqual((parsed["mac"], parsed["on"], parsed["label"]), ("d073d5112233", True, "Desk"))

    def test_cast_messages(self):
        wire = media.encode_message("sender-0", "receiver-0", media.NS_RECEIVER, {"type": "GET_STATUS", "requestId": 7})
        self.assertEqual(struct.unpack(">I", wire[:4])[0], len(wire) - 4)
        decoded = media.decode_message(wire[4:])
        self.assertEqual((decoded["source"], decoded["namespace"]), ("sender-0", media.NS_RECEIVER))
        self.assertEqual(decoded["payload"], {"type": "GET_STATUS", "requestId": 7})

    def test_mdns_records(self):
        def name(text):
            return b"".join(bytes([len(p)]) + p.encode() for p in text.split(".")) + b"\0"
        service = "_googlecast._tcp.local"
        instance = "Nest-Hub-1." + service
        records = [(service, 12, name(instance)), (instance, 33, struct.pack(">HHH", 0, 0, 8009) + name("hub.local")),
                   (instance, 16, b"\x0cfn=Kitchen 1\x0fmd=Google Nest X")]
        body = b"".join(name(n) + struct.pack(">HHIH", t, 1, 120, len(r)) + r for n, t, r in records)
        packet = struct.pack(">6H", 0, 0x8400, 0, len(records), 0, 0) + body
        parsed = net.parse_mdns(packet)
        self.assertIn((service, 12, instance), parsed)
        self.assertIn((instance, 33, 8009), parsed)
        self.assertEqual(next(v for n, t, v in parsed if t == 16)["fn"], "Kitchen 1")

    def test_lan_only(self):
        self.assertTrue(net.is_lan("192.168.1.20"))
        for address in ("127.0.0.1", "8.8.8.8", "224.0.0.251", "nonsense"):
            self.assertFalse(net.is_lan(address), address)


class HueTest(unittest.TestCase):
    def test_button_then_key_then_lights_and_rooms(self):
        found = Found("hue", "192.168.1.5", "Hue Bridge", ident="abc")
        not_pressed = json.dumps([{"error": {"type": 101, "description": "link button not pressed"}}]).encode()
        with mock.patch.object(net, "http_get", return_value=not_pressed), self.assertRaisesRegex(NeedsUser, "button"):
            lights.Hue().connect(found)
        with mock.patch.object(net, "http_get", return_value=json.dumps([{"success": {"username": "k1"}}]).encode()):
            self.assertEqual(lights.Hue().connect(found), {"key": "k1"})
        link = {"id": "hue:abc", "address": "192.168.1.5", "key": "k1", "name": "Hue Bridge"}
        replies = {"/lights": {"1": {"name": "Desk", "state": {"on": True, "bri": 127, "reachable": True}}},
                   "/groups": {"2": {"name": "Living room", "type": "Room", "state": {"any_on": False},
                                     "action": {"bri": 254}}, "3": {"name": "x", "type": "LightGroup"}}}
        with mock.patch.object(net, "http_get", side_effect=lambda url, **k: json.dumps(
                replies[url.split("/k1", 1)[1]]).encode()):
            devices = lights.Hue().devices(link)
        self.assertEqual([(d.id, d.name, d.on, d.brightness) for d in devices],
                         [("hue:abc/light/1", "Desk", True, 50), ("hue:abc/group/2", "Living room", False, 100)])
        with mock.patch.object(net, "http_get") as put:
            lights.Hue().switch(link, "group/2", None, 30)
        self.assertTrue(put.call_args.args[0].endswith("/groups/2/action"))
        self.assertEqual(json.loads(put.call_args.kwargs["body"]), {"on": True, "bri": 76})


class FakeDriver(Driver):
    id, title = "fake", "Fake"

    def __init__(self):
        self.address, self.on, self.calls, self.down = "192.168.1.30", False, [], False

    def discover(self, timeout, seen):
        return [Found(self.id, self.address, "Desk lamp", "Fake", "aa11"),
                Found(self.id, "192.168.1.31", "Hall bulb", "Fake", "bb22", note="press its button first")]

    def connect(self, found):
        if found.note:
            raise NeedsUser("Press the button.")
        return {}

    def devices(self, link):
        if self.down or link["address"] != self.address:
            raise OSError("no route")
        return [Device(self.device_id(link), link["name"], "light", self.on, 80 if self.on else None, link=link["id"])]

    def switch(self, link, sub, on, brightness):
        if link["address"] != self.address:
            raise TimeoutError("no answer")
        self.calls.append((on, brightness))
        self.on = True if brightness else bool(on)


class HubTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.driver = FakeDriver()
        patches = [mock.patch.object(hub_module, "DRIVERS", (self.driver,)),
                   mock.patch.object(hub_module, "BY_ID", {"fake": self.driver}),
                   mock.patch.object(net, "ssdp", return_value={}), mock.patch.object(net, "mdns", return_value={}),
                   mock.patch.object(hub_module.time, "sleep")]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.path = Path(self.tmp.name) / "devices.json"
        self.hub = DeviceHub(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_connect_by_name_and_all(self):
        self.assertEqual(self.hub.connect("desk lamp"), {"connected": ["Desk lamp"]})
        self.assertEqual(self.hub.connect("all"), {"already": ["Desk lamp"]}, "a device needing a button is skipped")
        self.assertIn("Press the button", self.hub.connect("hall")["waiting"][0])
        self.assertEqual(json.loads(self.path.read_text())["links"][0]["id"], "fake:aa11")
        self.assertEqual([d.name for d in DeviceHub(self.path).devices()], ["Desk lamp"], "kept across restarts")
        with self.assertRaisesRegex(DeviceError, "could not find"):
            self.hub.connect("toaster")

    def test_control_toggle_and_find(self):
        self.hub.connect("desk lamp")
        lamp = self.hub.find("desk")[0]
        now = self.hub.control(lamp, Command("toggle"))
        self.assertTrue(now.on)
        self.hub.control(now, Command("set", brightness=40))
        self.assertEqual(self.driver.calls, [(True, None), (None, 40)])

    def test_follows_a_new_address(self):
        self.hub.connect("desk lamp")
        lamp = self.hub.devices()[0]
        self.driver.address = "192.168.1.77"                 # the router gave it a new one
        self.hub.control(lamp, Command("turn_on"))
        self.assertEqual(self.hub.links[0]["address"], "192.168.1.77")

    def test_unreachable_is_a_state(self):
        self.hub.connect("desk lamp")
        self.hub.devices()
        self.driver.down = True
        self.assertEqual(self.hub.devices()[0].state, "unreachable")

    def test_forget(self):
        self.hub.connect("desk lamp")
        self.assertTrue(self.hub.forget("fake:aa11"))
        self.assertEqual(self.hub.devices(), [])


class ToolsTest(HubTest):
    def context(self, question, say_on=None):
        return ToolContext(integration_config=lambda i: {}, integration_ready=lambda i: False,
                           allowed_domains=frozenset({"light", "switch", "media_player"}), question=question,
                           devices=self.hub, say_on=say_on)

    def test_device_tools_appear_once_something_is_connected(self):
        names = {t.name for t in available_tools(self.context("hi"))}
        self.assertTrue({"devices_find", "device_connect"} <= names)
        self.assertNotIn("home_control", names)
        self.hub.connect("desk lamp")
        self.assertIn("home_control", {t.name for t in available_tools(self.context("hi"))})

    def test_find_connect_and_control_by_name(self):
        found = run_tool("devices_find", {}, self.context("what devices do I have?"))
        self.assertIn("Not connected yet: Desk lamp, Hall bulb", found.result["summary"])
        refused = run_tool("device_connect", {"name": "desk lamp"}, self.context("what devices do I have?"))
        self.assertFalse(refused.ok, "connecting needs the user's go-ahead")
        self.assertTrue(run_tool("device_connect", {"name": "desk lamp"}, self.context("yes, connect it")).ok)
        record = run_tool("home_control", {"entity_id": "desk lamp", "action": "turn_on"},
                          self.context("turn on the desk lamp"))
        self.assertTrue(record.ok, record.error)
        self.assertEqual(record.result["now"]["state"], "on")
        states = run_tool("home_states", {"search": "desk"}, self.context("is the desk lamp on?"))
        self.assertEqual(states.result["devices"][0]["state"], "on")
        self.assertFalse(run_tool("home_control", {"entity_id": "desk lamp", "action": "turn_off"},
                                  self.context("is the desk lamp on?")).ok, "a question never switches it")

    def test_asking_about_a_device_that_is_not_connected(self):
        self.hub.connect("desk lamp")
        states = run_tool("home_states", {"search": "hall"}, self.context("is the hall bulb on?"))
        self.assertEqual(states.result["not_connected"], ["Hall bulb"])

    def test_speaker_say_needs_a_request(self):
        said = []
        self.hub._links.append({"id": "cast:1", "driver": "cast", "address": "192.168.1.9", "name": "Kitchen"})
        self.hub._devices = [Device("cast:1", "Kitchen display", "speaker", False, link="cast:1")]
        self.hub._devices_at = hub_module.time.monotonic()
        ctx = self.context("what's for dinner", say_on=lambda d, m: said.append((d.name, m)))
        self.assertFalse(run_tool("speaker_say", {"device": "kitchen", "message": "Dinner!"}, ctx).ok)
        ctx.question = "tell the kitchen that dinner is ready"
        self.assertTrue(run_tool("speaker_say", {"device": "kitchen", "message": "Dinner is ready"}, ctx).ok)
        self.assertEqual(said, [("Kitchen display", "Dinner is ready")])


class ShareTest(unittest.TestCase):
    def test_serves_one_file_to_one_device(self):
        with mock.patch.object(share, "local_address_toward", return_value="127.0.0.1"):
            url = share.share_once(b"RIFFdata", "127.0.0.1")
        self.assertEqual(urllib.request.urlopen(url, timeout=3).read(), b"RIFFdata")  # noqa: S310
        with self.assertRaises(urllib.error.HTTPError):
            urllib.request.urlopen(url.rsplit("/", 1)[0] + "/other.wav", timeout=3)  # noqa: S310


if __name__ == "__main__":
    unittest.main()
