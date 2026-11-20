"""Philips Hue over Bluetooth (with a simulated bulb) and colors in words."""
import struct
import unittest
from types import SimpleNamespace
from unittest import mock

from somni_studio.devices import bluetooth, colors
from somni_studio.devices.base import Command, DeviceError, Found, NeedsUser
from somni_studio.devices.bluetooth import BRIGHTNESS, COLOR, POWER, WHITE, HueBluetooth


class FakeBulb:
    """What a Hue bulb's Bluetooth light service looks like from the PC."""

    def __init__(self, color=True):
        self.values = {POWER: b"\x00", BRIGHTNESS: bytes([127]), WHITE: struct.pack("<H", 370)}
        if color:
            self.values[COLOR] = struct.pack("<HH", 30000, 20000)
        self.services = SimpleNamespace(get_characteristic=lambda u: u if u in self.values else None)
        self.is_connected = True
        self.writes = []
        self.paired = False

    async def read_gatt_char(self, uuid):
        if uuid not in self.values:
            raise bluetooth._bleak_errors()[0]("no such characteristic")
        return self.values[uuid]

    async def write_gatt_char(self, uuid, data, response=True):
        self.writes.append((uuid, bytes(data)))
        self.values[uuid] = bytes(data)


def bulb_client(bulb):
    async def client(address, pair=False):
        bulb.paired = bulb.paired or pair
        return bulb
    return client


LINK = {"id": "hue_ble:aa:bb", "driver": "hue_ble", "address": "AA:BB", "name": "Bedroom lamp", "model": "",
        "color": True}


class HueBluetoothTest(unittest.TestCase):
    def setUp(self):
        bluetooth.LOOP.failed_at.clear()

    def test_reads_the_real_state(self):
        bulb = FakeBulb()
        bulb.values[POWER] = b"\x01"
        with mock.patch.object(bluetooth, "_client", bulb_client(bulb)):
            [lamp] = HueBluetooth().devices(LINK)
        self.assertEqual((lamp.on, lamp.brightness, lamp.kind), (True, 50, "light"))
        self.assertTrue(lamp.color, "a color bulb reports what it shows")
        self.assertEqual(lamp.row()["color"], lamp.color)

    def test_on_off_brightness_color_and_white(self):
        bulb = FakeBulb()
        driver = HueBluetooth()
        with mock.patch.object(bluetooth, "_client", bulb_client(bulb)):
            driver.control(LINK, "", Command("set", brightness=100, color="blue"))
            self.assertEqual(bulb.values[POWER], b"\x01")
            self.assertEqual(bulb.values[BRIGHTNESS], bytes([254]))
            x, y = struct.unpack("<HH", bulb.values[COLOR])
            self.assertLess(x / 65535, 0.2, "blue is at the bottom left of the xy chart")
            driver.control(LINK, "", Command("set", color="warm white"))
            self.assertEqual(struct.unpack("<H", bulb.values[WHITE])[0], 370, "2700K is 370 mireds")
            driver.control(LINK, "", Command("turn_off"))
            self.assertEqual(bulb.values[POWER], b"\x00")
            with self.assertRaisesRegex(DeviceError, "not a color"):
                driver.control(LINK, "", Command("set", color="sparkly"))

    def test_white_only_bulb(self):
        bulb = FakeBulb(color=False)
        with mock.patch.object(bluetooth, "_client", bulb_client(bulb)):
            with self.assertRaisesRegex(DeviceError, "white-only"):
                HueBluetooth().control({**LINK, "color": False}, "", Command("set", color="red"))
            HueBluetooth().control({**LINK, "color": False}, "", Command("set", color="daylight"))
        self.assertEqual(struct.unpack("<H", bulb.values[WHITE])[0], 154)

    def test_connect_pairs_and_names(self):
        bulb = FakeBulb()
        bulb.values[bluetooth.MODEL] = b"LCA006"
        bulb.values[bluetooth.NAME] = b"Bedroom lamp\x00"
        with mock.patch.object(bluetooth, "_client", bulb_client(bulb)):
            extra = HueBluetooth().connect(Found("hue_ble", "AA:BB", "Hue color lamp"))
        self.assertTrue(bulb.paired)
        self.assertEqual(extra, {"color": True, "model": "Philips Hue LCA006 (Bluetooth)", "name": "Bedroom lamp"})

    def test_out_of_range_is_said_plainly_and_not_retried_at_once(self):
        calls = []

        async def gone(address, pair=False):
            calls.append(address)
            raise bluetooth._bleak_errors()[0]("Device with address AA:BB was not found")
        driver = HueBluetooth()
        with mock.patch.object(bluetooth, "_client", gone):
            with self.assertRaisesRegex(DeviceError, "out of Bluetooth range"):
                driver.devices(LINK)
            with self.assertRaisesRegex(DeviceError, "out of Bluetooth range"):
                driver.devices(LINK)
            with self.assertRaises(NeedsUser):
                driver.connect(Found("hue_ble", "AA:BB", "Hue lamp"))
        self.assertEqual(len(calls), 2, "the second read did not wait for another timeout")

    def test_finds_hue_bulbs_in_adverts(self):
        heard = {"1": (SimpleNamespace(address="AA:BB", name=None),
                       SimpleNamespace(service_uuids=[bluetooth.HUE_ADVERT], local_name="Hue color lamp", rssi=-60)),
                 "2": (SimpleNamespace(address="CC:DD", name="Phone"),
                       SimpleNamespace(service_uuids=[], local_name="Phone", rssi=-50))}
        with mock.patch.object(bluetooth.LOOP, "run", side_effect=lambda coro, t: (coro.close(), heard)[1]):
            found = HueBluetooth().discover(3, {})
        self.assertEqual([(f.name, f.key) for f in found], [("Hue color lamp", "hue_ble:aa:bb")])


class ColorsTest(unittest.TestCase):
    def test_words_to_colors(self):
        self.assertEqual(colors.parse("Warm White").kelvin, 2700)
        self.assertEqual(colors.parse("3000k").kelvin, 3000)
        self.assertEqual(colors.parse("#FF0000").rgb, (255, 0, 0))
        self.assertEqual(colors.parse("red color").name, "red")
        x, y = colors.parse("red").xy
        self.assertGreater(x, 0.6)
        self.assertEqual(colors.describe_xy(x, y), "red")
        self.assertEqual(colors.describe_kelvin(2700), "warm white (2700K)")
        with self.assertRaises(ValueError):
            colors.parse("loud")

    def test_home_assistant_lights_get_colors_too(self):
        from somni_studio.tools import catalog
        from somni_studio.tools.catalog import ToolContext, run_tool
        ctx = ToolContext(integration_config=lambda _id: {"url": "http://ha.test", "token": "t"},
                          integration_ready=lambda i: i == "home_assistant", allowed_domains=frozenset({"light"}),
                          question="make the lamp purple")
        with mock.patch.object(catalog.HOME, "attributes", return_value={}), \
                mock.patch.object(catalog.HOME, "call", return_value={"entity_id": "light.lamp"}) as call:
            out = run_tool("home_control", {"entity_id": "light.lamp", "action": "set", "color": "purple"}, ctx)
            run_tool("home_control", {"entity_id": "light.lamp", "action": "set", "color": "cool white"}, ctx)
        self.assertTrue(out.ok, out.error)
        self.assertEqual(call.call_args_list[0].args[2:], ("turn_on", {"rgb_color": [140, 0, 255]}))
        self.assertEqual(call.call_args_list[1].args[3], {"color_temp_kelvin": 5000})


if __name__ == "__main__":
    unittest.main()
