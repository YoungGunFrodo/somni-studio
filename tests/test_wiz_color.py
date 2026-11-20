"""WiZ bulbs (Philips Smart LED "connected by WiZ"): colors and whites over Wi-Fi."""
import unittest
from unittest import mock

from somni_studio.devices.base import Command, DeviceError
from somni_studio.devices.lights import Wiz

LINK = {"id": "wiz:a8bb50aabbcc", "driver": "wiz", "address": "192.168.1.40", "name": "Desk lamp"}


class WizColorTest(unittest.TestCase):
    def control(self, command):
        with mock.patch.object(Wiz, "_call", return_value={}) as call:
            Wiz().control(LINK, "", command)
        return call.call_args.args[1:]

    def test_colors_whites_and_plain_switching(self):
        self.assertEqual(self.control(Command("set", color="red")),
                         ("setPilot", {"state": True, "r": 255, "g": 0, "b": 0}))
        self.assertEqual(self.control(Command("set", brightness=40, color="warm white")),
                         ("setPilot", {"state": True, "temp": 2700, "dimming": 40}))
        self.assertEqual(self.control(Command("set", color="1800K"))[1]["temp"], 2200, "kept in the bulb's range")
        self.assertEqual(self.control(Command("turn_off", color="red")), ("setPilot", {"state": False}))
        self.assertEqual(self.control(Command("set", brightness=70)), ("setPilot", {"state": True, "dimming": 70}))
        with self.assertRaises(DeviceError):
            self.control(Command("set", color="sparkly"))

    def test_reports_what_it_shows(self):
        for pilot, shown in (({"state": True, "dimming": 80, "temp": 2700}, "warm white (2700K)"),
                             ({"state": True, "dimming": 80, "r": 0, "g": 0, "b": 255}, "blue"),
                             ({"state": True, "dimming": 80, "sceneId": 4}, "scene 4")):
            with mock.patch.object(Wiz, "_call", return_value=pilot):
                [lamp] = Wiz().devices(LINK)
            self.assertEqual(lamp.color, shown)
            self.assertEqual(lamp.row()["color"], shown)


if __name__ == "__main__":
    unittest.main()
