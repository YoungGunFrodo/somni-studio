import json
import tempfile
import unittest
from pathlib import Path

from somni_studio.settings import (
    STUDIO_ROOT,
    SettingsError,
    StudioSettings,
    field_types,
    load_settings,
    resolve_path,
    save_settings,
)


class SettingsTest(unittest.TestCase):
    def test_defaults_are_valid_and_local(self):
        s = StudioSettings.from_dict({})
        self.assertEqual(s.server.host, "127.0.0.1")
        self.assertTrue(s.server.is_loopback)
        self.assertFalse(s.somni.read_only, "the assistant's own home learns from conversations")
        self.assertEqual(s.somni.home_path, STUDIO_ROOT / "data" / "home")
        self.assertTrue(s.voice.enabled)

    def test_unknown_keys_and_sections_are_rejected(self):
        with self.assertRaisesRegex(SettingsError, "somni.hme"):
            StudioSettings.from_dict({"somni": {"hme": "x"}})
        with self.assertRaisesRegex(SettingsError, "section"):
            StudioSettings.from_dict({"sommni": {}})

    def test_types_are_checked(self):
        with self.assertRaisesRegex(SettingsError, "server.port must be int"):
            StudioSettings.from_dict({"server": {"port": "8765"}})
        with self.assertRaisesRegex(SettingsError, "read_only must be bool"):
            StudioSettings.from_dict({"somni": {"read_only": 1}})
        with self.assertRaisesRegex(SettingsError, "unknown setting somni.base_model"):
            StudioSettings.from_dict({"somni": {"base_model": "x"}})   # the model lives in the home
        with self.assertRaisesRegex(SettingsError, "port must be int"):
            StudioSettings.from_dict({"server": {"port": True}})      # bool is not an int here

    def test_int_is_accepted_for_float(self):
        self.assertEqual(StudioSettings.from_dict({"graph": {"poll_seconds": 3}}).graph.poll_seconds, 3.0)

    def test_ranges_and_choices(self):
        for patch in ({"server": {"port": 0}}, {"provider": {"kind": "gpt"}}, {"graph": {"poll_seconds": 0.0}},
                      {"provider": {"kind": "openai", "url": "localhost:1234"}}, {"voice": {"speed": 3.0}},
                      {"voice": {"threads": 0}}):
            with self.subTest(patch=patch), self.assertRaises(SettingsError):
                StudioSettings().merged(patch)

    def test_merged_keeps_other_values(self):
        s = StudioSettings().merged({"provider": {"model": "qwen"}})
        self.assertEqual(s.provider.model, "qwen")
        self.assertEqual(s.provider.kind, "somni")

    def test_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "studio.json"
            s = StudioSettings().merged({"somni": {"home": "~/elsewhere", "read_only": False}})
            save_settings(s, path)
            self.assertEqual(load_settings(path), s)
            self.assertFalse(path.with_suffix(".json.tmp").exists())
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["somni"]["home"], "~/elsewhere")

    def test_missing_file_gives_defaults_and_bad_json_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(load_settings(Path(d) / "none.json"), StudioSettings())
            bad = Path(d) / "bad.json"
            bad.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(SettingsError, "not valid JSON"):
                load_settings(bad)

    def test_relative_paths_resolve_against_the_studio_folder(self):
        self.assertEqual(resolve_path("../x"), (STUDIO_ROOT / "../x").resolve())
        self.assertEqual(resolve_path("~/y"), Path.home() / "y")

    def test_field_types_cover_every_setting(self):
        types = field_types()
        self.assertEqual(types["graph.poll_seconds"], "float")
        self.assertEqual(types["server.port"], "int")
        self.assertEqual(types["somni.read_only"], "bool")
        self.assertEqual(len(types), sum(len(v) for v in StudioSettings().to_dict().values()))


if __name__ == "__main__":
    unittest.main()
