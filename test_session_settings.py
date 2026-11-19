"""Autosave/restore regression tests, including device reconnect and safety gates."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from session_settings import (VERSION, clean_session, device_match, load_session,
                              save_session, valid_values, valid_device)


class FileSettingsTests(unittest.TestCase):
    def test_round_trip_in_custom_user_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "RetroAutoTune" / "last_session.json"
            saved = {
                "version": VERSION,
                "settings": {"key": "F#", "scale": "major", "preset": "extreme",
                             "song_mode": True, "harmony": "wall", "follow_strength": 0.95,
                             "volume": 0.36, "bpm": 120},
                "devices": {"input": {"name": "USB Mic", "hostapi": "Windows WASAPI", "occurrence": 0},
                            "output": {"name": "Headphones", "hostapi": "Windows WASAPI", "occurrence": 0}},
                "chaos_enabled": True,
                "before_chaos": {"reverb_mix": 0.20, "harmony": "off"},
            }
            self.assertEqual(save_session(saved, path), path)
            self.assertEqual(load_session(path), saved)
            self.assertFalse(list(path.parent.glob("*.tmp")))
            # Writes are atomic on Windows/Linux and do not duplicate keys.
            saved["settings"]["volume"] = 0.5
            save_session(saved, path)
            self.assertEqual(load_session(path)["settings"]["volume"], 0.5)

    def test_corruption_and_out_of_range_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last_session.json"
            path.write_text("{garbage", encoding="utf-8")
            self.assertEqual(load_session(path), {})
            invalid = {"version": VERSION, "settings": {"bpm": True, "gain": 5.0,
                        "volume": 100, "reverb": "False", "scale": "not-a-scale",
                        "harmony": "wall", "follow_strength": float("nan")},
                       "devices": {"input": {"name": "Mic", "hostapi": "MME", "occurrence": -1},
                                   "output": {"name": "Headset", "hostapi": "WASAPI", "occurrence": 0}},
                       "chaos_enabled": "true"}
            cleaned = clean_session(invalid)
            self.assertEqual(cleaned["settings"], {"harmony": "wall"})
            self.assertIsNone(cleaned["devices"]["input"])
            self.assertEqual(cleaned["devices"]["output"]["name"], "Headset")
            self.assertFalse(cleaned["chaos_enabled"])
            self.assertEqual(clean_session({"version": 999}), {})

    def test_device_match_never_falls_back_to_a_different_speaker(self):
        preferred = {"name": "Headphones (USB)", "hostapi": "WASAPI", "occurrence": 0}
        options = {"2: Speakers [WASAPI]": {"name": "Speakers", "hostapi": "WASAPI", "occurrence": 0}}
        self.assertEqual(device_match(preferred, options), "")
        options["8: Headphones (USB) [WASAPI]"] = preferred
        self.assertEqual(device_match(preferred, options), "8: Headphones (USB) [WASAPI]")
        self.assertIsNone(valid_device({"name": "Headphones", "hostapi": "WASAPI", "occurrence": "0"}))


class DesktopSessionTests(unittest.TestCase):
    def _fake_sounddevice(self):
        # Virtual devices only; no real audio device or input/output stream opened.
        state = {"devices": [
            {"name": "USB Microphone", "hostapi": 0, "max_input_channels": 1, "max_output_channels": 0},
            {"name": "Headphones", "hostapi": 1, "max_input_channels": 0, "max_output_channels": 2},
        ]}
        api = types.SimpleNamespace(
            query_devices=lambda: state["devices"],
            query_hostapis=lambda: [{"name": "MME"}, {"name": "Windows WASAPI"}],
        )
        return state, api

    def test_full_gui_restore_and_safety_after_device_reorder(self):
        import tkinter as tk
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"No GUI display: {exc}")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last_session.json"
            state, device_module = self._fake_sounddevice()
            with patch.dict(sys.modules, {"sounddevice": device_module}):
                app = None
                try:
                    from retro_autotune_gui import StudioApp
                    app = StudioApp(root, settings_file=path)
                    self.assertIsNone(app.worker)
                    self.assertEqual(app.input_var.get(), "")  # No unsafe auto-selection.
                    self.assertFalse(app.headphones_var.get())
                    mic_label = next(iter(app.devices_in))
                    output_label = next(iter(app.devices_out))
                    app.input_var.set(mic_label)
                    app.output_var.set(output_label)
                    app.key_var.set("F#")
                    app.scale_var.set("major")
                    app.bpm_var.set(135)
                    app.harmony_var.set("wall")
                    app.reverb_mix_var.set(0.22)
                    app.volume_var.set(0.41)
                    app.song_mode_var.set(True)
                    app.toggle_chaos()
                    app._save_session_now()
                    self.assertTrue(path.is_file())
                    original = load_session(path)
                    self.assertEqual(original["devices"]["output"]["name"], "Headphones")
                    self.assertFalse("headphones_confirmed" in original)  # Never persisted.
                    self.assertTrue(original["chaos_enabled"])
                finally:
                    root.destroy()

                # Disconnect/reorder Windows audio devices; old index 1 is now a speaker.
                state["devices"] = [
                    {"name": "Speakers (HDMI)", "hostapi": 1, "max_input_channels": 0, "max_output_channels": 2},
                    {"name": "Laptop Microphone", "hostapi": 1, "max_input_channels": 1, "max_output_channels": 0},
                    {"name": "USB Microphone", "hostapi": 0, "max_input_channels": 1, "max_output_channels": 0},
                    {"name": "Headphones", "hostapi": 1, "max_input_channels": 0, "max_output_channels": 2},
                ]
                restored_root = tk.Tk()
                try:
                    restored = StudioApp(restored_root, settings_file=path)
                    self.assertEqual(restored.devices_in[restored.input_var.get()], 2)
                    self.assertEqual(restored.devices_out[restored.output_var.get()], 3)
                    self.assertFalse(restored.headphones_var.get())
                    self.assertIsNone(restored.worker)
                    self.assertEqual(restored.key_var.get(), "F#")
                    self.assertEqual(restored.scale_var.get(), "major")
                    self.assertEqual(restored.bpm_var.get(), 135)
                    self.assertAlmostEqual(restored.volume_var.get(), 0.41)
                    self.assertTrue(restored.song_mode_var.get())
                    self.assertTrue(restored.chaos_enabled)
                    restored.toggle_chaos()  # Reverts to pre-chaos session despite restart.
                    self.assertAlmostEqual(restored.reverb_mix_var.get(), 0.22)
                    restored._save_session_now()
                finally:
                    restored_root.destroy()

                # Saved headphones missing: do NOT fall back to the newly found speakers.
                state["devices"] = [
                    {"name": "Speakers (HDMI)", "hostapi": 1, "max_input_channels": 0, "max_output_channels": 2},
                    {"name": "USB Microphone", "hostapi": 0, "max_input_channels": 1, "max_output_channels": 0},
                ]
                missing_root = tk.Tk()
                try:
                    missing = StudioApp(missing_root, settings_file=path)
                    self.assertEqual(missing.output_var.get(), "")
                    self.assertEqual(missing.devices_in[missing.input_var.get()], 1)
                    self.assertEqual(missing.preferred_output["name"], "Headphones")
                    self.assertFalse(missing.headphones_var.get())
                    missing._save_session_now()
                    self.assertEqual(load_session(path)["devices"]["output"]["name"], "Headphones")
                finally:
                    missing_root.destroy()


if __name__ == "__main__":
    unittest.main()
