"""New v4.4 automated tests; hardware playback still requires a real Windows test."""
import queue
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from retro_autotune import BackingBand, HOP, SAMPLE_RATE, VoiceEffects
from retro_autotune_gui import AudioWorker
from mp3_recorder import Mp3Recorder


class VoiceFollowTests(unittest.TestCase):
    def test_follow_strength_controls_chord_cadence(self):
        for strength, interval in ((0.0, 32), (0.3, 16), (0.5, 16),
                                   (0.7, 8), (0.95, 4), (1.0, 4)):
            band = BackingBand(key='C', scale='major', follow_voice=True,
                               follow_strength=strength, bass_enabled=True,
                               guitar_enabled=True)
            self.assertEqual(band.follow_step_interval(), interval)
            # A sustained D4 melody best fits the ii chord (D minor).
            for i in range(25):
                band.feed_voice(62, confidence=0.99, rms=0.18)
            band._schedule_step(4, 0, [], [], [])
            if strength >= 0.9:
                self.assertEqual(band.chord_name, 'D')
            else:
                self.assertEqual(band.chord_name, 'C')
            for _ in range(25):
                band.feed_voice(62, confidence=0.99, rms=0.18)
            band._schedule_step(interval, 0, [], [], [])
            self.assertEqual(band.chord_name, 'D')
            previous = band.chord_degree
            for _ in range(100):
                band.feed_voice(None, confidence=0, rms=0)
            band._schedule_step(interval * 2, 0, [], [], [])
            self.assertEqual(band.chord_degree, previous, 'Should hold through silence')

    def test_audio_worker_live_follow_slider(self):
        fx = VoiceEffects(follow_voice=True, follow_strength=0.5)
        AudioWorker.apply_command(fx, 'follow_strength', 1.0)
        self.assertEqual(fx.band.follow_step_interval(), 4)
        AudioWorker.apply_command(fx, 'follow_strength', 0.0)
        self.assertEqual(fx.band.follow_step_interval(), 32)
        AudioWorker.apply_command(fx, 'band_level', 0.9)
        AudioWorker.apply_command(fx, 'drum_level', 0.75)
        AudioWorker.apply_command(fx, 'bass_level', 0.8)
        self.assertEqual(fx.band.band_level, 0.9)
        self.assertEqual(fx.band.drum_level, 0.75)
        self.assertEqual(fx.band.bass_level, 0.8)


class MP3Tests(unittest.TestCase):
    def test_mono_and_stereo_encoder_records_processed_audio(self):
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.TemporaryDirectory() as tmp:
            for channels in (1, 2):
                events = queue.SimpleQueue()
                path = Path(tmp) / f'mix_{channels}.mp3'
                recorder = Mp3Recorder(path, channels, events, sample_rate=SAMPLE_RATE)
                recorder.start()
                for i in range(100):
                    t = (np.arange(HOP) + i * HOP) / SAMPLE_RATE
                    x = (0.17 * np.sin(2 * np.pi * 330 * t)).astype('float32')
                    block = x[:, None] if channels == 1 else np.stack([x, 0.5*x], axis=1)
                    recorder.feed(block)
                recorder.stop()
                recorder.thread.join(timeout=20)
                self.assertFalse(recorder.thread.is_alive())
                messages = []
                while not events.empty():
                    messages.append(events.get_nowait())
                self.assertIn('record_saved', [kind for kind, _ in messages], messages)
                self.assertGreater(path.stat().st_size, 3000)
                # Decode the MP3 to verify genuine audio, not just a nonempty file.
                cmd = [exe, '-v', 'error', '-i', str(path), '-f', 'f32le',
                       '-acodec', 'pcm_f32le', 'pipe:1']
                decoded = subprocess.run(cmd, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, check=True, timeout=20)
                recovered = np.frombuffer(decoded.stdout, dtype='<f4')
                self.assertGreater(float(np.max(np.abs(recovered))), 0.025)


class ResizeAndChaosTests(unittest.TestCase):
    def test_gui_reflows_and_chaos_keeps_headphone_level(self):
        import tkinter as tk
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f'No GUI display available: {exc}')
        try:
            from retro_autotune_gui import StudioApp
            app = StudioApp(root)
            root.update()
            for width, expected in ((1140, 3), (850, 2), (640, 1)):
                root.geometry(f'{width}x760')
                root.update()
                self.assertEqual(app._layout_mode[0], expected)
                self.assertGreater(app.band_controls.winfo_width(), width / expected / 2)
            original_volume = app.volume_var.get()
            original_mix = app.reverb_mix_var.get()
            app.toggle_chaos()
            root.update()
            self.assertEqual(app.harmony_var.get(), 'wall')
            self.assertEqual(app.follow_strength_var.get(), 1)
            self.assertEqual(app.volume_var.get(), original_volume)
            app.toggle_chaos()
            root.update()
            self.assertEqual(app.reverb_mix_var.get(), original_mix)
            self.assertEqual(app.volume_var.get(), original_volume)
            self.assertEqual(app.harmony_var.get(), 'choir')
        finally:
            root.destroy()


if __name__ == '__main__':
    unittest.main()
