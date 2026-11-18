"""Non-hardware smoke tests. GUI window tests require a desktop or Xvfb."""
import unittest
import numpy as np
from retro_autotune import HOP, VoiceEffects, BackingBand
from retro_autotune_gui import AudioWorker

class ControlTests(unittest.TestCase):
    def test_live_controls(self):
        fx = VoiceEffects(key='C', scale='minor', preset='extreme', mono_output=True,
                          reverb='cher', drums=True, bass=True, backing_vocals='choir')
        controls = [
            ('preset', 'classic'), ('reverb', False), ('reverb_mix', 0.3),
            ('harmony', 'duo'), ('harmony_level', 0.2), ('drums', False),
            ('bass', False), ('guitar', True), ('follow_voice', True),
            ('guitar_level', 0.67), ('bpm', 130), ('groove', 'pop'),
            ('voice_octave', -1), ('octave_mix', 0.0), ('gain', 0.85),
            ('key_scale', ('F#', 'major')),
        ]
        for name, value in controls:
            AudioWorker.apply_command(fx, name, value)
        self.assertEqual(fx.preset, 'classic')
        self.assertFalse(fx.reverb_enabled)
        self.assertEqual(fx.band.bpm, 130)
        self.assertEqual(fx.band.root, 6)
        self.assertEqual(fx.voice_octave, -1)
        self.assertEqual(fx.scale, 'major')
        self.assertTrue(fx.band.guitar_enabled)
        self.assertTrue(fx.band.follow_voice)
        self.assertEqual(fx.band.guitar_level, 0.67)
        self.assertEqual(fx.harmony_mode, 'duo')
        y = fx.process(np.zeros(HOP, dtype=np.float32))
        self.assertEqual(y.shape, (HOP, 1))
        self.assertTrue(np.isfinite(y).all())

if __name__ == '__main__':
    unittest.main()

class AudioThreadTests(unittest.TestCase):
    def test_worker_fake_portaudio(self):
        """Exercise the real callback code without needing a Windows sound card."""
        import sys
        import types
        import threading
        import time

        class FakeStream:
            def __init__(self, **kwargs):
                self.callback = kwargs['callback']
                self.active = True
                self.do_stop = threading.Event()
                self.count = 0
                self.generated_audio = []
            def __enter__(self):
                def work():
                    for i in range(18):
                        if self.do_stop.is_set(): break
                        t = np.arange(HOP) + i * HOP
                        samples = (0.18*np.sin(2*np.pi*220*t/48000)).astype(np.float32)
                        output = np.zeros((HOP, 1), np.float32)
                        self.callback(samples[:, None], output, HOP, None, None)
                        self.generated_audio.append(output)
                        self.count += 1
                        time.sleep(0.005)
                self.thread = threading.Thread(target=work, daemon=True)
                self.thread.start()
                return self
            def __exit__(self, *args):
                self.do_stop.set()
                self.thread.join(timeout=5)
                self.active = False

        fake = types.SimpleNamespace(
            Stream=FakeStream,
            CallbackAbort=type('CallbackAbort', (Exception,), {}),
            check_input_settings=lambda **kw: None,
            check_output_settings=lambda **kw: None,
        )
        previous = sys.modules.get('sounddevice')
        sys.modules['sounddevice'] = fake
        try:
            worker = AudioWorker({
                'input': 1,'output': 5,'stereo': False, 'key': 'C', 'scale': 'minor',
                'preset': 'extreme', 'reverb': True, 'reverb_mix':0.5,
                'octave_mix':0.2, 'harmony':'choir', 'harmony_level':0.5,
                'drums':True, 'bass':True, 'guitar':True, 'follow_voice':True,
                'guitar_level':0.45, 'bpm':110, 'groove':'dance',
                'voice_octave':0, 'gain':1.0, 'volume':0.55,
            })
            worker.start()
            worker.command('voice_octave', -1)
            worker.command('bpm', 115)
            time.sleep(0.4)
            worker.stop()
            worker.thread.join(timeout=5)
            self.assertFalse(worker.thread.is_alive())
            self.assertIsNone(worker.callback_error)
            events = []
            while not worker.events.empty():
                events.append(worker.events.get_nowait()[0])
            self.assertIn('running', events)
            self.assertNotIn('error', events)
            self.assertFalse(worker.meters.empty())
            sample = worker.meters.get_nowait()
            self.assertEqual(sample['samples'].shape, (2048,))
            self.assertTrue(np.isfinite(sample['samples']).all())
        finally:
            if previous is None:
                sys.modules.pop('sounddevice', None)
            else:
                sys.modules['sounddevice'] = previous

class AudioRoutingTests(unittest.TestCase):
    """Device host-API fallback must not break live vocal rendering."""

    def test_cross_api_pair_uses_split_input_output_streams(self):
        self._exercise_split_streams(different_apis=True)

    def test_portaudio_minus_9993_retries_with_split_streams(self):
        self._exercise_split_streams(different_apis=False)

    def _exercise_split_streams(self, different_apis):
        import sys
        import time
        import threading
        import types

        threads_started = []
        outputs = []
        class FakeStream:
            def __init__(self, callback, **kwargs):
                self.callback = callback
                self.kwargs = kwargs
                self.active = False
                self.stopped = threading.Event()
            def __enter__(self):
                self.active = True
                self.thread = threading.Thread(target=self._emit, daemon=True)
                self.thread.start()
                return self
            def _emit(self):
                for i in range(50):
                    if self.stopped.is_set():
                        break
                    if self.kwargs.get('device') == 1:
                        t = np.arange(HOP) + i * HOP
                        data = (0.15 * np.sin(2 * np.pi * 220 * t / 48000)).astype('float32')
                        self.callback(data[:, None], HOP, None, None)
                    else:
                        data = np.zeros((HOP, self.kwargs['channels']), dtype='float32')
                        self.callback(data, HOP, None, None)
                        outputs.append(data.copy())
                    time.sleep(0.004)
            def __exit__(self, *exc):
                self.stopped.set()
                self.thread.join(timeout=3)
                self.active = False

        class DuplexShouldNotBeUsed:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            def __enter__(self):
                if different_apis:
                    raise AssertionError('Duplex opened across incompatible host APIs')
                raise RuntimeError('Error opening Stream: Illegal combination of I/O devices [PaErrorCode -9993]')
            def __exit__(self, *exc):
                return False

        def mkstream(kind):
            def constructor(**kwargs):
                threads_started.append(kind)
                return FakeStream(**kwargs)
            return constructor

        fake = types.SimpleNamespace(
            Stream=DuplexShouldNotBeUsed,
            InputStream=mkstream('mic'),
            OutputStream=mkstream('phones'),
            CallbackAbort=type('CallbackAbort', (Exception,), {}),
            query_devices=lambda i: {'hostapi': 0 if i == 1 else (1 if different_apis else 0)},
            check_input_settings=lambda **kw: None,
            check_output_settings=lambda **kw: None,
        )
        previous = sys.modules.get('sounddevice')
        sys.modules['sounddevice'] = fake
        worker = AudioWorker({
            'input': 1, 'output': 5, 'stereo': False, 'key': 'C', 'scale': 'minor',
            'preset': 'extreme', 'reverb': True, 'reverb_mix': 0.5,
            'octave_mix': 0.2, 'harmony': 'duo', 'harmony_level': 0.5,
            'drums': True, 'bass': True, 'guitar': True, 'follow_voice': True,
            'guitar_level': 0.45, 'bpm': 110, 'groove': 'dance',
            'voice_octave': 0, 'gain': 1.0, 'volume': 0.55,
        })
        try:
            worker.start()
            time.sleep(0.16)
            worker.stop()
            worker.thread.join(timeout=5)
            self.assertFalse(worker.thread.is_alive())
            self.assertIsNone(worker.callback_error)
            events = []
            while not worker.events.empty():
                events.append(worker.events.get_nowait())
            self.assertIn('running', [e[0] for e in events], events)
            self.assertNotIn('error', [e[0] for e in events], events)
            self.assertIn('mic', threads_started)
            self.assertIn('phones', threads_started)
            self.assertGreater(len(outputs), 1)
            self.assertTrue(all(a.shape == (HOP, 1) and np.isfinite(a).all() for a in outputs))
        finally:
            worker.stop()
            worker.thread.join(timeout=5)
            if previous is None:
                sys.modules.pop('sounddevice', None)
            else:
                sys.modules['sounddevice'] = previous


class BackingBandTests(unittest.TestCase):
    def test_guitar_synthesis_and_vocal_chord_following(self):
        band = BackingBand(key="C", scale="major", bpm=110,
                           guitar_enabled=True, bass_enabled=True,
                           drums_enabled=True, follow_voice=True)
        self.assertEqual(band.chord_degree, 0)
        for _ in range(28):
            band.feed_voice(note=62, confidence=0.99, rms=0.18)  # D4
        # Vocal changes must not interrupt the ongoing chord mid-bar.
        self.assertEqual(band.chord_degree, 0)
        drums, bass, guitar = [], [], []
        band._schedule_step(16, 0, drums, bass, guitar)
        self.assertEqual(band.chord_degree, 1)  # D minor diatonic harmony
        self.assertEqual(band.chord_name, "D")
        self.assertEqual(band.bass_midi(1, 0), 38)
        self.assertEqual(len(guitar), 1)
        self.assertGreater(np.max(abs(guitar[0][0])), .05)
        for _ in range(100):
            band.feed_voice(note=None, confidence=0, rms=0)
        band._schedule_step(32, 0, [], [], [])
        self.assertEqual(band.chord_degree, 1)  # Hold across a rest

    def test_fixed_progression_and_voice_dynamic_independence(self):
        band = BackingBand(key="C", scale="minor", guitar_enabled=True,
                           bass_enabled=True, follow_voice=False)
        for _ in range(40):
            band.feed_voice(note=65, confidence=.99, rms=.2)
        band._schedule_step(16, 0, [], [], [])
        self.assertEqual(band.bass_midi(1, 0), 44)  # Original Ab root
        self.assertIn((band.chord_degree, "down"), band.guitar_samples)

    def test_full_mix_with_guitar_follow(self):
        fx = VoiceEffects(key="G", scale="major", drums=True, bass=True,
                          guitar=True, follow_voice=True, backing_vocals="wall",
                          preset="extreme", reverb="cher", mono_output=False)
        signals=[]
        for i in range(70):
            samples = (0.17 * np.sin(2*np.pi*220 *
                        (np.arange(HOP)+i*HOP)/48000)).astype("float32")
            signals.append(fx.process(samples))
        mix = np.concatenate(signals)
        self.assertEqual(mix.shape, (70*HOP,2))
        self.assertTrue(np.isfinite(mix).all())
        self.assertLessEqual(float(np.max(abs(mix))), .88)
        self.assertGreater(float(np.max(abs(mix))), .05)
