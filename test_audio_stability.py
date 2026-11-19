"""Pure/simulated audio dropout and DSP-deadline regression tests (no sound card)."""
import unittest
from unittest.mock import patch
import numpy as np
from retro_autotune import (HOP, SAMPLE_RATE, BackingBand, FeedbackDelay, SimpleHighPass,
                            StreamingPitchShifter, VoiceEffects, detect_pitch_yin)
from retro_autotune_gui import (AUDIO_PROFILES, AudioWorker, MicGapSmoother, SplitMicBuffer,
                                stream_latency)


class AudioStabilityTests(unittest.TestCase):
    def test_audio_profiles_choose_explicit_latency_without_changing_fast(self):
        self.assertEqual(stream_latency("Fast"), "low")
        self.assertEqual(stream_latency("Balanced"), .030)
        self.assertEqual(stream_latency("Stable"), .065)
        self.assertEqual(stream_latency("unknown"), "low")
        self.assertEqual(set(AUDIO_PROFILES), {"Fast", "Balanced", "Stable"})

    def test_split_mic_dropout_crossfades_instead_of_hard_zero(self):
        smoother = MicGapSmoother(ramp=96)
        original = np.full((HOP, 1), .24, dtype=np.float32)
        self.assertIs(smoother.process(original, False), original)
        missing = smoother.process(np.zeros_like(original), True)
        self.assertAlmostEqual(float(missing[0, 0]), .24, places=5)
        self.assertAlmostEqual(float(missing[95, 0]), 0.0, places=5)
        self.assertTrue(np.all(missing[96:, 0] == 0.0))
        self.assertTrue(np.all(np.diff(missing[:96, 0]) <= 0.0))
        repeated = smoother.process(np.zeros_like(original), True)
        self.assertTrue(np.all(repeated == 0.0))
        resumed = smoother.process(original, False)
        self.assertAlmostEqual(float(resumed[0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(resumed[95, 0]), .24, places=5)
        self.assertTrue(np.all(np.diff(resumed[:96, 0]) >= 0.0))
        # No processing on normally delivered mic blocks.
        self.assertIs(smoother.process(original, False), original)

    def test_cpu_guard_reduces_yin_calls_without_disabling_band(self):
        fx = VoiceEffects(key="C", scale="minor", preset="extreme", mono_output=True,
                          reverb="off", backing_vocals="off", drums=True, bass=True, guitar=True)
        t = np.arange(HOP * 12)
        audio = (0.20 * np.sin(2 * np.pi * 220 * t / SAMPLE_RATE)).astype(np.float32)
        with patch("retro_autotune.detect_pitch_yin", wraps=detect_pitch_yin) as detector:
            fx.pitch_detection_stride = 1
            for i in range(6):
                fx.process(audio[i * HOP:(i + 1) * HOP])
            normal = detector.call_count
            fx.pitch_detection_stride = 3
            for i in range(6, 12):
                fx.process(audio[i * HOP:(i + 1) * HOP])
            reduced = detector.call_count - normal
        self.assertEqual(normal, 6)
        self.assertEqual(reduced, 2)
        self.assertTrue(fx.band.drums_enabled)
        self.assertTrue(fx.band.bass_enabled)
        self.assertTrue(fx.band.guitar_enabled)

    def test_neutral_lead_impulse_reaches_output_in_641_samples(self):
        fx = VoiceEffects(preset="classic", octave_mix=0, reverb="off")
        blocks = np.zeros((5, HOP), dtype=np.float32)
        blocks[0, 0] = 0.8
        out = np.concatenate([fx.process(block)[:, 0] for block in blocks])
        self.assertEqual(np.argmax(np.abs(out)), 641)
        self.assertFalse(np.any(out[:641]))

    def test_grain_heads_read_the_right_samples_across_ring_wraps_and_octave_toggles(self):
        shifter = StreamingPitchShifter()
        source = np.random.default_rng(8).normal(size=HOP * 32).astype(np.float32)
        timeline = np.arange(len(source))
        phase = 0.0
        for i, block in enumerate(source.reshape(-1, HOP)):
            shift = (0, -12.0, 3.4, 12.0)[i % 4]
            audible = i % 3 != 1
            out = shifter.process(block, shift, audible=audible)
            delta = (1 - 2 ** (shift / 12)) / 1280
            a = (phase + delta * np.arange(HOP)) % 1
            b = (a + .5) % 1
            pos = i * HOP + np.arange(HOP)
            gain = (1 - np.cos(2 * np.pi * a)) * .5
            expected = (gain * np.interp(pos - 1 - 1280*a, timeline, source, left=0) +
                        (1-gain) * np.interp(pos - 1 - 1280*b, timeline, source, left=0))
            phase = (phase + HOP*delta) % 1
            if audible:
                np.testing.assert_allclose(out, expected, atol=1e-6)
            else:
                self.assertIsNone(out)

    def test_block_filters_match_sample_recurrences_across_wraps(self):
        x = np.random.default_rng(5).normal(size=HOP*18).astype(np.float32)
        hp = SimpleHighPass()
        last = filtered = 0.0
        ref = np.empty_like(x)
        for i, value in enumerate(x):
            filtered = hp.alpha * (filtered + float(value) - last)
            ref[i] = filtered
            last = float(value)
        # Nonuniform call boundaries exercise the carried filter state.
        np.testing.assert_allclose(np.concatenate([hp.process(part) for part in
                                                   np.array_split(x, 19)]), ref, atol=1e-6)
        length = 691
        delay = FeedbackDelay(length)
        expected = np.zeros_like(x)
        history = np.zeros(len(x) + length, dtype=np.float32)
        for i, value in enumerate(x):
            expected[i] = history[i]
            prev = history[i-1] if i else 0
            history[i+length] = value + .79*(.58*history[i] + .42*prev)
        out = np.concatenate([delay.process(block, .79, .42) for block in x.reshape(-1, HOP)])
        np.testing.assert_allclose(out, expected, atol=2e-6)

    def test_muted_reverb_fades_out_then_restarts_without_stale_audio(self):
        fx = VoiceEffects(reverb="off", reverb_mix=.6, octave_mix=0)
        block = np.random.default_rng(6).normal(0, .2, HOP).astype(np.float32)
        silence = np.zeros_like(block)
        with patch.object(fx.reverb_engine, "process", wraps=fx.reverb_engine.process) as render:
            for _ in range(10):
                fx.process(block)
            self.assertEqual(render.call_count, 0)
            fx.reverb_enabled = True
            for _ in range(40):
                fx.process(block)
            fx.reverb_enabled = False
            for _ in range(160):
                fx.process(block)
            self.assertGreater(render.call_count, 40)
            count = render.call_count
            for _ in range(20):
                fx.process(silence)
            self.assertEqual(render.call_count, count)
            fx.reverb_enabled = True
            for _ in range(30):
                self.assertLess(np.max(abs(fx.process(silence))), 1e-5)

    def test_every_step_at_110_bpm_is_rendered_on_its_nearest_sample(self):
        band = BackingBand(bpm=110)
        rendered = []
        with patch.object(band, "_schedule_step",
                          side_effect=lambda step, offset, *_: rendered.append(
                              (step, band.sample_counter + offset))):
            for _ in range(1200):
                band.process()
        duration = 60*SAMPLE_RATE/(110*4)
        expected = [(i, round(i*duration)) for i in range(100)
                    if round(i*duration) < band.sample_counter]
        self.assertIn((88, 576000), expected)
        self.assertEqual(rendered, expected)

    def test_live_key_change_prepares_strums_outside_render(self):
        worker = AudioWorker({"volume": .5})
        fx = VoiceEffects()
        worker.command("key_scale", ("F#", "minor"))
        with patch.object(BackingBand, "prepare_chords", side_effect=AssertionError("live synthesis")):
            worker.apply_command(fx, *worker.commands.get_nowait())
            self.assertEqual((fx.band.root, fx.scale), (6, "minor"))
            self.assertTrue(np.all(np.isfinite(fx.process(np.zeros(HOP, dtype=np.float32)))))

    def test_split_drift_and_driver_batches_keep_samples_continuous(self):
        ramp = np.arange(HOP, dtype=np.float32)
        for batch in (1, 4, 8):
            for ppm in (-900, 900):
                with self.subTest(batch=batch, ppm=ppm):
                    mic = SplitMicBuffer()
                    produced = 0
                    last = None
                    for i in range(4500):
                        due = int((i + batch) * (1 + ppm / 1e6)) // batch * batch
                        while produced < due:
                            mic.push((produced * HOP + ramp)[:, None])
                            produced += 1
                        out = mic.read()[:, 0]
                        if mic.read_position is None:
                            continue
                        if last is not None:
                            # A slipped/repeated block would break this ramp.
                            edges = np.diff(out, prepend=last)
                            self.assertGreaterEqual(edges.min(), .5)
                            self.assertLessEqual(edges.max(), 1.5)
                        last = out[-1]
                        self.assertLessEqual(mic.input_position - mic.read_position, (batch+2)*HOP)
                    self.assertIsNotNone(last)
                    self.assertEqual((mic.gaps, mic.drops), (0, 0))

    def test_split_absorbs_irregular_arrivals_from_both_devices(self):
        rng = np.random.default_rng(327)
        ticks = np.arange(6000)
        # Each callback may arrive over a hop early or late. The microphone
        # clock also gains 400 ppm. Callbacks on each device stay ordered.
        capture = np.maximum.accumulate(ticks / 1.0004 + rng.uniform(-1.15, 1.15, len(ticks)))
        playback = np.maximum.accumulate(ticks + rng.uniform(-1.15, 1.15, len(ticks)))
        mic = SplitMicBuffer()
        block = np.ones((HOP, 1), dtype=np.float32)
        produced = 0
        for due in playback:
            while produced < len(capture) and capture[produced] <= due:
                mic.push(block)
                produced += 1
            out = mic.read()
            if mic.read_position is not None:
                self.assertEqual(out[-1, 0], 1)
                self.assertLessEqual(mic.input_position - mic.read_position, 7*HOP)
        self.assertIsNotNone(mic.read_position)
        self.assertEqual((mic.gaps, mic.drops), (0, 0))

    def test_split_stalls_crossfade_then_return_to_recent_input(self):
        mic = SplitMicBuffer()
        block = np.full((HOP, 1), .25, dtype=np.float32)
        mic.push(block)
        mic.read()
        mic.push(block)
        mic.read()
        # Reusing a driver buffer cannot alter data already published.
        block.fill(-.25)
        self.assertAlmostEqual(mic.read()[-1, 0], .25)
        gap = mic.read()
        self.assertAlmostEqual(gap[0, 0], .25)
        self.assertFalse(np.any(gap[96:]))
        self.assertGreater(mic.gaps, 0)
        for _ in range(3):
            mic.push(block)
            resumed = mic.read()
        self.assertAlmostEqual(resumed[-1, 0], -.25)
        for _ in range(18):
            mic.push(-block)
        jumped = mic.read()
        self.assertGreater(mic.drops, 0)
        self.assertAlmostEqual(jumped[0, 0], -.25)
        self.assertAlmostEqual(jumped[95, 0], .25)
        self.assertLessEqual(mic.input_position - mic.read_position, 2*HOP)


if __name__ == "__main__":
    unittest.main()
