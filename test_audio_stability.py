"""Pure/simulated audio dropout and DSP-deadline regression tests (no sound card)."""
import unittest
from unittest.mock import patch
import numpy as np
from retro_autotune import HOP, SAMPLE_RATE, VoiceEffects, detect_pitch_yin
from retro_autotune_gui import MicGapSmoother, stream_latency, AUDIO_PROFILES


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


if __name__ == "__main__":
    unittest.main()
