"""Deterministic tests for the offline phrase-memory song arranger."""
import math
import unittest

import numpy as np

from retro_autotune import HOP, BackingBand, VoiceEffects
from retro_autotune_gui import AudioWorker
from song_mode import SongBrain


class PhrasePlayer:
    def __init__(self, brain, bpm=110):
        self.brain = brain
        self.frames = 0
        self.last_step = -1
        self.bpm = bpm

    def block(self, note, confidence=0.99, rms=0.15, chord_degree=1):
        self.frames += self.brain.hop
        step = int(self.frames / (self.brain.sample_rate * 60 / (self.bpm * 4)))
        self.brain.observe(note, confidence if note is not None else 0,
                           rms if note is not None else 0, step)
        for s in range(self.last_step + 1, step + 1):
            self.brain.on_step(s, chord_degree)
        self.last_step = step

    def phrase(self, notes, blocks=24, chords=(1, 4, 5)):
        for i, note in enumerate(notes):
            for _ in range(blocks + i % 2):
                self.block(note, chord_degree=chords[min(len(chords)-1, i//2)])
        self.silence(45)

    def silence(self, count):
        for _ in range(count):
            self.block(None)


class SongBrainTests(unittest.TestCase):
    def setUp(self):
        self.brain = SongBrain(enabled=True)
        self.player = PhrasePlayer(self.brain)

    def test_recognizes_repeat_and_changes_sections_only_at_bar(self):
        melody = [60, 62, 64, 67, 65]
        self.player.phrase(melody)
        self.assertEqual(self.brain.phrases_heard, 1)
        self.assertEqual(self.brain.repeats_heard, 0)
        self.assertEqual(len(self.brain.motifs), 1)
        self.assertEqual(self.brain.pending_section, 'VERSE')
        self.player.silence(90)
        self.assertEqual(self.brain.section, 'VERSE')
        # The same motif up a tone with slightly modified timing remains a repeat.
        self.player.phrase([62, 64, 66, 69, 67], blocks=26)
        self.assertEqual(self.brain.phrases_heard, 2)
        self.assertEqual(self.brain.repeats_heard, 1)
        self.assertEqual(len(self.brain.motifs), 1)
        self.assertEqual(self.brain.pending_section, 'CHORUS')
        self.player.silence(230)
        self.assertEqual(self.brain.section, 'CHORUS')
        self.assertGreater(len(self.brain.motifs[0]['chords']), 0)

    def test_nonrepeating_phrase_does_not_cause_chorus(self):
        self.player.phrase([60, 62, 64, 67, 65])
        self.player.silence(90)
        self.player.phrase([60, 67, 63, 65, 61])
        self.assertEqual(self.brain.repeats_heard, 0)
        self.assertEqual(len(self.brain.motifs), 2)
        self.assertNotEqual(self.brain.pending_section, 'CHORUS')

    def test_noise_short_phrase_and_unvoiced_notes_are_ignored(self):
        for note in (60, 82, 43, 65):
            for _ in range(2):
                self.player.block(note)
        self.player.silence(50)
        self.assertEqual(self.brain.phrases_heard, 0)
        for note in (60, 62, 64, 67):
            for _ in range(20):
                self.player.block(note, confidence=0.2)
        self.player.silence(50)
        self.assertEqual(self.brain.phrases_heard, 0)

    def test_remembers_chords_by_beat_on_phrase_return(self):
        notes=[60, 62, 64, 67, 65]
        self.player.phrase(notes, chords=(1, 4, 5))
        phrase = self.brain.motifs[0]
        self.assertTrue(len(phrase['chords']) >= 2, phrase['chords'])
        # After beginning a new phrase, the 4th stable note matches the stored
        # motif prefix, allowing accompaniment to reuse prior chord positions.
        for i, note in enumerate(notes[:4]):
            for _ in range(26):
                self.player.block(note, chord_degree=0)
        self.assertIsNotNone(self.brain.current_motif)
        self.assertEqual(self.brain.current_motif['id'], phrase['id'])
        suggested = self.brain.remembered_chord(self.player.last_step)
        self.assertIn(suggested, (1, 4, 5))
        self.assertNotEqual(suggested, 0)

    def test_off_reset_and_bounded_memory(self):
        self.brain = SongBrain(enabled=True, max_motifs=2)
        self.player = PhrasePlayer(self.brain)
        for melody in ([60,62,64,67], [60,67,63,65], [65,60,67,62]):
            self.player.phrase(melody)
        self.assertEqual(self.brain.phrases_heard, 3)
        self.assertEqual(len(self.brain.motifs), 2)
        self.brain.set_enabled(False)
        self.player.phrase([60,62,64,67])
        self.assertEqual(self.brain.phrases_heard, 3)
        self.brain.reset()
        self.assertEqual(self.brain.phrases_heard, 0)
        self.assertEqual(len(self.brain.motifs), 0)

    def test_no_song_mode_sound_regression_and_live_gui_toggle(self):
        fx = VoiceEffects(song_mode=False, drums=True, bass=True, guitar=True,
                          follow_voice=True, backing_vocals='duo', preset='extreme')
        self.assertFalse(fx.band.song.enabled)
        self.assertEqual(fx.band.song.arrangement(), (1, 1, 1))
        AudioWorker.apply_command(fx, 'song_mode', True)
        self.assertTrue(fx.band.song.enabled)
        tone = (0.1*np.sin(2*np.pi*220*np.arange(HOP)/48000)).astype('float32')
        for i in range(12):
            block=fx.process(tone)
            self.assertEqual(block.shape, (HOP,2))
            self.assertTrue(np.isfinite(block).all())
        AudioWorker.apply_command(fx, 'key_scale', ('G','minor'))
        self.assertEqual(fx.band.song.phrases_heard, 0)
        AudioWorker.apply_command(fx, 'song_mode', False)
        self.assertFalse(fx.band.song.enabled)


if __name__ == '__main__':
    unittest.main()
