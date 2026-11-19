"""Deterministic, offline musical phrase memory for RetroAutoTune.

No model, audio dependencies, GUI, or disk I/O. This class receives already-
filtered MIDI notes from the audio engine and runs on beat/frame boundaries.
It guesses *chorus-like* repeated hooks; these labels are heuristics rather
than reliable song-structure transcription.
"""
from __future__ import annotations

import math


class SongBrain:
    """Track sung phrases and bring back accompaniment used for prior motifs.

    All methods are called by a single audio processing thread. Recording a
    phrase stores only compact note/beat/chord metadata, never raw microphone
    samples. Phrase memory is bounded to avoid unbounded session growth.
    """

    def __init__(self, sample_rate=48000, hop=512, enabled=False, max_motifs=24):
        self.sample_rate = int(sample_rate)
        self.hop = int(hop)
        self.max_motifs = int(max_motifs)
        self.enabled = bool(enabled)
        self.motifs = []
        self.section = "INTRO"
        self.section_bar = 0
        self.pending_section = None
        self.phrases_heard = 0
        self.repeats_heard = 0
        self.current_motif = None
        self.start_step = 0
        self.last_voice_step = 0
        self.silence_frames = 0
        self.active_note = None
        self.active_frames = 0
        self.pending_note = None
        self.pending_frames = 0
        self.events = []
        self.chord_log = []
        self.last_completed_motif = None
        self.reset_age = 0

    def set_enabled(self, enabled):
        enabled = bool(enabled)
        if self.enabled != enabled:
            self.enabled = enabled
            self._clear_active()
            self.pending_section = None
            if not enabled:
                self.section = "INTRO"

    def reset(self):
        self.motifs.clear()
        self.phrases_heard = 0
        self.repeats_heard = 0
        self._clear_active()
        self.section = "INTRO"
        self.pending_section = None
        self.section_bar = 0
        self.last_completed_motif = None

    def _clear_active(self):
        self.current_motif = None
        self.start_step = 0
        self.silence_frames = 0
        self.active_note = None
        self.active_frames = 0
        self.pending_note = None
        self.pending_frames = 0
        self.events = []
        self.chord_log = []

    @staticmethod
    def _pattern(events):
        if not events:
            return ()
        base = events[0][0]
        return tuple(int(pitch - base) for pitch, _ in events)

    @staticmethod
    def _duration_pattern(events):
        if not events:
            return ()
        durations = [max(1, duration) for _, duration in events]
        typical = sorted(durations)[len(durations) // 2]
        return tuple(duration / typical for duration in durations)

    @classmethod
    def _score(cls, events, candidate, prefix=False, limit=math.inf):
        """Pitch-contour similarity with mild rhythmic tolerance.

        A one-semitone mistake or small timing variation should still match.
        Repeated static tones or incompatible phrase lengths must not match.
        """
        a, b = cls._pattern(events), cls._pattern(candidate)
        if len(a) < 3 or len(b) < 3:
            return 99.0
        if prefix:
            if len(a) > len(b):
                return 99.0
            b = b[:len(a)]
            candidate = candidate[:len(a)]
        elif abs(len(a) - len(b)) > 1:
            return 99.0
        if max(a) - min(a) < 2 or max(b) - min(b) < 2:
            return 99.0
        # For a one-note ornament use a small monotonic alignment penalty.
        da = [math.log2(max(.1, d)) for d in cls._duration_pattern(events)]
        db = [math.log2(max(.1, d)) for d in cls._duration_pattern(candidate)]
        m, n = len(a), len(b)
        gap = 1.25
        max_cost = limit * max(m, n)
        # Each move away from the diagonal must be paid for with gaps, including
        # the moves needed to finish at (m, n). Omit paths that cannot match.
        delta = n - m
        max_gaps = int((max_cost + 1e-12) / gap) if math.isfinite(max_cost) else m + n
        if max_gaps < abs(delta):
            return 99.0
        margin = (max_gaps - abs(delta)) // 2
        low = min(0, delta) - margin
        high = max(0, delta) + margin
        pitch_costs = (0.0, 0.0, .70, 2*.70, 3*.70, 4*.70)
        prev = [j * gap for j in range(n + 1)]
        for i in range(1, m + 1):
            curr = [i * gap] + [math.inf] * n
            for j in range(max(1, i + low), min(n, i + high) + 1):
                pitch_error = abs(a[i - 1] - b[j - 1])
                note_cost = pitch_costs[pitch_error] if pitch_error < 6 else 3.0
                timing_cost = min(1.0, abs(da[i-1] - db[j-1])) * 0.12
                curr[j] = min(prev[j - 1] + note_cost + timing_cost,
                              prev[j] + gap, curr[j - 1] + gap)
            if min(curr) > max_cost + 1e-12:
                return 99.0
            prev = curr
        return prev[n] / max(m, n)

    def _match(self, events, prefix=False):
        best, score = None, 99.0
        threshold = 0.24 if prefix else 0.35
        for motif in self.motifs:
            d = self._score(events, motif["events"], prefix=prefix, limit=min(score, threshold))
            if d < score:
                best, score = motif, d
                if score == 0.0:
                    break
        if score <= threshold:
            return best
        return None

    def _finish_note(self):
        if self.active_note is not None and self.active_frames >= int(self.sample_rate * .085):
            self.events.append((self.active_note, self.active_frames))
        self.active_note = None
        self.active_frames = 0

    def observe(self, note, confidence, rms, step):
        """One call per audio hop. ``step`` is the current 16th-note grid step."""
        if not self.enabled:
            return
        voiced = (note is not None and confidence >= 0.78 and rms >= 0.004)
        if voiced:
            pitch = int(round(note))
            self.last_voice_step = step
            self.silence_frames = 0
            if self.active_note == pitch:
                self.active_frames += self.hop
                self.pending_note = None
                self.pending_frames = 0
            else:
                if self.pending_note == pitch:
                    self.pending_frames += self.hop
                else:
                    self.pending_note = pitch
                    self.pending_frames = self.hop
                # Minimum 3 hops to commit a pitch, damping false octave blips.
                if self.pending_frames >= self.hop * 3:
                    if self.active_note is None and not self.events:
                        self.start_step = step
                    self._finish_note()
                    self.active_note = pitch
                    self.active_frames = self.pending_frames
                    self.pending_note = None
                    self.pending_frames = 0
                    # Re-evaluate with each additional note; an early prefix
                    # must not lock us onto a different tune for the whole line.
                    if len(self.events) >= 3:
                        self.current_motif = self._match(self.events, prefix=True)
        else:
            self.silence_frames += self.hop
            self.pending_note = None
            self.pending_frames = 0
            if self.active_note is not None or self.events:
                if self.silence_frames >= int(self.sample_rate * .36):
                    self._complete_phrase()
        # End long improvised lines; don't let one extended vocal note fill RAM.
        if len(self.events) >= 28:
            self._complete_phrase()

    def on_step(self, step, chord_degree):
        """Capture a compact chord timeline on beats and switch sections on bars."""
        if not self.enabled:
            return False
        changed = False
        if step % 16 == 0:
            bar = step // 16
            if self.pending_section is not None and (self.section == "INTRO" or bar - self.section_bar >= 2):
                if self.section != self.pending_section:
                    self.section = self.pending_section
                    self.section_bar = bar
                    changed = True
                self.pending_section = None
            if self.phrases_heard and self.active_note is None and not self.events and step - self.last_voice_step >= 48:
                if self.section not in ("INTRO", "BREAK") and bar - self.section_bar >= 2:
                    self.section = "BREAK"
                    self.section_bar = bar
                    changed = True
        if step % 4 == 0 and (self.active_note is not None or self.events):
            offset = (step // 4) - (self.start_step // 4)
            if 0 <= offset < 256 and (not self.chord_log or self.chord_log[-1][0] != offset):
                self.chord_log.append((offset, int(chord_degree)))
        return changed

    def remembered_chord(self, step):
        """Use past phrase's chord progression only after a confident prefix match."""
        if not self.enabled or self.current_motif is None:
            return None
        chords = self.current_motif["chords"]
        if not chords:
            return None
        offset = max(0, (step // 4) - (self.start_step // 4))
        degree = None
        for beat, chosen in chords:
            if beat > offset:
                break
            degree = chosen
        return degree

    def _complete_phrase(self):
        self._finish_note()
        events = self.events
        duration = sum(n for _, n in events)
        if 4 <= len(events) <= 28 and duration >= self.sample_rate * .55:
            match = self._match(events)
            if match is None:
                match = {"id": len(self.motifs) + self.phrases_heard + 1,
                         "events": tuple(events), "chords": tuple(self.chord_log),
                         "repeats": 0}
                if len(self.motifs) >= self.max_motifs:
                    self.motifs.pop(0)
                self.motifs.append(match)
                if self.section == "CHORUS" and self.phrases_heard >= 2:
                    self.pending_section = "VERSE"
                elif self.section in ("INTRO", "BREAK"):
                    self.pending_section = "VERSE"
            else:
                match["repeats"] += 1
                self.repeats_heard += 1
                self.pending_section = "CHORUS"
                # The first performance's chord choices are the reference;
                # only fill an empty reference with a later usable timeline.
                if not match["chords"]:
                    match["chords"] = tuple(self.chord_log)
            self.phrases_heard += 1
            self.last_completed_motif = match["id"]
        self._clear_active()

    def arrangement(self):
        """(drums, bass, guitar) section mix multipliers, softly smoothed by band."""
        if not self.enabled:
            return (1.0, 1.0, 1.0)
        return {
            "INTRO": (0.65, 0.72, 0.55),
            "VERSE": (0.88, 0.90, 0.78),
            "CHORUS": (1.12, 1.05, 1.18),
            "BREAK": (0.60, 0.65, 0.38),
        }[self.section]
