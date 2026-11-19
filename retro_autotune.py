#!/usr/bin/env python3
"""Live hard-snap Auto-Tune, choir and voice-following guitar/bass/drums.

Install: py -m pip install numpy sounddevice
List devices: py retro_autotune.py --list
Run: py retro_autotune.py --input 1 --output 5 --headphones --key C --scale minor --preset extreme --reverb cher
Self-test: py retro_autotune.py --selftest

Experimental vocal effect / synthetic backing, not acoustic echo cancellation.
"""

import argparse
from functools import lru_cache
import faulthandler
import math
import sys
import time
import traceback

import numpy as np

from song_mode import SongBrain

SAMPLE_RATE = 48000
HOP = 512
PITCH_BUFFER = 3072
YIN_WINDOW = 1536
MIN_HZ = 75.0
MAX_HZ = 850.0

KEYS = {
    "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3,
    "E": 4, "F": 5, "F#": 6, "GB": 6, "G": 7,
    "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11,
}
SCALES = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "pentatonic": (0, 2, 4, 7, 9),
    "chromatic": tuple(range(12)),
}
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def frequency_to_midi(hz):
    return 69.0 + 12.0 * math.log2(hz / 440.0)


def allowed_notes(key, scale):
    pc = KEYS[key.upper()]
    permitted = {(pc + interval) % 12 for interval in SCALES[scale]}
    return np.asarray([n for n in range(24, 109) if n % 12 in permitted], dtype=np.float64)


def nearest_note(midi, candidates):
    return float(candidates[np.argmin(np.abs(candidates - midi))])


def midi_name(note):
    n = int(round(note))
    return NOTE_NAMES[n % 12] + str(n // 12 - 1)


def detect_pitch_yin(samples, sr=SAMPLE_RATE):
    """FFT-assisted YIN, returns (Hz, confidence); Hz=0 means unvoiced."""
    max_lag = min(int(sr / MIN_HZ), len(samples) - YIN_WINDOW)
    min_lag = max(2, int(sr / MAX_HZ))
    if max_lag <= min_lag:
        return 0.0, 0.0

    segment = np.asarray(samples[-(YIN_WINDOW + max_lag):], dtype=np.float64)
    # Test the reference window's own AC energy. A loud attack in the newer
    # samples must not make preceding silence look like a voiced reference.
    segment = segment - np.mean(segment[:YIN_WINDOW])
    reference = segment[:YIN_WINDOW]
    if np.mean(reference * reference) < 1e-7:
        return 0.0, 0.0

    nfft = 1 << (len(segment) + len(reference) - 2).bit_length()
    spectrum_ref = np.fft.rfft(reference, nfft)
    spectrum_sig = np.fft.rfft(segment, nfft)
    correlation = np.fft.irfft(np.conj(spectrum_ref) * spectrum_sig, nfft)[:max_lag + 1]

    cumulative_energy = np.concatenate(([0.0], np.cumsum(segment * segment)))
    lags = np.arange(max_lag + 1)
    energy_ref = cumulative_energy[YIN_WINDOW]
    energy_shifted = cumulative_energy[YIN_WINDOW + lags] - cumulative_energy[lags]
    difference = np.maximum(energy_ref + energy_shifted - 2.0 * correlation, 0.0)

    # Cumulative mean normalized difference function.
    lag_values = np.arange(1, max_lag + 1)
    cumulative = np.cumsum(difference[1:])
    cmndf = np.ones(max_lag + 1)
    cmndf[1:] = difference[1:] * lag_values / np.maximum(cumulative, 1e-12)

    below = np.flatnonzero(cmndf[min_lag:max_lag + 1] < 0.20)
    if len(below):
        lag = int(below[0] + min_lag)
        while lag < max_lag and cmndf[lag + 1] < cmndf[lag]:
            lag += 1
    else:
        lag = int(np.argmin(cmndf[min_lag:max_lag + 1]) + min_lag)

    confidence = max(0.0, min(1.0, 1.0 - float(cmndf[lag])))
    if confidence < 0.77:
        return 0.0, confidence

    # Sub-sample parabola interpolation improves vibrato tracking.
    refined = float(lag)
    if 1 < lag < max_lag:
        prev_v, center_v, next_v = cmndf[lag - 1:lag + 2]
        denominator = prev_v - 2.0 * center_v + next_v
        if abs(denominator) > 1e-12:
            refined += float(np.clip(0.5 * (prev_v - next_v) / denominator, -0.5, 0.5))
    return sr / refined, confidence


class StreamingPitchShifter:
    """Two crossfaded grain heads with a 1–1281 sample read delay.

    A deliberately small, vocal-oriented prototype. Transient smearing and
    grain texture are expected, especially during note changes.
    """

    steps = np.arange(HOP, dtype=np.float64)

    def __init__(self, sample_rate=SAMPLE_RATE):
        # Two copies let either head read straight across the ring boundary.
        # History must cover the longest grain delay plus the block just written.
        self.capacity = 4096
        self.buffer = np.zeros(2 * self.capacity, dtype=np.float32)
        self.write_position = 0
        self.phase = 0.0
        self.base_delay = 1.0
        self.delay_sweep = 1280.0

    def _read_delay(self, positions):
        lower = positions.astype(np.intp)  # positions are always nonnegative
        fraction = positions - lower
        a = self.buffer[lower]
        b = self.buffer[lower + 1]
        return a + (b - a) * fraction

    def process(self, block, semitones, *, audible=True):
        if len(block) != HOP:
            raise ValueError("Unexpected audio block size")
        start = self.write_position
        mirror = self.capacity + start
        self.buffer[start:start + HOP] = block
        self.buffer[mirror:mirror + HOP] = block
        playback_ratio = 2.0 ** (float(semitones) / 12.0)
        phase_step = (1.0 - playback_ratio) / self.delay_sweep
        phase = self.phase
        self.phase = (phase + HOP * phase_step) % 1.0
        self.write_position = (start + HOP) % self.capacity
        if not audible:
            return  # Keep history and phase in sync for a live octave toggle.
        positions = mirror + self.steps
        phase_a = phase + phase_step * self.steps
        phase_a -= np.floor(phase_a)
        phase_b = phase_a + 0.5
        phase_b -= np.floor(phase_b)
        delay_a = self.base_delay + phase_a * self.delay_sweep
        delay_b = self.base_delay + phase_b * self.delay_sweep
        head_a = self._read_delay(positions - delay_a)
        head_b = self._read_delay(positions - delay_b)

        gain_a = 0.5 - 0.5 * np.cos(2.0 * np.pi * phase_a)
        output = gain_a * head_a + (1.0 - gain_a) * head_b
        return output.astype(np.float32)


class SimpleHighPass:
    """One-pole high pass at ~80Hz, used to suppress low frequency rumble."""

    def __init__(self, sample_rate=SAMPLE_RATE, cutoff=80.0):
        exponent = -2.0 * math.pi * cutoff / sample_rate
        self.alpha = math.exp(exponent)
        # Unroll the complementary one-pole low-pass with a weighted prefix sum.
        # Bound the exponent even for unusual cutoffs or long input arrays.
        width = min(HOP, max(1, int(32 / max(abs(exponent), 1e-12))))
        self.decay = np.exp(exponent * np.arange(width, dtype=np.float64))
        self.weights = (1.0 - self.alpha) / self.decay
        self.low = 0.0

    def process(self, audio):
        out = np.empty_like(audio)
        width = len(self.decay)
        for start in range(0, len(audio), width):
            block = audio[start:start + width]
            n = len(block)
            low = self.decay[:n] * (self.alpha * self.low +
                                     np.cumsum(block * self.weights[:n]))
            out[start:start + n] = block - low
            self.low = float(low[-1])
        return out


class BlockDelayLine:
    """Integer delay with contiguous block reads and writes."""

    def __init__(self, delay_samples, block_size):
        if delay_samples <= block_size:
            raise ValueError("Delay must exceed callback block size")
        self.delay_samples = int(delay_samples)
        self.block_size = block_size
        # Round up to whole blocks and mirror block zero at the end. The extra
        # block of history keeps the read span independent of the write span.
        self.capacity = ((self.delay_samples - 1) // block_size + 2) * block_size
        self.buffer = np.zeros(self.capacity + block_size, dtype=np.float32)
        self.position = 0

    def _read(self):
        start = (self.position - self.delay_samples) % self.capacity
        return self.buffer[start:start + self.block_size]

    def _write(self, block):
        self.buffer[self.position:self.position + self.block_size] = block
        if self.position == 0:
            self.buffer[self.capacity:] = block
        self.position = (self.position + self.block_size) % self.capacity

    def clear(self):
        self.buffer.fill(0)


class FeedbackDelay(BlockDelayLine):
    """Integer delay with stable, damped internal feedback."""

    def __init__(self, delay_samples, block_size=HOP):
        super().__init__(delay_samples, block_size)
        self.prev_sample = 0.0

    def clear(self):
        super().clear()
        self.prev_sample = 0.0

    def process(self, x, feedback=0.0, damping=0.0):
        if len(x) != self.block_size:
            raise ValueError("Unexpected audio block size")
        if not 0.0 <= feedback < 1.0:
            raise ValueError("Unstable feedback value")
        delayed = self._read().copy()
        if damping:
            # FIR low-pass on the recirculated signal prevents fizzy resonance.
            previous = np.empty_like(delayed)
            previous[0] = self.prev_sample
            previous[1:] = delayed[:-1]
            filtered = (1.0 - damping) * delayed + damping * previous
        else:
            filtered = delayed
        self.prev_sample = float(delayed[-1])
        self._write(x + feedback * filtered if feedback else x)
        return delayed


class AllPassDiffuser(BlockDelayLine):
    """Schroeder all-pass stage for a smooth synthetic reverb field."""

    def __init__(self, delay_samples, gain=0.52):
        super().__init__(delay_samples, HOP)
        self.gain = float(gain)

    def process(self, signal):
        output = self._read() - self.gain * signal
        self._write(signal + self.gain * output)
        return output


class CherStyleReverb:
    """Wide synthetic pop-vocal plate with predelay and two slap echoes.

    'Cher' describes an inspired preset, not a recreation of any recording.
    All internal feedback factors are strictly < 1 to prevent DSP runaway.
    Reverb processing runs after gating so tails are not chopped off.
    """

    def __init__(self, sample_rate=SAMPLE_RATE):
        self.predelay = FeedbackDelay(round(sample_rate * 0.046))
        # Prime-ish delay lengths and slightly different L/R timing give width.
        left_lengths = (1217, 1319, 1429, 1543, 1667)
        right_lengths = (1259, 1361, 1481, 1597, 1721)
        self.combs = [
            [FeedbackDelay(n) for n in left_lengths],
            [FeedbackDelay(n) for n in right_lengths],
        ]
        self.diffusers = [
            [AllPassDiffuser(691), AllPassDiffuser(907)],
            [AllPassDiffuser(733), AllPassDiffuser(977)],
        ]
        self.slaps = [
            FeedbackDelay(round(sample_rate * 0.108)),
            FeedbackDelay(round(sample_rate * 0.137)),
        ]

    def clear(self):
        self.predelay.clear()
        for stages in (*self.combs, *self.diffusers, self.slaps):
            for stage in stages:
                stage.clear()

    def process(self, voice):
        predelayed = self.predelay.process(voice)
        result = np.empty((len(voice), 2), dtype=np.float32)
        for channel in range(2):
            comb_mix = np.zeros_like(voice)
            for delay in self.combs[channel]:
                comb_mix += delay.process(predelayed, feedback=0.845, damping=0.42)
            comb_mix *= 1.0 / len(self.combs[channel])
            for stage in self.diffusers[channel]:
                comb_mix = stage.process(comb_mix)
            slap = self.slaps[channel].process(voice, feedback=0.22, damping=0.3)
            result[:, channel] = 0.85 * comb_mix + 0.15 * slap
        return result


class SampleVoiceMixer:
    """Mix short pre-rendered one-shot synth samples, with tails across blocks.

    No sound files and no sample synthesis on the realtime callback thread.
    """

    def __init__(self):
        self.active = []  # (sample array, next sample index)

    def render(self, scheduled):
        """scheduled: iterable of (sample_array, offset_inside_block)."""
        output = np.zeros(HOP, dtype=np.float32)
        survivors = []
        for sample, position in self.active:
            count = min(HOP, len(sample) - position)
            if count > 0:
                output[:count] += sample[position:position + count]
            if position + count < len(sample):
                survivors.append((sample, position + count))
        for sample, offset in scheduled:
            count = min(HOP - offset, len(sample))
            if count > 0:
                output[offset:offset + count] += sample[:count]
            if count < len(sample):
                survivors.append((sample, count))
        self.active = survivors
        return output


def make_drum_samples(sr=SAMPLE_RATE):
    """Small original synthesized kick, snare and closed hat samples."""
    rng = np.random.default_rng(2008)
    tk = np.arange(round(sr * 0.28), dtype=np.float32) / sr
    # Falling kick oscillator, with a fast click to emphasize each downbeat.
    ph = 2.0 * np.pi * (51.0 * tk + 116.0 * (1.0 - np.exp(-36 * tk)) / 36)
    kick = np.sin(ph) * np.exp(-tk * 16) * (1 - np.exp(-tk * 600))
    kick += 0.15 * rng.standard_normal(len(tk)) * np.exp(-tk * 135)

    ts = np.arange(round(sr * 0.22), dtype=np.float32) / sr
    sn = rng.standard_normal(len(ts))
    sn = sn - np.convolve(sn, np.ones(13, dtype=np.float32) / 13, mode="same")
    snare = (0.55 * sn + 0.35 * np.sin(2 * np.pi * 186 * ts)) * np.exp(-ts * 24)

    th = np.arange(round(sr * 0.085), dtype=np.float32) / sr
    hn = rng.standard_normal(len(th))
    hn = hn - np.convolve(hn, np.ones(19, dtype=np.float32) / 19, mode="same")
    hat = hn * np.exp(-th * 75) * 0.35
    return {"kick": kick.astype(np.float32),
            "snare": snare.astype(np.float32),
            "hat": hat.astype(np.float32)}


def make_bass_sample(midi, sr=SAMPLE_RATE):
    """Short analog-synth-like electric bass pluck with click-free envelope."""
    t = np.arange(round(sr * 0.51), dtype=np.float32) / sr
    hz = 440 * 2 ** ((midi - 69) / 12)
    x = np.sin(2 * np.pi * hz * t)
    x += 0.39 * np.sin(4 * np.pi * hz * t)
    x += 0.12 * np.sin(6 * np.pi * hz * t)
    attack = 1 - np.exp(-t * 230)
    decay = np.exp(-t * 5.8)
    tail = np.clip((0.51 - t) / 0.12, 0, 1)
    return (0.78 * np.tanh(x * 1.18) * attack * decay * tail).astype(np.float32)


@lru_cache(maxsize=96)
def make_guitar_sample(midi, sr=SAMPLE_RATE):
    """Original, CPU-light plucked-string synth (not a guitar recording)."""
    n = int(sr * 0.62)
    t = np.arange(n, dtype=np.float64) / sr
    f = 440.0 * (2.0 ** ((float(midi) - 69) / 12.0))
    rng = np.random.default_rng(7000 + int(midi))
    phase = 2.0 * np.pi * f * t
    # Harmonics decay at different rates, mimicking plucked-string brightness.
    x = (0.76 * np.sin(phase) * np.exp(-3.6*t) +
         0.27 * np.sin(2*phase + .17) * np.exp(-8*t) +
         0.17 * np.sin(3*phase + .45) * np.exp(-13*t) +
         0.08 * np.sin(4*phase + .6) * np.exp(-24*t))
    pick = rng.standard_normal(n) * np.exp(-t*180.0) * .08
    attack = (1.0 - np.exp(-t*380.0))
    tail = np.clip((.62-t)/.08, 0.0, 1.0)
    return (0.38 * np.tanh(x + pick) * attack * tail).astype(np.float32)


class BackingBand:
    """Steady tempo, vocal-responsive drums, harmonic bass and plucked guitar.

    When follow_voice is enabled, the singer's recent pitch-class histogram is
    evaluated on tempo subdivisions chosen by follow_strength. Notes are
    weighted towards diatonic chords; silence retains the previous chord.
    Even the fastest mode changes on the beat grid, never per syllable.
    """

    def __init__(self, key="C", scale="minor", bpm=110, groove="pop",
                 drums_enabled=False, bass_enabled=False, guitar_enabled=False,
                 follow_voice=False, follow_strength=0.50, song_mode=False, drum_level=0.42, bass_level=0.42,
                 guitar_level=0.42, band_level=0.75):
        self.bpm = int(bpm)
        self.groove = groove
        self.drums_enabled = bool(drums_enabled)
        self.bass_enabled = bool(bass_enabled)
        self.guitar_enabled = bool(guitar_enabled)
        self.follow_voice = bool(follow_voice)
        self.follow_strength = float(np.clip(follow_strength, 0.0, 1.0))
        self.song = SongBrain(sample_rate=SAMPLE_RATE, hop=HOP, enabled=song_mode)
        self.song_mix = np.ones(3, dtype=np.float32)
        self.drum_level = float(drum_level)
        self.bass_level = float(bass_level)
        self.guitar_level = float(guitar_level)
        self.band_level = float(band_level)
        self.sample_counter = 0
        self.next_step_sample = 0.0
        self.step_index = 0
        self.steps_per_sample = 1 / (SAMPLE_RATE * 60.0 / (self.bpm * 4.0))
        self.drum_samples = make_drum_samples()
        self.bass_samples = {m: make_bass_sample(m) for m in range(30, 60)}
        self.drums, self.bass, self.guitar = SampleVoiceMixer(), SampleVoiceMixer(), SampleVoiceMixer()
        self.drum_fade = float(self.drums_enabled)
        self.bass_fade = float(self.bass_enabled)
        self.guitar_fade = float(self.guitar_enabled)
        self.voice_energy = 0.0
        self.voice_onset = 0.0
        self.pitch_evidence = np.zeros(12, dtype=np.float64)
        self.voiced_updates = 0
        self.chord_degree = 0
        self.chord_name = "--"
        self.last_chord_step = -16
        self.set_key_scale(key, scale)

    @staticmethod
    def prepare_chords(key, scale):
        """Build the guitar strums before a live key change reaches the renderer."""
        root = KEYS[key.upper()]
        # Pentatonic/chromatic melodies can still use standard diatonic triads.
        intervals = SCALES[scale] if len(SCALES[scale]) == 7 else SCALES["major"]
        triads = []
        guitar_samples = {}
        for degree in range(len(intervals)):
            semitones = tuple(int(intervals[(degree + offset) % 7] +
                                  12 * ((degree + offset) // 7))
                              for offset in (0, 2, 4))
            # A C4-centred guitar voicing is good for the synth's finite range.
            chord = [48 + root + x for x in semitones]
            if chord[0] > 60:
                chord = [x - 12 for x in chord]
            pcs = tuple(p % 12 for p in chord)
            triads.append(pcs)
            notes = [make_guitar_sample(p) for p in chord]
            for direction in ("down", "up"):
                ordered = notes if direction == "down" else list(reversed(notes))
                strum = np.zeros(max(map(len, notes)) + 1250, dtype=np.float32)
                for idx, note in enumerate(ordered):
                    offset = idx * 520  # Eleven milliseconds between strings
                    strum[offset:offset + len(note)] += note * 0.72
                guitar_samples[(degree, direction)] = strum
        return intervals, triads, guitar_samples

    def set_key_scale(self, key, scale, prepared=None):
        self.root = KEYS[key.upper()]
        self.scale = scale
        self.chords = (0, 8, 3, 10) if scale == "minor" else (0, 7, 9, 5)
        if prepared is None:
            prepared = self.prepare_chords(key, scale)
        self.degree_intervals, self.triads, self.guitar_samples = prepared
        self.pitch_evidence.fill(0.0)
        self.voiced_updates = 0
        self.song.reset()  # A changed key/scale invalidates remembered chord degrees.
        self.chord_degree = 0
        self.last_chord_step = -16
        self.chord_name = NOTE_NAMES[(self.root + self.degree_intervals[0]) % 12]
        # Avoid leaving old-key chord samples ringing on a scale/key change.
        self.guitar.active.clear()
        self.bass.active.clear()

    def set_bpm(self, bpm):
        bpm = int(np.clip(bpm, 60, 180))
        if bpm != self.bpm:
            remaining = max(0.0, self.next_step_sample - self.sample_counter)
            self.next_step_sample = self.sample_counter + remaining * self.bpm / bpm
            self.bpm = bpm
            self.steps_per_sample = 1 / (SAMPLE_RATE * 60.0 / (bpm * 4.0))

    def feed_voice(self, note=None, confidence=0.0, rms=0.0):
        """Meter + corrected melody note. Never infer a chord from unvoiced noise."""
        voiced = note is not None and confidence >= 0.78 and rms >= 0.004
        energy = min(1.0, float(rms) * 7.0) if voiced else 0.0
        old = self.voice_energy
        self.voice_energy += (0.16 if energy > old else 0.045) * (energy - old)
        self.voice_onset = max(self.voice_onset * 0.91, max(0.0, energy - old))
        self.song.observe(note if voiced else None, confidence, rms, self.step_index)
        if self.follow_voice:
            # Higher follow strength emphasizes recent sung notes more strongly.
            self.pitch_evidence *= 0.997 - 0.010 * self.follow_strength
            if voiced:
                self.pitch_evidence[int(round(note)) % 12] += (0.2 + energy)
                self.voiced_updates += 1

    def _degree_from_root_interval(self, semitones):
        target = (self.root + semitones) % 12
        return min(range(7), key=lambda i: min((self.root + self.degree_intervals[i] - target) % 12,
                                               (target - self.root - self.degree_intervals[i]) % 12))

    def _choose_adaptive_chord(self):
        strength = self.follow_strength
        minimum_updates = 10 if strength < 0.66 else (6 if strength < 0.9 else 4)
        if self.voiced_updates < minimum_updates or self.pitch_evidence.max() < 0.45:
            return  # Hold through rests and avoid following a stray syllable.
        evidence = self.pitch_evidence
        scores = []
        for degree, chord in enumerate(self.triads):
            # Chord tones explain the melody. Reward root gently; discourage
            # arbitrary chord changes on ambiguous common tones.
            value = sum(evidence[pc] for pc in chord)
            value += 0.13 * evidence[chord[0]]
            if degree == self.chord_degree:
                # Strong follows make it easier for new melody notes to move
                # the harmony, but still prevent flutter around chord tones.
                value += (0.20 - 0.14 * strength) * float(np.sum(evidence))
            scores.append(value)
        chosen = int(np.argmax(scores))
        self.chord_degree = chosen
        self.chord_name = NOTE_NAMES[(self.root + self.degree_intervals[chosen]) % 12]
        self.pitch_evidence *= 0.22 if strength >= 0.66 else 0.35
        self.voiced_updates = 0

    def follow_step_interval(self):
        """Chord reevaluation cadence in sixteenth notes (tempo stays fixed)."""
        if self.follow_strength >= 0.90:
            return 4   # quarter note; strong, beat-quantized follow
        if self.follow_strength >= 0.65:
            return 8   # every two beats
        if self.follow_strength >= 0.30:
            return 16  # each bar, as in v4.3
        return 32      # every two bars, laid-back support

    def bass_midi(self, bar_index, bar_step):
        if self.follow_voice:
            interval = self.degree_intervals[self.chord_degree]
        else:
            interval = self.chords[bar_index % len(self.chords)]
        root = 36 + self.root + interval
        while root > 47:
            root -= 12
        while root < 36:
            root += 12
        fifth = root + 7 if root < 41 else root - 5
        return fifth if bar_step in (6, 12) else root

    def _schedule_step(self, step, offset, drum_events, bass_events, guitar_events):
        pos, bar = step % 16, step // 16
        if self.follow_voice:
            if step % self.follow_step_interval() == 0:
                # When a remembered melodic motif returns, prefer the chord
                # timeline from its first appearance over independent changes.
                remembered = self.song.remembered_chord(step)
                if remembered is None:
                    self._choose_adaptive_chord()
                else:
                    self.chord_degree = remembered
                    self.chord_name = NOTE_NAMES[(self.root + self.degree_intervals[remembered]) % 12]
                    self.pitch_evidence *= 0.30
                    self.voiced_updates = 0
        elif pos == 0:
            self.chord_degree = self._degree_from_root_interval(self.chords[bar % len(self.chords)])
            self.chord_name = NOTE_NAMES[(self.root + self.degree_intervals[self.chord_degree]) % 12]
        self.song.on_step(step, self.chord_degree)
        if self.groove == "dance":
            kick_steps, snare_steps = (0, 4, 8, 12), (4, 12)
        elif self.groove == "half-time":
            kick_steps, snare_steps = (0, 6, 11), (8,)
        else:
            kick_steps, snare_steps = (0, 8, 14), (4, 12)
        if self.drums_enabled:
            if pos in kick_steps:
                drum_events.append((self.drum_samples["kick"], offset))
            if pos in snare_steps:
                drum_events.append((self.drum_samples["snare"], offset))
            # A small pickup at section changes, without modifying tempo.
            if self.song.enabled and pos == 14 and self.song.pending_section == "CHORUS":
                drum_events.append((self.drum_samples["snare"], offset))
            if pos % 2 == 0 or (self.follow_voice and self.voice_onset > 0.22 and pos % 4 == 0):
                drum_events.append((self.drum_samples["hat"], offset))
        if self.bass_enabled and pos in (0, 6, 8, 12):
            bass_events.append((self.bass_samples[self.bass_midi(bar, pos)], offset))
        if self.guitar_enabled and pos in (0, 4, 8, 12):
            direction = "up" if pos in (4, 12) else "down"
            # Off-beats are naturally quieter, retaining the song's pulse.
            guitar_events.append((self.guitar_samples[(self.chord_degree, direction)], offset))

    def process(self):
        start, stop = self.sample_counter, self.sample_counter + HOP
        drum_events, bass_events, guitar_events = [], [], []
        step_duration = 1.0 / self.steps_per_sample
        # Decide block ownership on the same integer sample grid as playback.
        # A step rounded to 'stop' belongs at offset zero of the next block.
        while round(self.next_step_sample) < stop:
            offset = round(self.next_step_sample) - start
            self._schedule_step(self.step_index, offset, drum_events, bass_events, guitar_events)
            self.step_index += 1
            self.next_step_sample += step_duration
        self.sample_counter = stop
        drums = self.drums.render(drum_events)
        bass = self.bass.render(bass_events)
        guitar = self.guitar.render(guitar_events)
        self.drum_fade += 0.18 * (float(self.drums_enabled) - self.drum_fade)
        self.bass_fade += 0.18 * (float(self.bass_enabled) - self.bass_fade)
        self.guitar_fade += 0.18 * (float(self.guitar_enabled) - self.guitar_fade)
        # Dynamics follow the singer, while tempo stays steady. This ensures
        # soft phrases do not trigger jittery tempo/chord changes.
        dynamics = self.voice_energy * (0.45 + 1.35 * self.follow_strength) if self.follow_voice else 0.0
        dynamics = min(1.0, dynamics)
        drum_gain = 0.77 + 0.26 * dynamics
        bass_gain = 0.82 + 0.25 * dynamics
        guitar_gain = 0.75 + 0.36 * dynamics
        goals = np.asarray(self.song.arrangement(), dtype=np.float32)
        self.song_mix += 0.035 * (goals - self.song_mix)
        return self.band_level * (drums * self.drum_level * self.drum_fade * drum_gain * self.song_mix[0] +
                                  bass * self.bass_level * self.bass_fade * bass_gain * self.song_mix[1] +
                                  guitar * self.guitar_level * self.guitar_fade * guitar_gain * self.song_mix[2])


# Each entry describes a synthetic backing singer: diatonic position, pan,
# relative level, and fixed delay.  Delays are staggered so the larger presets
# do not sound like eight identical voices starting at the same instant.
HARMONY_VOICES = (
    ("third-up", 2, -0.55, 0.34, 672, 0.00),
    ("fifth-up", 4, +0.55, 0.30, 944, +0.035),
    ("third-down", -2, +0.25, 0.27, 1152, -0.045),
    ("unison-left", 0, -0.82, 0.29, 1392, -0.13),
    ("fifth-down", -4, -0.24, 0.24, 1600, +0.030),
    ("octave-up", 7, +0.80, 0.19, 1888, +0.050),
    ("octave-down", -7, -0.68, 0.17, 2128, -0.035),
    ("unison-right", 0, +0.88, 0.29, 2384, +0.12),
)
HARMONY_COUNTS = {"off": 0, "duo": 1, "choir": 4, "wall": 8}


class VoiceEffects:
    """Shared realtime audio path, usable by the live callback and self-tests."""

    def __init__(self, key="C", scale="minor", preset="classic", reverb="off",
                 reverb_mix=0.38, octave_mix=None, gate_db=-48, gain=1.0,
                 retune=1.0, mono_output=False, backing_vocals="off",
                 harmony_level=0.55, drums=False, bass=False, guitar=False, follow_voice=False, bpm=110,
                 groove="pop", band_level=0.75, drum_level=0.42, bass_level=0.42,
                 guitar_level=0.42, voice_octave=0, follow_strength=0.50, song_mode=False):
        if voice_octave not in (-1, 0, 1):
            raise ValueError("voice_octave must be -1, 0, or +1")
        self.voice_octave = int(voice_octave)
        self.candidates = allowed_notes(key, scale)
        self.scale = scale
        self.pitch_shifter = StreamingPitchShifter()
        self.octave_shifter = StreamingPitchShifter()
        if backing_vocals not in HARMONY_COUNTS:
            raise ValueError("Unknown backing vocal mode")
        self.harmony_shifters = [StreamingPitchShifter() for _ in HARMONY_VOICES]
        self.harmony_delays = [FeedbackDelay(config[4]) for config in HARMONY_VOICES]
        self.harmony_pans = [(math.sqrt(2) * math.cos((v[2] + 1) * math.pi / 4),
                              math.sqrt(2) * math.sin((v[2] + 1) * math.pi / 4))
                             for v in HARMONY_VOICES]
        self.harmony_shifts = [0.0] * len(HARMONY_VOICES)
        self.harmony_envelopes = np.zeros(len(HARMONY_VOICES), dtype=np.float32)
        self.harmony_mode = backing_vocals
        self.harmony_level = float(harmony_level)
        self.band = BackingBand(key=key, scale=scale, bpm=bpm, groove=groove,
                                drums_enabled=drums, bass_enabled=bass, guitar_enabled=guitar,
                                follow_voice=follow_voice, follow_strength=follow_strength, song_mode=song_mode, band_level=band_level,
                                drum_level=drum_level, bass_level=bass_level,
                                guitar_level=guitar_level)
        self.music_duck = 1.0
        self.reverb_engine = CherStyleReverb()
        self.high_pass = SimpleHighPass()
        self.detector_history = np.zeros(PITCH_BUFFER, dtype=np.float32)
        self.gate_threshold = 10 ** (gate_db / 20)
        self.gate_level = 0.0
        self.current_target = None
        self.shift = 0.0
        self.last_note = "--"
        self.last_confidence = 0.0
        # Keep normal fast note tracking, but the audio callback may temporarily
        # set this to 2 or 3 if the machine cannot meet its realtime deadline.
        # This saves FFT/YIN work before sacrificing any choir/instrument layers.
        self.pitch_detection_stride = 1
        self._pitch_detection_counter = 0
        self.preset = preset
        self.reverb_enabled = reverb == "cher"
        self.reverb_mix = float(reverb_mix)
        self.wet_current = 0.0
        self.octave_mix_override = octave_mix
        self.gain = float(gain)
        self.retune = float(retune)
        self.mono_output = bool(mono_output)

    @property
    def octave_mix(self):
        return self.octave_mix_override if self.octave_mix_override is not None else (0.25 if self.preset == "extreme" else 0.0)

    def process(self, input_samples):
        if len(input_samples) != HOP:
            raise ValueError("Unexpected audio block size")
        samples = np.asarray(input_samples, dtype=np.float32)
        self.detector_history[:-HOP] = self.detector_history[HOP:]
        self.detector_history[-HOP:] = samples
        rms = float(np.sqrt(np.mean(samples * samples)))
        gate_goal = 1.0 if rms >= self.gate_threshold else 0.0
        self.gate_level += (0.35 if gate_goal > self.gate_level else 0.10) * (gate_goal - self.gate_level)
        self._pitch_detection_counter += 1
        if gate_goal and self._pitch_detection_counter % self.pitch_detection_stride == 0:
            hz, confidence = detect_pitch_yin(self.detector_history)
            self.last_confidence = confidence
            if hz > 0.0:
                midi = frequency_to_midi(hz)
                nearest = nearest_note(midi, self.candidates)
                # In extreme mode pitch-center changes immediately at note boundaries.
                hysteresis = 0.0 if self.preset == "extreme" else 0.13
                if (self.current_target is None or
                        abs(midi - nearest) + hysteresis < abs(midi - self.current_target)):
                    self.current_target = nearest
                max_shift = 12.0 if self.preset == "extreme" else 7.0
                wanted = float(np.clip(self.current_target - midi, -max_shift, max_shift))
                # Extreme mode disregards a slow retune setting: the jumps are part of the effect.
                speed = 1.0 if self.preset == "extreme" else self.retune
                self.shift += speed * (wanted - self.shift)
                self.last_note = midi_name(self.current_target)
                note_pos = int(np.argmin(np.abs(self.candidates - self.current_target)))
                # Harmonies stay in the selected scale, even when singing a note
                # between two allowed notes.  Octave voices span 12 semitones.
                chromatic = len(SCALES[self.scale]) == 12
                positions = (4, 7, -3, 0, -7, 12, -12, 0) if chromatic else (2, 4, -2, 0, -4, 7, -7, 0)
                for i, offset in enumerate(positions):
                    pos = max(0, min(note_pos + offset, len(self.candidates) - 1))
                    detune = HARMONY_VOICES[i][5]
                    self.harmony_shifts[i] = float(self.candidates[pos] - midi + detune)

        self.band.feed_voice(self.current_target if gate_goal and self.last_confidence > 0.78 else None,
                             self.last_confidence if gate_goal else 0.0, rms)

        # Gate only the INPUT to DSP. Reverb receives processed input after the
        # gate, but its long tail is allowed to ring out without being gated.
        filtered = self.high_pass.process(samples) * self.gate_level
        # Transpose the entire corrected vocal, independently of the optional
        # robot octave *layer*. Harmony voices track the lead octave as well.
        octave_semitones = self.voice_octave * 12.0
        tuned = self.pitch_shifter.process(filtered, self.shift + octave_semitones)
        mix = self.octave_mix
        if mix > 0:
            # Upper octave creates an intentionally exaggerated synth-vocal edge.
            upper = self.octave_shifter.process(filtered, self.shift + octave_semitones + 12.0)
            tuned = (tuned * (1.0 - mix) + upper * mix).astype(np.float32)
        else:
            # Keep the octave engine's input history warm for runtime E toggles.
            self.octave_shifter.process(filtered, self.shift + octave_semitones + 12.0, audible=False)

        drive = 2.25 if self.preset == "extreme" else 1.35
        voice = (0.74 * np.tanh(tuned * (self.gain * drive))).astype(np.float32)
        # A chorus is a *separate stereo bus*: different pitches, small detunes,
        # staggered entrances and pan positions. Mono headphone output folds it
        # down naturally. Per-singer fades avoid clicks when changing modes.
        count = HARMONY_COUNTS.get(self.harmony_mode, 0)
        choir = np.zeros((HOP, 2), dtype=np.float32)
        level = float(np.clip(self.harmony_level, 0.0, 1.5))
        for i, (name, offset, pan, weight, delay, detune) in enumerate(HARMONY_VOICES):
            goal = level if i < count else 0.0
            old_env = float(self.harmony_envelopes[i])
            new_env = old_env + 0.18 * (goal - old_env)
            self.harmony_envelopes[i] = new_env
            if old_env < 0.001 and new_env < 0.001:
                continue  # No CPU cost for inactive voices.
            if old_env < 0.001 and goal > 0.0:
                # Reset stale buffers on activation; prevents old audio ghosts.
                self.harmony_shifters[i] = StreamingPitchShifter()
                self.harmony_delays[i] = FeedbackDelay(delay)
            shifted = self.harmony_shifters[i].process(
                filtered, self.harmony_shifts[i] + octave_semitones)
            shifted = self.harmony_delays[i].process(shifted)
            singer = np.tanh(shifted * (self.gain * 1.30)).astype(np.float32)
            # Convert equal-power pan to unity at the centre.
            left, right = self.harmony_pans[i]
            amount = new_env * weight
            choir[:, 0] += singer * amount * left
            choir[:, 1] += singer * amount * right
        # Gentle choir-bus saturation lets big harmonies sound full without
        # making the tuned lead disappear. Both mono/stereo use this same bus.
        choir = (0.78 * np.tanh(choir * 1.15)).astype(np.float32)
        dry = voice[:, None] + choir
        wet_goal = self.reverb_mix if self.reverb_enabled else 0.0
        self.wet_current += 0.09 * (wet_goal - self.wet_current)
        if wet_goal > 0.0 or self.wet_current > 1e-5:
            wet = self.reverb_engine.process((dry[:, 0] + dry[:, 1]) * 0.5)
            final = dry * (1.0 - 0.30 * self.wet_current) + wet * self.wet_current
        else:
            if self.wet_current:
                self.reverb_engine.clear()  # A later restart begins with fresh audio.
                self.wet_current = 0.0
            final = dry
        # The backing band runs on its own clock; microphone gate never stops it.
        music = self.band.process()
        # Light, smooth ducking under sung phrases keeps vocals intelligible.
        duck_goal = 0.70 if self.gate_level > 0.15 else 1.0
        self.music_duck += 0.08 * (duck_goal - self.music_duck)
        final = final + (music * self.music_duck)[:, None]
        # Shared output soft limiter safely bounds voice + band + reverb.
        final = (0.87 * np.tanh(final * 1.06)).astype(np.float32)
        if self.mono_output:
            return (final[:, :1] + final[:, 1:]) * 0.5
        return final


def self_test():
    candidates = allowed_notes("C", "major")
    assert nearest_note(60.9, candidates) == 60
    assert nearest_note(61.8, candidates) == 62
    for hz in (100.0, 110.0, 150.0, 220.0, 440.0, 660.0):
        t = np.arange(PITCH_BUFFER) / SAMPLE_RATE
        signal = (0.7 * np.sin(2 * np.pi * hz * t) +
                  0.2 * np.sin(4 * np.pi * hz * t)).astype(np.float32)
        measured, confidence = detect_pitch_yin(signal)
        error = abs(12 * math.log2(measured / hz)) if measured > 0 else 999
        assert error < 0.25, (hz, measured, confidence, error)
        print(f"{hz:6.1f} Hz -> {measured:7.2f} Hz; conf={confidence:.2f}; error={error:.3f} semitones")
    pitch_shifter = StreamingPitchShifter()
    hz = 220.0
    duration_samples = HOP * 250
    signal = 0.7 * np.sin(2 * np.pi * hz * np.arange(duration_samples) / SAMPLE_RATE)
    processed = np.concatenate([pitch_shifter.process(signal[i:i+HOP].astype(np.float32), 2.0)
                                for i in range(0, duration_samples, HOP)])
    middle = processed[-SAMPLE_RATE:]
    # 0.1Hz FFT resolution; ignore edge effects via Hann taper.
    spectrum = np.abs(np.fft.rfft(middle * np.hanning(len(middle)), n=SAMPLE_RATE * 10))
    freqs = np.fft.rfftfreq(SAMPLE_RATE * 10, d=1.0 / SAMPLE_RATE)
    search = (freqs > 215) & (freqs < 280)
    dominant = float(freqs[search][np.argmax(spectrum[search])])
    expected = hz * 2 ** (2 / 12)
    assert abs(dominant - expected) < 1.0, (dominant, expected)
    assert float(np.max(np.abs(processed))) > 0.1
    print(f"Streaming pitch shifter: +2 semitones, peak={dominant:.1f} Hz, expected={expected:.1f} Hz")
    print("Pitch detector, scale snapping and shifter passed self-tests.")

    # Reverb: impulse energy after the input has stopped, decays without
    # feedback runaway, and has a meaningful stereo difference.
    reverb = CherStyleReverb()
    impulse = np.zeros(HOP, dtype=np.float32)
    impulse[0] = 0.8
    tail = [reverb.process(impulse)]
    tail.extend(reverb.process(np.zeros(HOP, dtype=np.float32)) for _ in range(250))
    reverb_audio = np.concatenate(tail)
    assert np.all(np.isfinite(reverb_audio))
    assert np.max(np.abs(reverb_audio)) < 1.0
    assert np.max(np.abs(reverb_audio[20*HOP:])) > 0.00001, "Reverb tail missing"
    assert np.mean(reverb_audio[-10*HOP:] ** 2) < np.mean(reverb_audio[30*HOP:50*HOP] ** 2)
    assert np.max(np.abs(reverb_audio[:, 0] - reverb_audio[:, 1])) > 0.001
    print("Stereo plate/reverb decay passed.")

    # Full effect path: no microphone feedback path, correct channel count,
    # bounded samples, and audible reverberation after input gate closes.
    for preset in ("classic", "extreme"):
        effects = VoiceEffects(preset=preset, reverb="cher", gate_db=-56.0)
        stream = []
        for i in range(145):
            t = (np.arange(HOP) + i * HOP) / SAMPLE_RATE
            x = (0.25 * np.sin(2 * np.pi * 223.0 * t)).astype(np.float32) if i < 35 else np.zeros(HOP, dtype=np.float32)
            stream.append(effects.process(x))
        result = np.concatenate(stream)
        assert result.shape == (HOP * 145, 2)
        assert np.all(np.isfinite(result)) and np.max(np.abs(result)) <= 0.87
        assert np.max(np.abs(result[85*HOP:95*HOP])) > 0.00001, "Effect tail incorrectly gated"
        print(f"End-to-end {preset} stereo test passed (tail and peak limiting).")
    assert VoiceEffects(mono_output=True).process(np.zeros(HOP, dtype=np.float32)).shape == (HOP, 1)

    # Music must continue without any mic signal; muted options must be silent.
    backing = BackingBand(key="C", scale="minor", bpm=120, groove="pop",
                         drums_enabled=True, bass_enabled=True)
    audio = np.concatenate([backing.process() for _ in range(375)])
    assert audio.shape == (HOP * 375,) and np.all(np.isfinite(audio))
    assert np.max(np.abs(audio)) > 0.04
    assert backing.step_index == 32, backing.step_index  # four bars in four seconds
    assert backing.bass_midi(0, 0) == 36  # C minor root
    assert backing.bass_midi(1, 0) == 44  # Ab chord
    backing.drums_enabled = False
    backing.bass_enabled = False
    for _ in range(80):
        muted = backing.process()
    assert np.max(np.abs(muted)) < 0.00001, "Backing instruments did not mute"
    assert np.max(np.abs(BackingBand().process())) == 0.0
    print("Tempo-locked pop drum/bass band and live muting passed.")

    # Check different keys and live harmony detection in a real processing path.
    for mode in ("duo", "choir", "wall"):
        effects = VoiceEffects(key="C", scale="major", backing_vocals=mode,
                               drums=True, bass=True, bpm=120, preset="extreme",
                               reverb="cher", mono_output=True)
        for i in range(60):
            t = (i * HOP + np.arange(HOP)) / SAMPLE_RATE
            result = effects.process((0.25 * np.sin(2 * np.pi * 220 * t)).astype(np.float32))
        assert result.shape == (HOP, 1) and np.all(np.isfinite(result))
        assert np.max(np.abs(result)) <= 0.87
        assert effects.harmony_shifts[0] > effects.shift  # harmony above lead
        assert effects.band.step_index > 0
    print("Duet, 4-singer choir and 8-singer wall + band/reverb tests passed.")

    # Lead octave must be a real transposition (not merely mixing in an
    # octave-high backing layer). Verify fundamental shifts in both directions.
    for octave, expected in ((-1, 110.0), (0, 220.0), (1, 440.0)):
        effects = VoiceEffects(key="A", scale="chromatic", voice_octave=octave,
                               octave_mix=0.0, gate_db=-70, gain=0.35,
                               backing_vocals="off", reverb="off")
        rendered = []
        for i in range(200):
            t = (i * HOP + np.arange(HOP)) / SAMPLE_RATE
            block = (0.30 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
            rendered.append(effects.process(block)[:, 0])
        audio = np.concatenate(rendered)[-SAMPLE_RATE:]
        spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / SAMPLE_RATE)
        region = (freqs > expected * 0.8) & (freqs < expected * 1.2)
        peak = float(freqs[region][np.argmax(spectrum[region])])
        # Grain-based real-time transposition is intentionally approximate;
        # the goal here is a clearly audible, correctly directed octave.
        error_semitones = abs(12 * math.log2(peak / expected))
        assert error_semitones < 0.5, (octave, peak, expected, error_semitones)
        assert np.all(np.isfinite(audio))
        print(f"Lead octave {octave:+d}: strongest note {peak:.1f} Hz (expected {expected:.0f} Hz).")

    # Check that live octave changes preserve the selected shifter path.
    fx = VoiceEffects(preset="extreme", voice_octave=0, backing_vocals="choir",
                      drums=True, bass=True, reverb="cher")
    for octave in (1, -1, 0):
        fx.voice_octave = octave
        y = fx.process(np.zeros(HOP, dtype=np.float32))
        assert y.shape == (HOP, 2) and np.all(np.isfinite(y))
    print("Live octave switching and harmony tracking passed.")
    print("All self-tests passed.")


def run(args):
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}. Install with: py -m pip install -r requirements.txt")

    if args.list:
        print(sd.query_devices())
        return
    if not args.headphones:
        raise SystemExit("Refusing to start: wear headphones and explicitly pass --headphones. Speakers can cause feedback.")
    if args.input is None or args.output is None:
        raise SystemExit("Select both device IDs with --input N --output N (find them using --list).")
    key = args.key.upper()
    if key not in KEYS:
        raise SystemExit(f"Unknown key: {args.key}. Try C, C#, Db, D, ... B.")
    if args.input == args.output:
        print("Using same device for input and output (fine for many USB headset interfaces).")

    processor = VoiceEffects(
        key=key, scale=args.scale, preset=args.preset, reverb=args.reverb,
        reverb_mix=args.reverb_mix, octave_mix=args.octave_mix,
        gate_db=args.gate_db, gain=args.gain, retune=args.retune,
        mono_output=not args.stereo, backing_vocals=args.backing_vocals,
        harmony_level=args.harmony_level, drums=args.drums, bass=args.bass,
        guitar=args.guitar, follow_voice=args.follow_voice,
        follow_strength=args.follow_strength, song_mode=args.song_mode,
        bpm=args.bpm, groove=args.groove, band_level=args.band_level,
        drum_level=args.drum_level, bass_level=args.bass_level,
        guitar_level=args.guitar_level,
        voice_octave=args.voice_octave,
    )
    stopped_due_to_overload = False
    overloads = 0
    callback_error = None

    def audio_callback(indata, outdata, frames, callback_time, status):
        nonlocal stopped_due_to_overload, overloads, callback_error
        outdata.fill(0)
        try:
            if frames != HOP:
                raise ValueError(f"Audio driver supplied {frames} frames instead of {HOP}")
            if status:
                overloads += 1
                if overloads >= 20:
                    stopped_due_to_overload = True
                    raise RuntimeError(f"PortAudio status repeatedly reported: {status}")
            else:
                overloads = max(0, overloads - 1)
            outdata[:] = processor.process(indata[:, 0])
        except Exception:
            callback_error = traceback.format_exc()
            outdata.fill(0)
            raise sd.CallbackAbort

    # v1 worked with one output channel; retain exactly that setup by default.
    # Stereo is now opt-in because not all Windows/PortAudio endpoints support it.
    output_channels = 2 if args.stereo and not args.mono_output else 1
    print(f"Input device {args.input} -> HEADPHONES output device {args.output} ({output_channels} ch)")
    print(f"Key {key} {args.scale} | preset {args.preset} | reverb {args.reverb} | gate={args.gate_db:g}dB "
          f"| lead octave {args.voice_octave:+d}")
    print(f"Backing vocals {args.backing_vocals} | drums {'ON' if args.drums else 'OFF'} "
          f"| bass {'ON' if args.bass else 'OFF'} | guitar {'ON' if args.guitar else 'OFF'} "
          f"| follow {'ON' if args.follow_voice else 'OFF'} | Song Mode {'ON' if args.song_mode else 'OFF'} "
          f"| {args.bpm} BPM | {args.groove} groove")
    print("Wired headphones recommended; disable Windows 'Listen to this device' / hardware monitoring.")
    if sys.platform == "win32":
        print("Keys: E=extreme, R=reverb, H=backing vocals, D=drums, B=bass, G=guitar, F=follow voice, "
              "M=Song Mode, O=lead octave (normal/up/down), [ / ]=tempo -/+5, "
              "+/-=reverb level, Q=quit.")
    else:
        print("Ctrl+C to stop. Presets/reverb can be set via command line.")
    print("Starting audio stream...", flush=True)
    try:
        sd.check_input_settings(device=args.input, channels=1, samplerate=SAMPLE_RATE, dtype="float32")
        sd.check_output_settings(device=args.output, channels=output_channels,
                                 samplerate=SAMPLE_RATE, dtype="float32")
        with sd.Stream(samplerate=SAMPLE_RATE, blocksize=HOP, dtype="float32",
                       channels=(1, output_channels), device=(args.input, args.output),
                       latency="low", callback=audio_callback) as stream:
            print("Audio stream STARTED. Listening to microphone. Press Q or Ctrl+C to stop.", flush=True)
            if sys.platform == "win32":
                # Ignore stale keystrokes accumulated before the console was ready.
                import msvcrt
                while msvcrt.kbhit():
                    msvcrt.getwch()
            while True:
                time.sleep(0.15)
                if callback_error is not None:
                    raise RuntimeError("Audio callback failed:\n" + callback_error)
                if stopped_due_to_overload:
                    raise RuntimeError("Audio callback overloaded; try stopping heavy applications.")
                if not stream.active:
                    raise RuntimeError("Audio device stopped unexpectedly (driver or stream error).")
                if sys.platform == "win32":
                    while msvcrt.kbhit():
                        keypress = msvcrt.getwch().lower()
                        if keypress == "e":
                            processor.preset = "classic" if processor.preset == "extreme" else "extreme"
                        elif keypress == "r":
                            processor.reverb_enabled = not processor.reverb_enabled
                        elif keypress == "h":
                            processor.harmony_mode = {"off": "duo", "duo": "choir", "choir": "wall", "wall": "off"}[processor.harmony_mode]
                        elif keypress == "d":
                            processor.band.drums_enabled = not processor.band.drums_enabled
                        elif keypress == "b":
                            processor.band.bass_enabled = not processor.band.bass_enabled
                        elif keypress == "g":
                            processor.band.guitar_enabled = not processor.band.guitar_enabled
                        elif keypress == "f":
                            processor.band.follow_voice = not processor.band.follow_voice
                        elif keypress == "m":
                            processor.band.song.set_enabled(not processor.band.song.enabled)
                        elif keypress == "o":
                            processor.voice_octave = {0: 1, 1: -1, -1: 0}[processor.voice_octave]
                        elif keypress == "[":
                            processor.band.set_bpm(processor.band.bpm - 5)
                        elif keypress == "]":
                            processor.band.set_bpm(processor.band.bpm + 5)
                        elif keypress in ("+", "="):
                            processor.reverb_mix = min(0.8, processor.reverb_mix + 0.05)
                        elif keypress == "-":
                            processor.reverb_mix = max(0.0, processor.reverb_mix - 0.05)
                        elif keypress == "q":
                            print("\nMonitoring stopped.")
                            return
                print(f"\rTarget: {processor.last_note:<4} | conf: {processor.last_confidence:.2f} "
                      f"| {processor.preset:<7} | rev {'ON' if processor.reverb_enabled else 'OFF'} "
                      f"| octave {processor.voice_octave:+d} | harm {processor.harmony_mode:<5} | D {'ON' if processor.band.drums_enabled else 'off'} "
                      f"B {'ON' if processor.band.bass_enabled else 'off'} "
                      f"G {'ON' if processor.band.guitar_enabled else 'off'} | follow {'ON' if processor.band.follow_voice else 'off'} "
                      f"| chord {processor.band.chord_name:<3} | Song {processor.band.song.section if processor.band.song.enabled else 'OFF':<6} "
                      f"| {processor.band.bpm:>3} BPM   ",
                      end="", flush=True)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
    except Exception as exc:
        raise SystemExit(f"Audio stream error: {exc}\n"
                         "This version defaults to MONO to match the working v1. "
                         "Verify your device IDs with --list and audio device settings.")


def main():
    parser = argparse.ArgumentParser(description="2008-style live microphone Auto-Tune + harmony and backing band")
    parser.add_argument("--list", action="store_true", help="List PortAudio device IDs")
    parser.add_argument("--selftest", action="store_true", help="Test pitch detection, no audio devices needed")
    parser.add_argument("--input", type=int, help="Input mic device ID")
    parser.add_argument("--output", type=int, help="Output HEADPHONES device ID")
    parser.add_argument("--headphones", action="store_true", help="Confirm headphones physically connected")
    parser.add_argument("--key", default="C", help="Musical key, e.g. C, F#, Bb")
    parser.add_argument("--scale", default="minor", choices=tuple(SCALES))
    parser.add_argument("--retune", type=float, default=1.0, help="0.0 to 1.0; 1.0=instant 2008 hard snap")
    parser.add_argument("--gate-db", type=float, default=-48.0)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--preset", choices=("classic", "extreme"), default="classic",
                        help="Extreme: instant center snaps, doubled upper octave, strong vocal drive")
    parser.add_argument("--reverb", choices=("off", "cher"), default="off",
                        help="Cher-inspired lush pop plate + stereo slapback (not an exact replica)")
    parser.add_argument("--reverb-mix", type=float, default=0.38,
                        help="Reverb wet level 0.0..0.8 (default 0.38)")
    parser.add_argument("--voice-octave", type=int, choices=(-1, 0, 1), default=0,
                        help="Transpose main tuned vocal and harmonies: -1 down, 0 normal, +1 up")
    parser.add_argument("--octave-mix", type=float, default=None,
                        help="High-octave robot layer 0.0..0.6 (classic default 0; extreme 0.25)")
    parser.add_argument("--mono-output", action="store_true",
                        help="Compatibility alias to force the default 1-channel output")
    parser.add_argument("--stereo", action="store_true",
                        help="Opt in to wide 2-channel reverb when your headphone device supports stereo")
    parser.add_argument("--backing-vocals", choices=tuple(HARMONY_COUNTS), default="off",
                        help="Off, 1-voice duo, 4-voice choir or 8-voice WALL harmonies")
    parser.add_argument("--harmony-level", type=float, default=0.55,
                        help="Backing voice strength, 0.0..1.5")
    parser.add_argument("--drums", action="store_true", help="Start with synthesized drum groove enabled")
    parser.add_argument("--bass", action="store_true", help="Start with synthesized bass enabled")
    parser.add_argument("--guitar", action="store_true", help="Start with synthesized strummed guitar enabled")
    parser.add_argument("--follow-voice", action="store_true", help="Singer selects chords on bar boundaries and affects band dynamics")
    parser.add_argument("--song-mode", action="store_true", help="Offline phrase memory and verse/chorus-like adaptive band arrangement")
    parser.add_argument("--follow-strength", type=float, default=0.50,
                        help="Singer-follow intensity 0.0..1.0: from laid-back 2-bar to beat-level chord updates")
    parser.add_argument("--bpm", type=int, default=110, help="Tempo for drums and bass (60..180)")
    parser.add_argument("--groove", choices=("pop", "dance", "half-time"), default="pop")
    parser.add_argument("--band-level", type=float, default=0.75, help="Backing band master volume, 0..1")
    parser.add_argument("--drum-level", type=float, default=0.42, help="Drum relative volume, 0..1")
    parser.add_argument("--bass-level", type=float, default=0.42, help="Bass relative volume, 0..1")
    parser.add_argument("--guitar-level", type=float, default=0.42, help="Guitar relative volume, 0..1")
    parser.add_argument("--debug", action="store_true",
                        help="Write retro_autotune_crash.log with Python native-crash details")
    args = parser.parse_args()
    if not 0 < args.retune <= 1:
        parser.error("--retune must be >0 and <=1")
    if not 0 < args.gain <= 2:
        parser.error("--gain must be >0 and <=2")
    if not 0 <= args.reverb_mix <= 0.8:
        parser.error("--reverb-mix must be 0..0.8")
    if args.octave_mix is not None and not 0 <= args.octave_mix <= 0.6:
        parser.error("--octave-mix must be 0..0.6")
    if args.stereo and args.mono_output:
        parser.error("Choose at most one of --stereo and --mono-output")
    if not 60 <= args.bpm <= 180:
        parser.error("--bpm must be 60..180")
    if not 0.0 <= args.follow_strength <= 1.0:
        parser.error("--follow-strength must be 0..1")
    for name in ("band_level", "drum_level", "bass_level", "guitar_level"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be 0..1")
    if not 0.0 <= args.harmony_level <= 1.5:
        parser.error("--harmony-level must be 0..1.5")
    if args.debug:
        with open("retro_autotune_crash.log", "w", encoding="utf-8") as crash_log:
            faulthandler.enable(file=crash_log, all_threads=True)
            try:
                self_test() if args.selftest else run(args)
            finally:
                faulthandler.disable()
    elif args.selftest:
        self_test()
    else:
        run(args)


if __name__ == "__main__":
    main()
