#!/usr/bin/env python3
"""Neon live studio interface for the Retro Auto-Tune v4.4 audio engine.

Tkinter only for controls/visuals, NumPy and sounddevice for audio.
No Tk functions run on the PortAudio callback thread. Use wired headphones.
"""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
import queue
import threading
import time
import traceback
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np

from retro_autotune import (
    HOP, SAMPLE_RATE, KEYS, SCALES, VoiceEffects, allowed_notes
)
from mp3_recorder import Mp3Recorder

BG = "#090d19"
PANEL = "#151d30"
INNER = "#11182a"
EDGE = "#283650"
WHITE = "#edf6ff"
MUTED = "#879ab5"
CYAN = "#36eaff"
PINK = "#ff46af"
LIME = "#b3ff6f"
ORANGE = "#ffb45c"
RED = "#ff6b7d"


class AudioWorker:
    """Audio device ownership and rendering remain outside the Tk mainloop."""

    def __init__(self, config):
        self.config = dict(config)
        self.commands = queue.SimpleQueue()
        self.meters = queue.Queue(maxsize=3)
        self.events = queue.SimpleQueue()
        self.stop_event = threading.Event()
        self.thread = None
        self.running = False
        self.callback_count = 0
        self.callback_error = None
        self.overloads = 0
        self.volume = float(config["volume"])
        self.history = np.zeros(2048, dtype=np.float32)
        self.recorder = None
        self.record_starting = False

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True, name="AutoTune-Audio")
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.stop_recording()

    def start_recording(self, destination):
        """Launch MP3 encoder off the real-time audio processing thread."""
        if self.stop_event.is_set() or self.recorder is not None or self.record_starting:
            return False
        self.record_starting = True

        def prepare():
            try:
                recorder = Mp3Recorder(destination,
                                       2 if self.config["stereo"] else 1,
                                       self.events, sample_rate=SAMPLE_RATE)
                recorder.start()
                if self.stop_event.is_set():
                    recorder.stop()
                else:
                    self.recorder = recorder
                    self.events.put(("recording_started", str(destination)))
            except Exception:
                self.events.put(("record_error", traceback.format_exc()))
            finally:
                self.record_starting = False

        threading.Thread(target=prepare, name="AutoTune-MP3-Setup", daemon=True).start()
        return True

    def stop_recording(self):
        recorder, self.recorder = self.recorder, None
        if recorder is not None:
            recorder.stop()
            self.events.put(("record_stopping", "Finishing MP3…"))

    def command(self, name, value):
        self.commands.put((name, value))

    @staticmethod
    def apply_command(fx, name, value):
        """Called at the start of a block, not simultaneously with processing."""
        if name == "preset":
            fx.preset = value
        elif name == "reverb":
            fx.reverb_enabled = bool(value)
        elif name == "reverb_mix":
            fx.reverb_mix = float(value)
        elif name == "harmony":
            fx.harmony_mode = value
        elif name == "harmony_level":
            fx.harmony_level = float(value)
        elif name == "drums":
            fx.band.drums_enabled = bool(value)
        elif name == "bass":
            fx.band.bass_enabled = bool(value)
        elif name == "guitar":
            fx.band.guitar_enabled = bool(value)
        elif name == "follow_voice":
            fx.band.follow_voice = bool(value)
        elif name == "follow_strength":
            fx.band.follow_strength = float(np.clip(value, 0.0, 1.0))
        elif name == "guitar_level":
            fx.band.guitar_level = float(value)
        elif name == "drum_level":
            fx.band.drum_level = float(value)
        elif name == "bass_level":
            fx.band.bass_level = float(value)
        elif name == "band_level":
            fx.band.band_level = float(value)
        elif name == "bpm":
            fx.band.set_bpm(value)
        elif name == "groove":
            fx.band.groove = value
        elif name == "voice_octave":
            fx.voice_octave = int(value)
        elif name == "octave_mix":
            fx.octave_mix_override = float(value)
        elif name == "gain":
            fx.gain = float(value)
        elif name == "key_scale":
            key, scale = value
            fx.candidates = allowed_notes(key, scale)
            fx.scale = scale
            fx.current_target = None
            fx.last_note = "--"
            fx.band.set_key_scale(key, scale)

    @staticmethod
    def host_api_mismatch(sd, mic, headphones):
        """A PortAudio duplex Stream requires input/output on the same host API.

        If device metadata is unavailable (e.g. in a fake driver) retain the
        original duplex path and allow a -9993 error to trigger the fallback.
        """
        if not hasattr(sd, "query_devices"):
            return False
        input_info = sd.query_devices(mic)
        output_info = sd.query_devices(headphones)
        return input_info["hostapi"] != output_info["hostapi"]

    @staticmethod
    def is_device_pair_error(exc):
        """PortAudio -9993: illegal combination of I/O devices."""
        return any(str(arg) == "-9993" or "Illegal combination of I/O devices" in str(arg)
                   for arg in getattr(exc, "args", ()))

    def _run(self):
        try:
            import sounddevice as sd
            c = self.config
            channels = 2 if c["stereo"] else 1
            sd.check_input_settings(device=c["input"], channels=1,
                                    samplerate=SAMPLE_RATE, dtype="float32")
            sd.check_output_settings(device=c["output"], channels=channels,
                                     samplerate=SAMPLE_RATE, dtype="float32")
            fx = VoiceEffects(
                key=c["key"], scale=c["scale"], preset=c["preset"],
                reverb="cher" if c["reverb"] else "off",
                reverb_mix=c["reverb_mix"],
                octave_mix=c["octave_mix"],
                gate_db=-48, gain=c["gain"], mono_output=not c["stereo"],
                backing_vocals=c["harmony"], harmony_level=c["harmony_level"],
                drums=c["drums"], bass=c["bass"], guitar=c.get("guitar", False),
                follow_voice=c.get("follow_voice", False),
                follow_strength=c.get("follow_strength", 0.50),
                guitar_level=c.get("guitar_level", 0.42), bpm=c["bpm"],
                bass_level=c.get("bass_level", 0.42),
                drum_level=c.get("drum_level", 0.42),
                band_level=c.get("band_level", 0.75),
                groove=c["groove"], voice_octave=c["voice_octave"],
            )

            def render(indata, outdata, frames, status):
                """Common processor for duplex and split-stream outputs."""
                outdata.fill(0)
                try:
                    if self.stop_event.is_set():
                        return
                    if frames != HOP:
                        raise ValueError(f"Driver delivered {frames} frames, expected {HOP}")
                    if status:
                        self.overloads += 1
                        if self.overloads >= 20:
                            raise RuntimeError(f"Repeated audio underrun/overrun: {status}")
                    else:
                        self.overloads = max(0, self.overloads - 1)
                    for _ in range(32):
                        try:
                            name, value = self.commands.get_nowait()
                        except queue.Empty:
                            break
                        if name == "volume":
                            self.volume = float(value)
                        else:
                            self.apply_command(fx, name, value)
                    processed = fx.process(indata[:, 0])
                    outdata[:] = processed * self.volume
                    recorder = self.recorder
                    if recorder is not None:
                        recorder.feed(outdata)
                    self.history[:-HOP] = self.history[HOP:]
                    self.history[-HOP:] = outdata[:, 0]
                    self.callback_count += 1
                    if self.callback_count % 4 == 0:
                        snapshot = {
                            "samples": self.history.copy(),
                            "input_db": 20 * math.log10(max(1e-6, float(np.sqrt(np.mean(indata[:, 0] ** 2))))),
                            "output_db": 20 * math.log10(max(1e-6, float(np.sqrt(np.mean(outdata[:, 0] ** 2))))),
                            "note": fx.last_note if fx.gate_level > 0.10 else "--",
                            "chord": fx.band.chord_name,
                            "follow_voice": fx.band.follow_voice,
                            "follow_strength": fx.band.follow_strength,
                            "confidence": fx.last_confidence if fx.gate_level > 0.10 else 0.0,
                            "beat": fx.band.step_index // 4 % 4,
                            "step": fx.band.step_index,
                            "bpm": fx.band.bpm,
                            "overloads": self.overloads,
                        }
                        try:
                            self.meters.put_nowait(snapshot)
                        except queue.Full:
                            pass  # Never block live audio to refresh a visual.
                except Exception:
                    self.callback_error = traceback.format_exc()
                    outdata.fill(0)
                    self.stop_event.set()
                    raise sd.CallbackAbort

            def duplex_callback(indata, outdata, frames, callback_time, status):
                render(indata, outdata, frames, status)

            def monitor(streams, mode):
                self.running = True
                self.events.put(("running", f"LIVE  ·  {mode} · headphones"))
                while not self.stop_event.wait(0.08):
                    if self.callback_error:
                        raise RuntimeError("Audio processing stopped:\n" + self.callback_error)
                    if not all(stream.active for stream in streams):
                        raise RuntimeError("An audio device stopped unexpectedly. Check the headphone connection.")
                if self.callback_error:
                    raise RuntimeError("Audio processing stopped:\n" + self.callback_error)

            def split_streams():
                # Two independent PortAudio streams allow devices from different
                # Windows audio host APIs (for example MME mic + WASAPI headphones).
                # Keep a bounded FIFO: if device clocks drift, drop old microphone
                # frames rather than allowing progressively larger vocal latency.
                microphone_frames = queue.Queue(maxsize=4)
                silence = np.zeros((HOP, 1), dtype=np.float32)

                def input_callback(indata, frames, callback_time, status):
                    if self.stop_event.is_set():
                        return
                    if frames != HOP:
                        self.callback_error = f"Input delivered {frames} frames, expected {HOP}"
                        self.stop_event.set()
                        raise sd.CallbackAbort
                    if status:
                        self.overloads += 1
                    data = indata.copy()
                    try:
                        microphone_frames.put_nowait(data)
                    except queue.Full:
                        try:
                            microphone_frames.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            microphone_frames.put_nowait(data)
                        except queue.Full:
                            pass

                def output_callback(outdata, frames, callback_time, status):
                    try:
                        captured = microphone_frames.get_nowait()
                    except queue.Empty:
                        captured = silence
                    render(captured, outdata, frames, status)

                self.events.put(("starting", "Using separate microphone/headphone streams for Windows API compatibility..."))
                with sd.InputStream(device=c["input"], samplerate=SAMPLE_RATE,
                                    blocksize=HOP, dtype="float32", channels=1,
                                    latency="low", callback=input_callback) as mic_stream:
                    with sd.OutputStream(device=c["output"], samplerate=SAMPLE_RATE,
                                         blocksize=HOP, dtype="float32", channels=channels,
                                         latency="low", callback=output_callback) as output_stream:
                        monitor((mic_stream, output_stream), "split-device mode")

            self.events.put(("starting", "Connecting the selected audio devices..."))
            if self.host_api_mismatch(sd, c["input"], c["output"]):
                split_streams()
            else:
                try:
                    with sd.Stream(samplerate=SAMPLE_RATE, blocksize=HOP, dtype="float32",
                                   channels=(1, channels), device=(c["input"], c["output"]),
                                   latency="low", callback=duplex_callback) as stream:
                        monitor((stream,), "duplex mode")
                except Exception as exc:
                    if not self.is_device_pair_error(exc) or self.stop_event.is_set():
                        raise
                    split_streams()
        except Exception:
            self.events.put(("error", traceback.format_exc()))
        finally:
            self.stop_recording()
            self.running = False
            self.events.put(("stopped", "Audio monitoring stopped"))


class StudioApp:
    def __init__(self, root):
        self.root = root
        root.title("RETRO / TUNE 2008 — LIVE STUDIO")
        root.configure(bg=BG)
        screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{min(1140, max(620, screen_w-90))}x{min(850, max(480, screen_h-120))}")
        root.minsize(620, 480)
        self.worker = None
        self.closing = False
        self.last_snapshot = None
        self.devices_in = {}
        self.devices_out = {}
        self.status_var = tk.StringVar(value="STOPPED  ·  Select your microphone and headphones")
        self.note_var = tk.StringVar(value="--")
        self.pitch_var = tk.StringVar(value="Sing into the microphone")
        self.chord_var = tk.StringVar(value="Backing chord: --")
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.headphones_var = tk.BooleanVar(value=False)
        self.stereo_var = tk.BooleanVar(value=False)
        self.key_var = tk.StringVar(value="C")
        self.scale_var = tk.StringVar(value="minor")
        self.preset_var = tk.StringVar(value="extreme")
        self.reverb_var = tk.BooleanVar(value=True)
        self.harmony_var = tk.StringVar(value="choir")
        self.drums_var = tk.BooleanVar(value=True)
        self.bass_var = tk.BooleanVar(value=True)
        self.guitar_var = tk.BooleanVar(value=True)
        self.follow_voice_var = tk.BooleanVar(value=True)
        self.follow_strength_var = tk.DoubleVar(value=0.50)
        self.guitar_level_var = tk.DoubleVar(value=0.42)
        self.drum_level_var = tk.DoubleVar(value=0.42)
        self.bass_level_var = tk.DoubleVar(value=0.42)
        self.band_level_var = tk.DoubleVar(value=0.75)
        self.groove_var = tk.StringVar(value="dance")
        self.octave_var = tk.IntVar(value=0)
        self.bpm_var = tk.IntVar(value=110)
        self.reverb_mix_var = tk.DoubleVar(value=0.50)
        self.harmony_level_var = tk.DoubleVar(value=0.85)
        self.volume_var = tk.DoubleVar(value=0.55)
        self.octave_mix_var = tk.DoubleVar(value=0.20)
        self.gain_var = tk.DoubleVar(value=1.0)
        self.chaos_enabled = False
        self.before_chaos = None
        self.recording = False
        self.record_filename = ""
        self._event_queues = []
        self._layout_mode = None
        self._setup_styles()
        self._build()
        self.refresh_devices()
        root.bind("<Configure>", self._resize_layout)
        root.after(120, lambda: self._resize_layout())
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(40, self._poll)

    def _setup_styles(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Studio.TCombobox", fieldbackground=INNER, background=PANEL,
                        foreground=WHITE, arrowcolor=CYAN, padding=5,
                        bordercolor=EDGE, lightcolor=EDGE, darkcolor=EDGE)
        style.map("Studio.TCombobox", fieldbackground=[("readonly", INNER)],
                  foreground=[("readonly", WHITE)])
        style.configure("Studio.Horizontal.TScale", background=PANEL, troughcolor=INNER)

    def _label(self, parent, text, color=MUTED, size=10, bold=False, **kw):
        return tk.Label(parent, text=text, fg=color, bg=parent.cget("bg"),
                        font=("Segoe UI", size, "bold" if bold else "normal"), **kw)

    def _panel(self, parent, title=None):
        holder = tk.Frame(parent, bg=PANEL, highlightthickness=1, highlightbackground=EDGE)
        if title:
            self._label(holder, title.upper(), CYAN, 10, True).pack(anchor="w", padx=16, pady=(12, 4))
        return holder

    def _combo(self, parent, var, values, width=17, callback=None):
        combo = ttk.Combobox(parent, textvariable=var, values=list(values),
                             state="readonly", width=width, style="Studio.TCombobox")
        if callback:
            combo.bind("<<ComboboxSelected>>", lambda _: callback())
        return combo

    def _button(self, parent, text, command, width=None, active=False, accent=CYAN):
        b = tk.Button(parent, text=text, command=command, cursor="hand2", bd=0,
                      relief="flat", highlightthickness=0, padx=12, pady=9,
                      font=("Segoe UI", 10, "bold"),
                      bg=accent if active else INNER, fg=BG if active else WHITE,
                      activebackground=accent, activeforeground=BG,
                      width=width)
        return b

    def _update_button(self, btn, on, accent=CYAN):
        btn.configure(bg=accent if on else INNER, fg=BG if on else WHITE)

    def _build(self):
        shell = tk.Frame(self.root, bg=BG)
        shell.pack(fill="both", expand=True)
        scroller = tk.Canvas(shell, bg=BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(shell, orient="vertical", command=scroller.yview)
        scroller.configure(yscrollcommand=scrollbar.set)
        scroller.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        body = tk.Frame(scroller, bg=BG)
        win = scroller.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda _: scroller.configure(scrollregion=scroller.bbox("all")))
        scroller.bind("<Configure>", lambda e: scroller.itemconfigure(win, width=e.width))
        def wheel(e):
            scroller.yview_scroll(-int(e.delta / 120), "units")
        scroller.bind_all("<MouseWheel>", wheel)

        header = tk.Frame(body, bg=BG)
        header.pack(fill="x", padx=18, pady=(18, 12))
        self._label(header, "RETRO / TUNE", CYAN, 25, True).pack(side="left")
        self._label(header, "  2008  •  LIVE VOCAL STUDIO", PINK, 11, True).pack(side="left", pady=11)
        self._label(header, "HEADPHONES ONLY", ORANGE, 10, True).pack(side="right", pady=12)

        devices = self._panel(body, "01   Audio routing & safety")
        devices.pack(fill="x", padx=18, pady=(0, 12))
        device_row = tk.Frame(devices, bg=PANEL)
        self.device_row = device_row
        device_row.pack(fill="x", padx=16, pady=5)
        input_col = tk.Frame(device_row, bg=PANEL)
        self.input_col = input_col
        input_col.pack(side="left", fill="x", expand=True, padx=(0, 9))
        self._label(input_col, "MICROPHONE", size=9).pack(anchor="w")
        self.input_combo = self._combo(input_col, self.input_var, [], width=35)
        self.input_combo.pack(fill="x", pady=4)
        output_col = tk.Frame(device_row, bg=PANEL)
        self.output_col = output_col
        output_col.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self._label(output_col, "HEADPHONE OUTPUT (NOT SPEAKERS)", size=9).pack(anchor="w")
        self.output_combo = self._combo(output_col, self.output_var, [], width=35)
        self.output_combo.pack(fill="x", pady=4)
        self.refresh_btn = self._button(device_row, "↻ Devices", self.refresh_devices)
        self.refresh_btn.pack(side="right", pady=(15, 0))
        check_row = tk.Frame(devices, bg=PANEL)
        check_row.pack(fill="x", padx=16, pady=(0, 11))
        self.headphones_check = tk.Checkbutton(
            check_row, text="I have plugged in headphones (no speakers / direct mic monitoring)",
            variable=self.headphones_var, fg=WHITE, bg=PANEL, selectcolor=INNER,
            activebackground=PANEL, activeforeground=WHITE, font=("Segoe UI", 10))
        self.headphones_check.pack(side="left")
        self.stereo_check = tk.Checkbutton(
            check_row, text="Stereo (optional)", variable=self.stereo_var,
            fg=MUTED, bg=PANEL, selectcolor=INNER,
            activebackground=PANEL, activeforeground=WHITE, font=("Segoe UI", 10))
        self.stereo_check.pack(side="right")
        self._label(devices, "Different Windows audio APIs? The app automatically uses separate streams (slightly more latency).",
                    MUTED, 9).pack(anchor="w", padx=16, pady=(0, 10))

        transport = tk.Frame(body, bg=BG)
        transport.pack(fill="x", padx=18, pady=(0, 12))
        action_row = tk.Frame(transport, bg=BG)
        action_row.pack(fill="x")
        self.start_btn = self._button(action_row, "▶  START LIVE", self.start_audio, active=True, accent=LIME)
        self.start_btn.pack(side="left")
        self.stop_btn = self._button(action_row, "■  STOP", self.stop_audio)
        self.stop_btn.configure(state="disabled")
        self.stop_btn.pack(side="left", padx=8)
        self.record_btn = self._button(action_row, "●  RECORD MP3", self.toggle_recording, accent=RED)
        self.record_btn.configure(state="disabled")
        self.record_btn.pack(side="left", padx=3)
        self.chaos_btn = self._button(action_row, "★  MAX CHAOS", self.toggle_chaos, accent=PINK)
        self.chaos_btn.pack(side="left", padx=8)
        self.status_label = self._label(transport, "", MUTED, 10, True,
                                       textvariable=self.status_var, anchor="w", justify="left")
        self.status_label.pack(fill="x", pady=(7, 0))
        self.transport = transport

        vizrow = tk.Frame(body, bg=BG)
        self.vizrow = vizrow
        vizrow.pack(fill="x", padx=18, pady=(0, 12))
        wave_panel = self._panel(vizrow, "Live vocal waveform  /  tuned output")
        self.wave_panel = wave_panel
        wave_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        self.wave_canvas = tk.Canvas(wave_panel, bg=INNER, height=184, highlightthickness=0)
        self.wave_canvas.pack(fill="x", padx=14, pady=(8, 14))
        self.wave_canvas.bind("<Configure>", lambda _: self._wave_grid())
        self.wave_line = self.wave_canvas.create_line(0, 92, 800, 92, fill=CYAN, width=2,
                                                       smooth=True)
        pitch_panel = self._panel(vizrow, "Pitch lock  /  detected note")
        self.pitch_panel = pitch_panel
        pitch_panel.configure(width=262)
        pitch_panel.pack(side="right", fill="y")
        pitch_panel.pack_propagate(False)
        self._label(pitch_panel, "", WHITE, 48, True, textvariable=self.note_var).pack(pady=(12, 2))
        self._label(pitch_panel, "", MUTED, 10, False,
                    textvariable=self.pitch_var).pack(pady=(0, 5))
        self._label(pitch_panel, "", LIME, 11, True,
                    textvariable=self.chord_var).pack(pady=(0, 8))
        tk.Frame(pitch_panel, bg=PANEL, height=5).pack(pady=5)

        meters = tk.Frame(body, bg=BG)
        self.meters_row = meters
        meters.pack(fill="x", padx=18, pady=(0, 12))
        spec_panel = self._panel(meters, "Audio spectrum  /  50 Hz – 8 kHz")
        self.spec_panel = spec_panel
        spec_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        self.spectrum_canvas = tk.Canvas(spec_panel, bg=INNER, height=155, highlightthickness=0)
        self.spectrum_canvas.pack(fill="x", padx=14, pady=(8, 14))
        self.spectrum_bars = []
        for i in range(32):
            r = self.spectrum_canvas.create_rectangle(0, 155, 1, 154, outline="", fill=CYAN)
            self.spectrum_bars.append(r)
        beat_panel = self._panel(meters, "Levels  /  rhythm")
        self.beat_panel = beat_panel
        beat_panel.configure(width=262)
        beat_panel.pack(side="right", fill="y")
        beat_panel.pack_propagate(False)
        self.input_level_var = tk.StringVar(value="MIC  -∞ dB")
        self.output_level_var = tk.StringVar(value="OUT  -∞ dB")
        self._label(beat_panel, "", WHITE, 11, True, textvariable=self.input_level_var).pack(anchor="w", padx=14, pady=(8, 2))
        self.mic_meter = tk.Canvas(beat_panel, bg=INNER, height=13, highlightthickness=0)
        self.mic_meter.pack(fill="x", padx=14)
        self._label(beat_panel, "", WHITE, 11, True, textvariable=self.output_level_var).pack(anchor="w", padx=14, pady=(10, 2))
        self.out_meter = tk.Canvas(beat_panel, bg=INNER, height=13, highlightthickness=0)
        self.out_meter.pack(fill="x", padx=14)
        beats = tk.Frame(beat_panel, bg=PANEL)
        beats.pack(pady=(12, 2))
        self.beat_boxes = [tk.Label(beats, text=str(i+1), bg=INNER, fg=MUTED,
                                    font=("Segoe UI", 10, "bold"), width=3, pady=4)
                           for i in range(4)]
        for b in self.beat_boxes:
            b.pack(side="left", padx=4)

        ctrls = tk.Frame(body, bg=BG)
        self.ctrls = ctrls
        ctrls.pack(fill="x", padx=18, pady=(0, 16))
        ctrls.grid_columnconfigure(0, weight=1, uniform="ctrl")
        ctrls.grid_columnconfigure(1, weight=1, uniform="ctrl")
        ctrls.grid_columnconfigure(2, weight=1, uniform="ctrl")
        pitch = self._panel(ctrls, "02   Pitch & lead voice")
        self.pitch_controls = pitch
        pitch.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        scale_row = tk.Frame(pitch, bg=PANEL)
        scale_row.pack(fill="x", padx=14, pady=8)
        for j, (label, var, options) in enumerate((
            ("KEY", self.key_var, tuple(KEYS)),
            ("SCALE", self.scale_var, tuple(SCALES)),
        )):
            part = tk.Frame(scale_row, bg=PANEL)
            part.pack(side="left", fill="x", expand=True, padx=(0, 6) if j == 0 else (6, 0))
            self._label(part, label, size=9).pack(anchor="w")
            self._combo(part, var, options, width=12, callback=self._key_scale_change).pack(fill="x", pady=4)
        self._label(pitch, "AUTO-TUNE STRENGTH", size=9).pack(anchor="w", padx=14, pady=(4, 0))
        self.preset_btn = self._button(pitch, "EXTREME  /  ROBOT", self._toggle_preset)
        self.preset_btn.pack(fill="x", padx=14, pady=7)
        self._label(pitch, "LEAD VOICE OCTAVE", size=9).pack(anchor="w", padx=14, pady=(10, 0))
        octrow = tk.Frame(pitch, bg=PANEL)
        octrow.pack(fill="x", padx=14, pady=6)
        self.octave_buttons = {}
        for octave, label in ((-1, "−1 LOW"), (0, "NORMAL"), (1, "+1 HIGH")):
            b = self._button(octrow, label, lambda x=octave: self._set_octave(x))
            b.pack(side="left", fill="x", expand=True, padx=2)
            self.octave_buttons[octave] = b
        self._slider(pitch, "VOCAL GAIN", self.gain_var, 0.3, 2.0,
                     lambda v: self._send("gain", round(float(v), 2)))
        self._slider(pitch, "ROBOT OCTAVE LAYER", self.octave_mix_var, 0.0, 0.6,
                     lambda v: self._send("octave_mix", round(float(v), 2)))

        vocal = self._panel(ctrls, "03   Backing vocals & space")
        self.vocal_controls = vocal
        vocal.grid(row=0, column=1, sticky="nsew", padx=7)
        self._label(vocal, "HARMONY VOICES", size=9).pack(anchor="w", padx=14, pady=(8, 0))
        hr = tk.Frame(vocal, bg=PANEL)
        hr.pack(fill="x", padx=14, pady=8)
        self.harmony_buttons = {}
        for mode in ("off", "duo", "choir", "wall"):
            b = self._button(hr, mode.upper(), lambda x=mode: self._set_harmony(x))
            b.pack(side="left", fill="x", expand=True, padx=2)
            self.harmony_buttons[mode] = b
        self._slider(vocal, "HARMONY LEVEL", self.harmony_level_var, 0.0, 1.5,
                     lambda v: self._send("harmony_level", round(float(v), 2)))
        self._label(vocal, "REVERB", size=9).pack(anchor="w", padx=14, pady=(10, 0))
        self.reverb_btn = self._button(vocal, "CHER-STYLE POP REVERB", self._toggle_reverb)
        self.reverb_btn.pack(fill="x", padx=14, pady=7)
        self._slider(vocal, "REVERB MIX", self.reverb_mix_var, 0.0, 0.8,
                     lambda v: self._send("reverb_mix", round(float(v), 2)))
        self._slider(vocal, "HEADPHONE VOLUME", self.volume_var, 0.0, 1.0,
                     lambda v: self._send("volume", round(float(v), 2)))

        band = self._panel(ctrls, "04   Voice-following backing band")
        self.band_controls = band
        band.grid(row=0, column=2, sticky="nsew", padx=(7, 0))
        instruments = tk.Frame(band, bg=PANEL)
        instruments.pack(fill="x", padx=14, pady=(8, 12))
        self.drums_btn = self._button(instruments, "DRUMS", self._toggle_drums)
        self.drums_btn.pack(side="left", fill="x", expand=True, padx=(0, 3))
        self.bass_btn = self._button(instruments, "BASS", self._toggle_bass)
        self.bass_btn.pack(side="left", fill="x", expand=True, padx=(3, 3))
        self.guitar_btn = self._button(instruments, "GUITAR", self._toggle_guitar)
        self.guitar_btn.pack(side="left", fill="x", expand=True, padx=(3, 0))
        self.follow_btn = self._button(band, "FOLLOW MY VOICE / ADAPT CHORDS", self._toggle_follow_voice)
        self.follow_btn.pack(fill="x", padx=14, pady=(0, 8))
        self._slider(band, "VOICE FOLLOW INTENSITY  /  RELAXED → TIGHT", self.follow_strength_var,
                     0.0, 1.0, lambda v: self._send("follow_strength", round(float(v), 2)))
        self._slider(band, "GUITAR LEVEL", self.guitar_level_var, 0.0, 1.0,
                     lambda v: self._send("guitar_level", round(float(v), 2)))
        self._slider(band, "BASS LEVEL", self.bass_level_var, 0.0, 1.0,
                     lambda v: self._send("bass_level", round(float(v), 2)))
        self._slider(band, "DRUM LEVEL", self.drum_level_var, 0.0, 1.0,
                     lambda v: self._send("drum_level", round(float(v), 2)))
        self._label(band, "GROOVE", size=9).pack(anchor="w", padx=14)
        self._combo(band, self.groove_var, ("pop", "dance", "half-time"),
                    callback=lambda: self._send("groove", self.groove_var.get())).pack(fill="x", padx=14, pady=6)
        self._label(band, "TEMPO  /  BPM", size=9).pack(anchor="w", padx=14, pady=(14, 0))
        bpmrow = tk.Frame(band, bg=PANEL)
        bpmrow.pack(fill="x", padx=14, pady=8)
        self._button(bpmrow, "− 5", lambda: self._set_bpm(self.bpm_var.get()-5)).pack(side="left")
        tk.Label(bpmrow, textvariable=self.bpm_var, bg=PANEL, fg=LIME,
                 font=("Segoe UI", 22, "bold")).pack(side="left", fill="x", expand=True)
        self._button(bpmrow, "+ 5", lambda: self._set_bpm(self.bpm_var.get()+5)).pack(side="right")
        self._label(band, "Tempo-locked drums · vocal-guided bass & guitar chords", MUTED, 9).pack(anchor="w", padx=14, pady=12)

        self._label(body, "v4.4  ·  Max Chaos, MP3 recording & tighter voice-follow  ·  wired headphones only",
                    MUTED, 9).pack(padx=18, pady=(0, 24))
        self._sync_controls()
        self._wave_grid()

    def _slider(self, panel, name, variable, lo, hi, callback):
        header = tk.Frame(panel, bg=PANEL)
        header.pack(fill="x", padx=14, pady=(8, 0))
        self._label(header, name, size=9).pack(side="left")
        value_label = self._label(header, f"{variable.get():.2f}", CYAN, 10, True)
        value_label.pack(side="right")
        def change(value):
            value_label.configure(text=f"{float(value):.2f}")
            callback(value)
        tk.Scale(panel, from_=lo, to=hi, variable=variable, command=change,
                 resolution=0.01, orient="horizontal", showvalue=False,
                 bg=PANEL, fg=WHITE, troughcolor=INNER, activebackground=CYAN,
                 highlightthickness=0, bd=0, sliderlength=18).pack(fill="x", padx=14)

    def _resize_layout(self, event=None):
        """Switch three/two/one control columns to fit available window width."""
        if event is not None and event.widget is not self.root:
            return
        width = self.root.winfo_width()
        columns = 3 if width >= 1080 else (2 if width >= 760 else 1)
        stacked = width < 830
        mode = (columns, stacked)
        if mode == self._layout_mode:
            return
        self._layout_mode = mode
        for i in range(3):
            self.ctrls.grid_columnconfigure(i, weight=1 if i < columns else 0,
                                            uniform="ctrl" if i < columns else "")
        for widget in (self.pitch_controls, self.vocal_controls, self.band_controls):
            widget.grid_forget()
        for i, widget in enumerate((self.pitch_controls, self.vocal_controls, self.band_controls)):
            row, column = divmod(i, columns)
            span = 2 if columns == 2 and i == 2 else 1
            widget.grid(row=row, column=column, columnspan=span,
                        sticky="nsew", padx=5, pady=(0, 10))

        self.input_col.pack_forget()
        self.output_col.pack_forget()
        if stacked:
            self.input_col.pack(side="top", fill="x", expand=True, padx=0)
            self.output_col.pack(side="top", fill="x", expand=True, padx=0)
        else:
            self.input_col.pack(side="left", fill="x", expand=True, padx=(0, 9))
            self.output_col.pack(side="left", fill="x", expand=True, padx=(9, 0))

        for first, second, fixed_height in ((self.wave_panel, self.pitch_panel, 214),
                                            (self.spec_panel, self.beat_panel, 214)):
            first.pack_forget()
            second.pack_forget()
            if stacked:
                first.pack(side="top", fill="x", pady=(0, 9))
                second.configure(width=260, height=fixed_height)
                second.pack(side="top", fill="x")
            else:
                first.pack(side="left", fill="both", expand=True, padx=(0, 10))
                second.configure(width=262, height=fixed_height)
                second.pack(side="right", fill="y")
        self._wave_grid()

    def toggle_chaos(self):
        """One-button maximal effects, reversible without changing headphone volume."""
        bindings = {
            "preset": (self.preset_var, "extreme"),
            "reverb": (self.reverb_var, True),
            "reverb_mix": (self.reverb_mix_var, 0.72),
            "harmony": (self.harmony_var, "wall"),
            "harmony_level": (self.harmony_level_var, 1.35),
            "drums": (self.drums_var, True),
            "bass": (self.bass_var, True),
            "guitar": (self.guitar_var, True),
            "follow_voice": (self.follow_voice_var, True),
            "follow_strength": (self.follow_strength_var, 1.00),
            "guitar_level": (self.guitar_level_var, 0.90),
            "bass_level": (self.bass_level_var, 0.78),
            "drum_level": (self.drum_level_var, 0.75),
            "band_level": (self.band_level_var, 0.90),
            "octave_mix": (self.octave_mix_var, 0.50),
            "groove": (self.groove_var, "dance"),
        }
        if not self.chaos_enabled:
            self.before_chaos = {name: var.get() for name, (var, _) in bindings.items()}
            settings = {name: new for name, (_, new) in bindings.items()}
            self.chaos_enabled = True
        else:
            settings = self.before_chaos or {}
            self.before_chaos = None
            self.chaos_enabled = False
        for name, value in settings.items():
            var = bindings[name][0]
            var.set(value)
            self._send(name, value)
        self._sync_controls()

    def toggle_recording(self):
        if self.worker is None or not self.worker.running:
            return
        if self.worker.recorder is not None:
            self.worker.stop_recording()
            self.recording = False
            self.record_btn.configure(text="●  RECORD MP3", bg=INNER, fg=WHITE)
            return
        destination = filedialog.asksaveasfilename(
            parent=self.root, title="Record the complete Auto-Tune mix as MP3",
            defaultextension=".mp3", filetypes=[("MP3 audio", "*.mp3")],
            initialfile=f"RetroAutoTune_{datetime.now():%Y%m%d_%H%M%S}.mp3",
            initialdir=str(Path.home() / "Music") if (Path.home() / "Music").is_dir() else str(Path.home()))
        if destination and self.worker.start_recording(destination):
            self.record_filename = destination
            self.record_btn.configure(state="disabled", text="Preparing MP3…")

    def _wave_grid(self):
        canvas = getattr(self, "wave_canvas", None)
        if not canvas:
            return
        w, h = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        canvas.delete("grid")
        for x in range(0, w, 45):
            canvas.create_line(x, 0, x, h, fill="#1b2b41", tags="grid")
        for y in range(0, h, 30):
            canvas.create_line(0, y, w, y, fill="#1b2b41", tags="grid")
        canvas.tag_lower("grid")

    def refresh_devices(self):
        if self.worker is not None:
            return
        current_in, current_out = self.input_var.get(), self.output_var.get()
        try:
            import sounddevice as sd
            all_devices = sd.query_devices()
            hosts = sd.query_hostapis()
            self.devices_in, self.devices_out = {}, {}
            for i, d in enumerate(all_devices):
                host = hosts[d["hostapi"]]["name"]
                text = f"{i}: {d['name']}  [{host}]"
                if d["max_input_channels"] > 0:
                    self.devices_in[text] = i
                if d["max_output_channels"] > 0:
                    self.devices_out[text] = i
            self.input_combo["values"] = list(self.devices_in)
            self.output_combo["values"] = list(self.devices_out)
            if current_in in self.devices_in:
                self.input_var.set(current_in)
            else:
                self.input_var.set("")
            if current_out in self.devices_out:
                self.output_var.set(current_out)
            else:
                self.output_var.set("")
            if not self.devices_in or not self.devices_out:
                self.status_var.set("No audio devices detected. Plug in headphones and refresh.")
        except Exception as exc:
            self.status_var.set("Audio devices unavailable: " + str(exc))

    def _send(self, name, value):
        if self.worker:
            self.worker.command(name, value)

    def _key_scale_change(self):
        self._send("key_scale", (self.key_var.get(), self.scale_var.get()))

    def _set_harmony(self, mode):
        self.harmony_var.set(mode)
        self._send("harmony", mode)
        self._sync_controls()

    def _toggle_preset(self):
        value = "classic" if self.preset_var.get() == "extreme" else "extreme"
        self.preset_var.set(value)
        self._send("preset", value)
        self._sync_controls()

    def _toggle_reverb(self):
        self.reverb_var.set(not self.reverb_var.get())
        self._send("reverb", self.reverb_var.get())
        self._sync_controls()

    def _toggle_drums(self):
        self.drums_var.set(not self.drums_var.get())
        self._send("drums", self.drums_var.get())
        self._sync_controls()

    def _toggle_bass(self):
        self.bass_var.set(not self.bass_var.get())
        self._send("bass", self.bass_var.get())
        self._sync_controls()

    def _toggle_guitar(self):
        self.guitar_var.set(not self.guitar_var.get())
        self._send("guitar", self.guitar_var.get())
        self._sync_controls()

    def _toggle_follow_voice(self):
        self.follow_voice_var.set(not self.follow_voice_var.get())
        self._send("follow_voice", self.follow_voice_var.get())
        self._sync_controls()

    def _set_octave(self, value):
        self.octave_var.set(value)
        self._send("voice_octave", value)
        self._sync_controls()

    def _set_bpm(self, value):
        value = max(60, min(180, int(value)))
        self.bpm_var.set(value)
        self._send("bpm", value)

    def _sync_controls(self):
        self.preset_btn.configure(text="EXTREME  /  ROBOT" if self.preset_var.get() == "extreme" else "CLASSIC  /  TUNED")
        self._update_button(self.preset_btn, self.preset_var.get() == "extreme", PINK)
        self._update_button(self.reverb_btn, self.reverb_var.get(), PINK)
        self._update_button(self.drums_btn, self.drums_var.get(), LIME)
        self._update_button(self.bass_btn, self.bass_var.get(), LIME)
        self._update_button(self.guitar_btn, self.guitar_var.get(), LIME)
        self._update_button(self.follow_btn, self.follow_voice_var.get(), CYAN)
        for mode, btn in self.harmony_buttons.items():
            self._update_button(btn, self.harmony_var.get() == mode, CYAN)
        for octave, btn in self.octave_buttons.items():
            self._update_button(btn, self.octave_var.get() == octave, ORANGE)
        self._update_button(self.chaos_btn, self.chaos_enabled, PINK)
        self.chaos_btn.configure(text="★  MAX CHAOS ON" if self.chaos_enabled else "★  MAX CHAOS")

    def start_audio(self):
        if self.worker:
            return
        mic, headphones = self.input_var.get(), self.output_var.get()
        if mic not in self.devices_in or headphones not in self.devices_out:
            messagebox.showwarning("Select devices", "Select a microphone and a headphone output first. Click ↻ Devices if necessary.")
            return
        if not self.headphones_var.get():
            messagebox.showwarning("Headphones required", "Plug in headphones, disable speaker monitoring, and check the safety confirmation.")
            return
        outname = headphones.lower()
        if any(term in outname for term in ("speakers", "loudspeaker", "hdmi", "television")):
            # Windows often labels a headphone jack as “Speakers (Realtek)”.
            # Do not reject a physically connected wired headset just because
            # the driver uses that name, but require a second explicit check.
            if not messagebox.askyesno(
                "Confirm headphone output",
                "Windows labels this device as a speaker/HDMI output. Some headphone "
                "jacks use that name. Are wired headphones physically plugged "
                "into this exact output, with speakers muted or disconnected?",
            ):
                return
        config = {
            "input": self.devices_in[mic], "output": self.devices_out[headphones],
            "stereo": self.stereo_var.get(), "key": self.key_var.get(), "scale": self.scale_var.get(),
            "preset": self.preset_var.get(), "reverb": self.reverb_var.get(),
            "reverb_mix": self.reverb_mix_var.get(), "harmony": self.harmony_var.get(),
            "harmony_level": self.harmony_level_var.get(), "drums": self.drums_var.get(),
            "bass": self.bass_var.get(), "guitar": self.guitar_var.get(),
            "follow_voice": self.follow_voice_var.get(), "guitar_level": self.guitar_level_var.get(),
            "follow_strength": self.follow_strength_var.get(),
            "drum_level": self.drum_level_var.get(), "bass_level": self.bass_level_var.get(),
            "band_level": self.band_level_var.get(),
            "bpm": self.bpm_var.get(), "groove": self.groove_var.get(),
            "voice_octave": self.octave_var.get(), "octave_mix": self.octave_mix_var.get(),
            "gain": self.gain_var.get(), "volume": self.volume_var.get(),
        }
        self.worker = AudioWorker(config)
        self._event_queues.append(self.worker.events)
        self.worker.start()
        self.status_var.set("STARTING  ·  Connecting audio...")
        self._transport_state(active=True)

    def _transport_state(self, active):
        self.start_btn.configure(state="disabled" if active else "normal")
        self.stop_btn.configure(state="normal" if active else "disabled")
        if not active:
            self.record_btn.configure(state="disabled", text="●  RECORD MP3", bg=INNER, fg=WHITE)
            self.recording = False
        state = "disabled" if active else "readonly"
        self.input_combo.configure(state=state)
        self.output_combo.configure(state=state)
        self.refresh_btn.configure(state="disabled" if active else "normal")
        self.stereo_check.configure(state="disabled" if active else "normal")

    def stop_audio(self):
        if self.worker:
            self.worker.stop()
            self.status_var.set("STOPPING  ·  Closing the audio stream...")
            self.stop_btn.configure(state="disabled")

    def _poll(self):
        if self.closing and (self.worker is None or not self.worker.thread.is_alive()):
            self.root.destroy()
            return
        for messages in self._event_queues:
            while True:
                try:
                    kind, info = messages.get_nowait()
                except queue.Empty:
                    break
                if kind in ("starting", "running"):
                    self.status_var.set(info)
                    self.status_label.configure(fg=LIME if kind == "running" else ORANGE)
                    if kind == "running" and self.worker:
                        self.record_btn.configure(state="normal")
                elif kind == "recording_started":
                    self.recording = True
                    self.record_btn.configure(text="■  STOP RECORDING", state="normal", bg=RED, fg=BG)
                    self.status_var.set("RECORDING  ·  " + Path(info).name)
                elif kind == "record_stopping":
                    self.recording = False
                    self.record_btn.configure(text="●  RECORD MP3", state="normal" if self.worker else "disabled",
                                              bg=INNER, fg=WHITE)
                    self.status_var.set(info)
                elif kind == "record_saved":
                    self.recording = False
                    self.status_var.set("MP3 saved  ·  " + info)
                elif kind == "record_error":
                    self.recording = False
                    self.record_btn.configure(text="●  RECORD MP3", state="normal" if self.worker else "disabled")
                    self.status_var.set("MP3 recording error")
                    messagebox.showerror("MP3 recording error", str(info)[-2200:])
                elif kind == "error":
                    # Show the actual exception, even when the PortAudio callback aborted.
                    self.status_var.set("ERROR  ·  Audio stream failed")
                    self.status_label.configure(fg=RED)
                    print(info)
                    messagebox.showerror("Audio error", info[-2300:])
                elif kind == "stopped":
                    self.status_var.set(info)
                    self.status_label.configure(fg=MUTED)
        if self.worker:
            last = None
            while True:
                try:
                    last = self.worker.meters.get_nowait()
                except queue.Empty:
                    break
            if last is not None:
                self.last_snapshot = last
                self._draw_visuals(last)
            if not self.worker.thread.is_alive():
                self.worker = None
                self._transport_state(active=False)
        if not self.closing:
            self.root.after(40, self._poll)
        else:
            self.root.after(70, self._poll)

    def _draw_visuals(self, snap):
        samples = snap["samples"]
        w = max(10, self.wave_canvas.winfo_width())
        h = max(10, self.wave_canvas.winfo_height())
        points = []
        waveform = samples[::8]
        for i, s in enumerate(waveform):
            points.extend((i * (w-1) / (len(waveform)-1), h/2 - float(s) * (h*0.45)))
        self.wave_canvas.coords(self.wave_line, *points)
        self.wave_canvas.tag_raise(self.wave_line)

        spec = np.abs(np.fft.rfft(samples * np.hanning(len(samples)))) / (len(samples)/2)
        frequencies = np.fft.rfftfreq(len(samples), 1.0 / SAMPLE_RATE)
        width = max(20, self.spectrum_canvas.winfo_width())
        height = max(10, self.spectrum_canvas.winfo_height())
        edges = np.geomspace(55, 8000, len(self.spectrum_bars) + 1)
        for i, bar in enumerate(self.spectrum_bars):
            lo = np.searchsorted(frequencies, edges[i], side="left")
            hi = max(lo + 1, np.searchsorted(frequencies, edges[i+1], side="right"))
            amplitude = float(np.max(spec[lo:hi])) if lo < len(spec) else 0.0
            db = 20 * math.log10(max(1e-6, amplitude))
            level = float(np.clip((db + 73) / 65, 0, 1))
            bw = width / len(self.spectrum_bars)
            x1 = i*bw + 2
            x2 = (i+1)*bw - 3
            y1 = height - max(2, level*(height-5))
            self.spectrum_canvas.coords(bar, x1, y1, x2, height)
            self.spectrum_canvas.itemconfigure(bar, fill=PINK if level > 0.85 else (LIME if i < 9 else CYAN))

        self.note_var.set(snap["note"])
        self.chord_var.set("Backing chord: " + snap.get("chord", "--") +
                           ("  • FOLLOW" if snap.get("follow_voice") else "  • FIXED"))
        self.pitch_var.set(f"Pitch confidence {snap['confidence']*100:.0f}%" if snap["note"] != "--" else "Waiting for vocals")
        for var, level, canvas, color in (
            (self.input_level_var, snap["input_db"], self.mic_meter, CYAN),
            (self.output_level_var, snap["output_db"], self.out_meter, PINK),
        ):
            var.set(("MIC" if color == CYAN else "OUT") + f"  {level:.0f} dBFS")
            canvas.delete("level")
            p = float(np.clip((level + 65) / 65, 0, 1))
            ww = canvas.winfo_width()
            canvas.create_rectangle(0, 0, ww*p, 15, fill=color, outline="", tags="level")
        for i, item in enumerate(self.beat_boxes):
            on = i == snap["beat"]
            item.configure(bg=LIME if on else INNER, fg=BG if on else MUTED)

    def _close(self):
        self.closing = True
        if self.worker:
            self.worker.stop()
        else:
            self.root.destroy()


def main():
    root = tk.Tk()
    StudioApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
