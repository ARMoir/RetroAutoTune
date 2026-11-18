# Retro Auto-Tune v4 — Live Visual Studio

**A neon, late-2000s-style graphical interface for the v3 Auto-Tune audio engine.** The original console version is still here, unchanged. The GUI has responsive live controls and animated audio visuals.

## Get started on Windows

1. Plug in **wired headphones** before starting; begin at a low Windows volume. Disable Windows **Listen to this device** and hardware mic sidetone/direct monitoring if they create an echo.
2. Extract the ZIP, keeping the files in the same folder.
3. Double-click **`INSTALL.bat`** once (or run `py -m pip install -r requirements.txt` in PowerShell).
4. Double-click **`START_GUI.bat`** (or run `py retro_autotune_gui.py`).
5. Choose your **microphone** and **headphone output** from the dropdowns. Windows can label a headphone jack as “Speakers”; if so, the app asks you to confirm the physical headphone connection.
6. Check **“I have plugged in headphones”**. Press **START LIVE**, sing, and change effects at any time. Press **STOP** before unplugging your headphones.

If `py` is not recognized, install Python 3.11+ and make sure the Windows Python launcher is available. Required third-party packages: `numpy`, `sounddevice`. Tkinter comes with most python.org Windows Python installers. USB headsets, Bluetooth headphones, and device combinations can have different latencies; **wired headphones** are recommended.

## Visuals and live controls

- **Waveform oscilloscope:** real-time tuned headphone-output waveform.
- **Sound spectrum:** animated logarithmic audio-energy bars from approximately 50 Hz to 8 kHz.
- **Pitch lock:** the corrected target note (for example, `C4`) and detector confidence.
- **Mic and output levels:** separate dBFS meters.
- **Beat indicators:** four-step 4/4 display follows the backing band tempo.
- **Control panel:** musical key/scale, Classic/Extreme robotic Auto-Tune, lead octave down/normal/up, robot upper-octave layer, harmony off/duo/choir, Cher-inspired plate/slap reverb, drums, bass, groove, BPM, and volume/gain sliders.

All sound controls (including scale/key changes) can be changed while audio is running. **Device selection and mono/stereo output must be changed while stopped.** The GUI defaults to **mono output** to preserve the working v1/v2.1/v3 hardware compatibility; select Stereo only if your output device supports it. Stereo is needed to hear the width of the reverb.

The GUI uses Tkinter, no extra GUI/framework dependencies. Spectral FFTs, waveform rendering, and UI events run on the GUI thread, **not in the PortAudio callback**. The audio stream runs on a worker thread with a bounded, drop-on-full meter queue. The output monitoring volume defaults to 55% of the engine's output to allow safer initial testing; adjust it as needed.

## Troubleshooting

- **No devices listed:** plug in headphones, click **↻ Devices**; confirm Windows recognizes the devices. Use `py retro_autotune.py --list` to see the device numbers.
- **Startup failure or instant stop:** choose physically connected devices, confirm the output jack, and check the error dialog. If one device is unplugged mid-stream, stop/restart after reconnecting.
- **Sounds doubled or feeds back:** do **not** monitor via speakers; use headphones and disable Windows/hardware mic monitoring. The program does not implement acoustic echo cancellation, so software alone cannot eliminate feedback.
- **Glitches or excessive latency:** start with mono output, turn off Choir for comparison, close other heavy audio applications, and use a wired USB audio device if available. Real-time pitch shifting is CPU- and audio-driver-dependent.
- **Pitch visuals show `--`:** the microphone signal may be gated or the note detection confidence is too low. Sing a sustained note and check the MIC meter.
- **Still want CLI controls?** The unmodified v3 console program is included. Run `py retro_autotune.py --list` then your working original command (with `--headphones`).

## Test commands

```powershell
py retro_autotune.py --selftest
py -m unittest -v test_gui
```

The tests exercise synthetic pitch detection, vocal transpose, backing instrument sequencing, reverb, controls and a **simulated** audio callback. They do **not** validate Windows hardware or subjective microphone sound quality. A GUI smoke test was also completed under a virtual desktop; real headphone playback needs to be checked on your computer.

## Notes

The backing band follows the manually selected BPM and repeats a four-chord pattern; it does not detect the tempo/chords of your singing. The harmony voices follow the detected vocal note. The effects are original experiments inspired by the sound of heavily processed late-2000s pop vocals, not recordings or clones of a particular artist. At large pitch intervals the grain shifter can have audible artifacts.