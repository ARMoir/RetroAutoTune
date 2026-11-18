# RetroAutoTune MicStation — Hardware Roadmap

**Status:** Long-term aspirational goal — no Raspberry Pi hardware build has been validated.

## Vision

Build a self-contained, 3D-printed microphone-based RetroAutoTune system with an integrated touchscreen and minimal physical controls. A performer plugs in a microphone/headset, powers on the device, taps **MAX CHAOS**, sings, and hears an entire synthesized, voice-reactive backing ensemble without needing a Windows computer. The finished product should preserve the playful extreme vocal effects and intuitive one-touch operation that define RetroAutoTune.

Candidate form factors:
- **Microphone enclosure:** electronics and a compact touchscreen integrated around a microphone body/stand or microphone-style housing.
- **Mic + small control pod:** separate electronics/touchscreen housing if enclosure space, cable strain, heat, microphone noise isolation or weight make an all-in-one microphone impractical.

The final mechanical design will be chosen after prototype measurements, not before.

## Design commitments for all future software updates

1. **Portable engine:** DSP, real-time state, note/chord tracking, band synthesis and MP3 recording should not depend on Tkinter, Windows APIs or GUI event-loop timing. Keep a reusable core.
2. **Portable audio routing:** Keep device input/output abstract enough for Windows PortAudio today and Linux ALSA/PipeWire/PortAudio on a future Pi. Do not hard-code Windows device IDs, host API names or paths.
3. **Touch-first UI compatibility:** Use responsive layouts and big controls, support compact displays and screen rotations, and avoid requiring a keyboard or mouse for core features.
4. **Performance discipline:** Avoid allocations, file I/O, FFmpeg encoding, screen painting, disk writes or network calls in the real-time audio callback. Permit optional visual effects to degrade gracefully to protect audio.
5. **Safety:** Default to wired headphones/controlled line output at safe levels; preserve explicit output selection and avoid acoustic feedback loops. Software confirmation is not a hardware feedback interlock.
6. **Do not regress desktop:** Windows GUI and console usage remain supported while experimental hardware-specific work is optional and isolated.

## Proposed reference hardware (candidates, not a bill of materials)

| Component | Initial direction | Validation needed |
| --- | --- | --- |
| Compute | Raspberry Pi 5 or comparable ARM SBC | CPU headroom at 48 kHz with MAX CHAOS, thermal throttling, size, power |
| Screen | 3.5–5 inch HDMI/DSI touch display | Linux drivers, orientation, touch targets, readability at arm's length |
| Microphone | Low-noise wired microphone/capsule | Mounting, proximity noise, self-noise, shielding, gain |
| Audio I/O | Linux-compatible USB audio interface or codec board | Stable full-duplex low-latency input/output, 48 kHz, headphone drive |
| Controls | Large physical MAX CHAOS button; record/stop; rotary volume | Debouncing, accessibility, safe defaults |
| Enclosure | Custom 3D print with removable service panels | Cooling, fastening, cable strain relief, weight, physical mic isolation |
| Power | Safe external power first; optional battery later | Peak Pi/USB/screen draw, shutdown safety, battery charging design |
| Storage | microSD or solid-state storage | Recording capacity, write reliability, easy file retrieval |

No parts should be purchased or enclosure dimensions finalized based on this table alone.

## Phased milestones

### Phase 1 — Software portability audit
- Identify Windows-specific assumptions in device enumeration, GUI startup, packaging, filesystem handling and MP3 encoding.
- Refactor the DSP/band engine behind a stable control API callable with or without a GUI.
- Preserve existing Windows behavior and add automated tests for engine functions, configuration serialization, and headless streaming simulation.
- Keep touch-friendly UI foundations such as MAX CHAOS, follow strength, levels and recording status.

**Exit:** DSP and control functions import and run on Linux/ARM-compatible Python without Tkinter or Windows assumptions; desktop regression tests pass.

### Phase 2 — Raspberry Pi audio proof of concept
- Test Raspberry Pi hardware with a wired mic and headphone interface, preferably using a single stable full-duplex device.
- Measure real end-to-end microphone-to-headphone latency at classic and MAX CHAOS settings; record block size, device, OS, CPU utilization and xruns/dropouts.
- Benchmark synthesized backing vocals, guitar, bass, drums, reverb and MP3 recording simultaneously.
- Profile and optimize hotspots before reducing audio quality or features.

**Exit:** Reproducible live sound and recorded performance on the chosen Pi with acceptable, *measured* monitoring latency and no sustained xruns in a multi-minute stress test. A sub-50 ms round-trip monitoring latency is a stretch target, not a promise.

### Phase 3 — Touchscreen / appliance mode
- Provide a minimal performance view with large controls: Start/Stop, MAX CHAOS, Record, effects, Voice Follow intensity, safe master volume, current pitch and input/output meters.
- Test common compact resolutions (e.g. 800×480 and 480×320); prioritize readable screens, scrolling/secondary pages and larger touch targets rather than tiny 3-column controls.
- Add on-device startup, audio device recovery and a clear error screen that does not require opening a terminal.
- Keep advanced editing available through an optional detailed view or desktop connection.

**Exit:** Core performance workflow works entirely by touch, including file save/recording indicator and clean shutdown.

### Phase 4 — Electronics and 3D-printed enclosure
- Measure assembled components, mic vibration isolation, enclosure acoustic coupling, airflow, thermal conditions and connector accessibility.
- Prototype control placement and screen viewing angle before a finished print.
- Add physical controls and status lighting if they improve usability.
- Keep ventilation, low-voltage power safety, removable panels and cable strain relief as explicit constraints.

**Exit:** Reliable bench prototype that remains comfortable, serviceable and cool enough during extended MAX CHAOS sessions, with no unacceptable mechanical noise in the mic.

### Phase 5 — Self-contained polish
- Power-on to usable performance UI; guided device setup; reliable saving and retrieval of MP3s.
- Optional battery capability only after thermal, power and safe charging work are validated.
- Produce build docs, printable enclosure CAD/STL files, a verified parts list, setup image/install procedure, and measured performance results.

**Exit:** Repeatable assembly and setup from documented components, tested on real hardware.

## Known risks and open decisions

- **Latency:** Time-domain pitch correction, harmonies, and Linux audio buffering may exceed the comfortable monitoring threshold. Test rather than assuming PC behavior carries over.
- **CPU and thermal constraints:** Eight-wall choir, MAX CHAOS and FFT visuals might overwhelm a compact enclosed Pi; prioritize uninterrupted audio.
- **Microphone feedback and isolation:** A speaker next to the capsule is especially risky. Design for headphones or safely isolated external amplification. The software cannot guarantee feedback suppression.
- **Recording:** ARM-compatible FFmpeg/MP3 encoding availability and microSD wear require testing; encoding must remain asynchronous.
- **Form factor:** A literal all-in-microphone housing could be too heavy or noisy; the mic-plus-control-pod layout remains a fallback.
- **Hardware selection:** No final Pi, touchscreen, codec or power design has been selected.

## Feature acceptance checklist

- [ ] Runs live Auto-Tune on Linux/ARM hardware without a PC
- [ ] Preserves Extreme mode, octave control, reverb, WALL harmonies and voice-responsive guitar/bass/drums
- [ ] MAX CHAOS usable as a single large touch/physical control
- [ ] Voice Follow intensity adjustable by touch without interrupting audio
- [ ] MP3 recording captures the complete processed mix without callback glitches
- [ ] Small-screen responsive UI works without mouse/keyboard
- [ ] Audio stream recovers intelligibly from device errors
- [ ] Wired monitoring and safe output behavior documented and tested
- [ ] Measured Pi latency, stability, CPU and thermal results published
- [ ] 3D-printable enclosure design and reproducible build guide published

## Release priority

This remains a **long-term direction**, not a commitment that current versions run on Raspberry Pi. First keep RetroAutoTune delightful and reliable on Windows; make compatibility-conscious decisions so the hardware edition becomes a practical extension rather than a rewrite.
