"""Portable, safe, versioned persistence of the GUI's last-used session.

The live audio callback must never load or write this file. The device names are
preferences, not authorization to connect audio: the GUI still requires a fresh
headphone safety confirmation on every launch.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import tempfile


VERSION = 1
# Name -> (type, minimum/choices, maximum). Not all numbers are arbitrary.
FIELDS = {
    "stereo": (bool, None, None),
    "key": (str, ("C", "C#", "DB", "D", "D#", "EB", "E", "F", "F#", "GB", "G", "G#", "AB", "A", "A#", "BB", "B"), None),
    "scale": (str, ("chromatic", "major", "minor", "pentatonic"), None),
    "preset": (str, ("classic", "extreme"), None),
    "reverb": (bool, None, None),
    "reverb_mix": (float, 0.0, 0.8),
    "harmony": (str, ("off", "duo", "choir", "wall"), None),
    "harmony_level": (float, 0.0, 1.5),
    "drums": (bool, None, None),
    "bass": (bool, None, None),
    "guitar": (bool, None, None),
    "follow_voice": (bool, None, None),
    "song_mode": (bool, None, None),
    "follow_strength": (float, 0.0, 1.0),
    "guitar_level": (float, 0.0, 1.0),
    "bass_level": (float, 0.0, 1.0),
    "drum_level": (float, 0.0, 1.0),
    "band_level": (float, 0.0, 1.0),
    "groove": (str, ("pop", "dance", "half-time"), None),
    "voice_octave": (int, -1, 1),
    "bpm": (int, 60, 180),
    "octave_mix": (float, 0.0, 0.6),
    "gain": (float, 0.3, 2.0),
    "volume": (float, 0.0, 1.0),
}


def settings_path():
    """User-writable stable path, independent of where the ZIP was extracted."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
    return base / "RetroAutoTune" / "last_session.json"


def valid_values(values):
    """Ignore corrupt/unknown fields. Never eval or accept nonfinite values."""
    if not isinstance(values, dict):
        return {}
    accepted = {}
    for name, value in values.items():
        if name not in FIELDS:
            continue
        typ, lo, hi = FIELDS[name]
        if typ is bool:
            if type(value) is not bool:
                continue
        elif typ is str:
            if not isinstance(value, str) or value not in lo:
                continue
        elif typ is int:
            if type(value) is not int or not lo <= value <= hi:
                continue
        else:
            if type(value) not in (float, int) or not math.isfinite(value) or not lo <= value <= hi:
                continue
            value = float(value)
        accepted[name] = value
    return accepted


def valid_device(identity):
    if not isinstance(identity, dict):
        return None
    name, host, occurrence = identity.get("name"), identity.get("hostapi"), identity.get("occurrence")
    if (not isinstance(name, str) or not name.strip() or len(name) > 350 or
            not isinstance(host, str) or not host.strip() or len(host) > 160 or
            type(occurrence) is not int or occurrence < 0 or occurrence > 100):
        return None
    return {"name": name, "hostapi": host, "occurrence": occurrence}


def device_match(preference, choices):
    """Find label matching (name, API, occurrence) when indices have shifted.

    choices is a dict of label -> stable identity. No silent best-effort device
    substitution: if a requested device is unplugged, return an empty selection.
    """
    if not preference:
        return ""
    return next((label for label, identity in choices.items()
                 if identity == preference), "")


def clean_session(raw):
    if not isinstance(raw, dict) or raw.get("version") != VERSION:
        return {}
    devices = raw.get("devices") if isinstance(raw.get("devices"), dict) else {}
    return {
        "version": VERSION,
        "settings": valid_values(raw.get("settings")),
        "devices": {"input": valid_device(devices.get("input")),
                    "output": valid_device(devices.get("output"))},
        "chaos_enabled": raw.get("chaos_enabled") is True,
        "before_chaos": valid_values(raw.get("before_chaos")),
    }


def load_session(path=None):
    try:
        data = json.loads(Path(path or settings_path()).read_text(encoding="utf-8"))
        return clean_session(data)
    except (OSError, ValueError, UnicodeError):
        return {}


def save_session(payload, path=None):
    target = Path(path or settings_path())
    data = clean_session(payload)
    if not data:
        raise ValueError("Invalid session settings payload")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=target.parent, prefix=".retro_settings_",
                                         suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
            json.dump(data, f, sort_keys=True, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return target
