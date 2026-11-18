"""Non-blocking MP3 recording of the *monitored* audio bus.

The PortAudio callback only enqueues a bounded float32 chunk.  An independent
writer thread feeds ffmpeg's stdin and finalizes the MP3 on stop.  Neither
subprocess I/O nor file I/O is performed by the real-time callback.
"""
from __future__ import annotations

import queue
import subprocess
import threading
from pathlib import Path


class Mp3Recorder:
    """Record interleaved float32 audio at the sample rate of the output stream."""

    def __init__(self, destination, channels, events, sample_rate=48000,
                 executable=None, max_queue_blocks=512):
        self.destination = Path(destination).expanduser()
        self.channels = int(channels)
        self.sample_rate = int(sample_rate)
        self.events = events
        self.executable = executable
        self.queue = queue.Queue(maxsize=max_queue_blocks)
        self.stop_token = object()
        self.accepting = False
        self.dropped_blocks = 0
        self.written_bytes = 0
        self.thread = None
        self.process = None

    def start(self):
        if self.channels not in (1, 2):
            raise ValueError("MP3 recording requires one or two channels")
        if self.thread is not None:
            raise RuntimeError("Recorder was already started")
        if self.executable is None:
            from imageio_ffmpeg import get_ffmpeg_exe
            self.executable = get_ffmpeg_exe()
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        args = [str(self.executable), '-hide_banner', '-loglevel', 'error',
                '-nostdin', '-y', '-f', 'f32le', '-ar', str(self.sample_rate),
                '-ac', str(self.channels), '-i', 'pipe:0', '-vn',
                '-codec:a', 'libmp3lame', '-b:a', '192k',
                '-f', 'mp3', str(self.destination)]
        # DEVNULL avoids blocking on an unread stderr pipe. Detailed failures
        # include the return code and are surfaced in the UI.
        self.process = subprocess.Popen(args, stdin=subprocess.PIPE,
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
        self.accepting = True
        self.thread = threading.Thread(target=self._write, name='AutoTune-MP3',
                                       daemon=False)
        self.thread.start()

    def feed(self, output_block):
        """Call from the callback: never wait for the encoder."""
        if not self.accepting:
            return
        # Always use a copy; PortAudio reuses its output buffer next callback.
        payload = output_block.astype('<f4', copy=False).tobytes(order='C')
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            self.dropped_blocks += 1

    def stop(self):
        if not self.accepting:
            return
        self.accepting = False
        # Prefer finishing cleanly even when the encoder fell behind.  Only
        # discard a queued chunk when a full queue prevents the stop signal.
        try:
            self.queue.put_nowait(self.stop_token)
        except queue.Full:
            self.queue.get_nowait()
            self.dropped_blocks += 1
            self.queue.put_nowait(self.stop_token)

    def _write(self):
        problem = None
        try:
            while True:
                chunk = self.queue.get()
                if chunk is self.stop_token:
                    break
                self.process.stdin.write(chunk)
                self.written_bytes += len(chunk)
            self.process.stdin.close()
            code = self.process.wait(timeout=20)
            if code != 0:
                raise RuntimeError(f"MP3 encoder returned exit code {code}")
            if self.written_bytes == 0 or not self.destination.exists() or self.destination.stat().st_size == 0:
                raise RuntimeError("No audio was written to the MP3 file")
            if self.dropped_blocks:
                problem = (f"Recording saved with {self.dropped_blocks} dropped audio "
                           f"blocks (encoder was too slow): {self.destination}")
        except Exception as exc:
            problem = f"Could not finish MP3 recording at {self.destination}: {exc}"
            try:
                if self.process.stdin and not self.process.stdin.closed:
                    self.process.stdin.close()
            except Exception:
                pass
            if self.process.poll() is None:
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
        if problem:
            self.events.put(('record_error', problem))
        else:
            self.events.put(('record_saved', str(self.destination)))
