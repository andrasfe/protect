"""Thin wrappers around ffmpeg/ffprobe for frame-exact decoding and encoding."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import numpy as np


class VideoError(RuntimeError):
    pass


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise VideoError(f"{tool} not found on PATH; install ffmpeg first")


@dataclass(frozen=True)
class VideoInfo:
    width: int          # display width (after rotation metadata is applied)
    height: int
    fps: Fraction       # nominal frame rate used when re-encoding
    frame_count: int    # from the container; may be approximate, decoding is authoritative
    duration: float
    has_audio: bool

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height


def probe(path: str | os.PathLike) -> VideoInfo:
    require_ffmpeg()
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_streams", "-show_format", str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise VideoError(f"ffprobe failed on {path}: {res.stderr.strip()}")
    data = json.loads(res.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise VideoError(f"no video stream in {path}")
    width, height = int(video["width"]), int(video["height"])
    rotation = 0
    for side in video.get("side_data_list", []) or []:
        if "rotation" in side:
            rotation = int(round(float(side["rotation"])))
    if abs(rotation) % 180 == 90:
        width, height = height, width
    fps = Fraction(video.get("avg_frame_rate") or "0/1")
    if fps <= 0:
        fps = Fraction(video.get("r_frame_rate") or "30/1")
    duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0.0)
    if video.get("nb_frames"):
        frame_count = int(video["nb_frames"])
    else:
        frame_count = int(round(duration * float(fps)))
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    return VideoInfo(width, height, fps, frame_count, duration, has_audio)


def _read_exact(stream, n: int) -> bytearray | None:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = stream.readinto(view[got:])
        if not k:
            break
        got += k
    if got == 0:
        return None
    if got != n:
        raise VideoError(f"truncated frame from decoder ({got} of {n} bytes)")
    return buf


def iter_frames(path: str | os.PathLike, info: VideoInfo | None = None,
                gray: bool = False) -> Iterator[np.ndarray]:
    """Yield every decoded frame in order, as BGR (H, W, 3) or gray (H, W) uint8.

    Frames are passed through without frame-rate conversion, so the i-th
    yielded frame is the i-th frame of the stream in every call.
    """
    require_ffmpeg()
    info = info or probe(path)
    pix_fmt, channels = ("gray", 1) if gray else ("bgr24", 3)
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
        "-map", "0:v:0", "-fps_mode", "passthrough",
        "-f", "rawvideo", "-pix_fmt", pix_fmt, "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            bufsize=0)
    shape = (info.height, info.width) if gray else (info.height, info.width, channels)
    n = info.width * info.height * channels
    completed = False
    try:
        while True:
            buf = _read_exact(proc.stdout, n)
            if buf is None:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(shape)
        completed = True
    finally:
        if not completed:
            proc.kill()
        proc.stdout.close()
        err = proc.stderr.read().decode(errors="replace")
        proc.stderr.close()
        code = proc.wait()
        if completed and code != 0:
            raise VideoError(f"ffmpeg decode failed for {path}: {err.strip()}")


def extract_frame(path: str | os.PathLike, time_s: float,
                  info: VideoInfo | None = None) -> np.ndarray:
    """Return the BGR frame shown at ``time_s`` seconds."""
    require_ffmpeg()
    info = info or probe(path)
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-ss", f"{time_s:.6f}", "-i", str(path),
        "-map", "0:v:0", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    res = subprocess.run(cmd, capture_output=True)
    n = info.width * info.height * 3
    if res.returncode != 0 or len(res.stdout) < n:
        raise VideoError(f"could not extract frame at {time_s}s: {res.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(res.stdout[:n], dtype=np.uint8).reshape(info.height, info.width, 3).copy()


class VideoWriter:
    """Encode BGR frames to H.264 with ffmpeg.

    Only the frames written here end up in the output: no stream from the
    source is copied unless ``audio_from`` is given, and container metadata
    (titles, dates, locations, chapters) is dropped.
    """

    def __init__(self, path: str | os.PathLike, size: tuple[int, int], fps: Fraction,
                 crf: int = 18, preset: str = "medium",
                 audio_from: str | os.PathLike | None = None):
        require_ffmpeg()
        self.path = Path(path)
        self.size = size
        w, h = size
        cmd = [
            "ffmpeg", "-v", "error", "-nostdin", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
            "-r", f"{fps.numerator}/{fps.denominator}", "-i", "-",
        ]
        if audio_from is not None:
            cmd += ["-i", str(audio_from), "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy"]
        else:
            cmd += ["-map", "0:v:0", "-an"]
        cmd += [
            "-sn", "-dn",
            "-map_metadata", "-1", "-map_chapters", "-1",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(self.path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.frames = 0

    def write(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if (w, h) != self.size or frame.ndim != 3 or frame.dtype != np.uint8:
            raise VideoError("frame does not match writer size/format")
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).data)
        except BrokenPipeError as exc:
            err = self._proc.stderr.read().decode(errors="replace")
            raise VideoError(f"encoder exited early: {err.strip()}") from exc
        self.frames += 1

    def close(self) -> None:
        self._proc.stdin.close()
        err = self._proc.stderr.read().decode(errors="replace")
        self._proc.stderr.close()
        if self._proc.wait() != 0:
            raise VideoError(f"encoder failed: {err.strip()}")

    def abort(self) -> None:
        try:
            self._proc.stdin.close()
        except OSError:
            pass
        self._proc.kill()
        self._proc.wait()
        self.path.unlink(missing_ok=True)


def container_tags(path: str | os.PathLike) -> dict:
    """Return format-level and stream-level tags, for checking metadata was stripped."""
    cmd = ["ffprobe", "-v", "error", "-print_format", "json",
           "-show_entries", "format_tags:stream_tags:stream=codec_type", str(path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise VideoError(res.stderr.strip())
    return json.loads(res.stdout)
