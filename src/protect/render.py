"""Apply a redaction plan to the video and encode the result."""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import video
from .job import Job
from .timeline import BLACKOUT, LOST, Plan

FILL = (0, 0, 0)


def frame_polygons(plan: Plan, i: int, size: tuple[int, int]) -> list[np.ndarray]:
    """Integer polygons for frame i, clipped to a sane coordinate range."""
    lim = 4 * max(size)
    return [np.clip(np.round(p), -lim, lim).astype(np.int32) for p in plan.polys[i]]


def apply(frame: np.ndarray, plan: Plan, i: int) -> None:
    """Black out frame i in place according to the plan."""
    st = plan.status[i]
    if st == LOST or st == BLACKOUT:
        frame[:] = 0
        return
    polys = frame_polygons(plan, i, (frame.shape[1], frame.shape[0]))
    if not polys:
        return
    r = int(math.ceil(plan.pad[i]))
    cv2.fillPoly(frame, polys, FILL)
    if r > 0:
        # thick outlines grow each polygon by r pixels on every side
        cv2.polylines(frame, polys, True, FILL, thickness=2 * r + 1, lineType=cv2.LINE_8)


def render(job: Job, plan: Plan, output: Path | None = None, progress: bool = True) -> Path:
    info = video.probe(job.input)
    output = Path(output or job.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.stem + ".partial" + (output.suffix or ".mp4"))
    writer = video.VideoWriter(partial, info.size, info.fps, crf=job.options.crf,
                               preset=job.options.preset,
                               audio_from=job.input if job.audio else None)
    t0 = time.time()
    i = 0
    ok = False
    try:
        for frame in video.iter_frames(job.input, info):
            if i >= len(plan):
                raise video.VideoError("video has more frames than the tracking data; re-run 'protect track'")
            frame = frame.copy() if not frame.flags.writeable else frame
            apply(frame, plan, i)
            writer.write(frame)
            i += 1
            if progress and i % 500 == 0:
                _progress(i, len(plan), t0)
        if i != len(plan):
            raise video.VideoError(f"decoded {i} frames but tracking data has {len(plan)}; re-run 'protect track'")
        writer.close()
        ok = True
    finally:
        if not ok:
            writer.abort()
    if progress:
        _progress(i, len(plan), t0, end="\n")
    os.replace(partial, output)
    return output


def _progress(done: int, total: int, t0: float, end: str = "") -> None:
    el = time.time() - t0
    fps = done / el if el > 0 else 0.0
    eta = (total - done) / fps if fps > 0 and total > done else 0.0
    sys.stderr.write(f"\r  rendered {done}/{total} frames  {fps:5.1f} fps  eta {eta / 60:4.1f} min   {end}")
    sys.stderr.flush()
