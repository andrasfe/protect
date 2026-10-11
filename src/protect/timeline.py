"""Turn per-frame tracking results into a per-frame redaction plan.

Every frame gets exactly one status:

OK        tracked with confidence; the regions are blacked out where tracking put them
FILLED    not trusted on its own, but close to tracked frames; the regions are blacked
          out using the hull of the neighbouring positions plus growing padding
KEPT      untracked, inside a ``keep`` range the user reviewed; left as is
LOST      untracked and not reviewed; the whole frame is blacked out
BLACKOUT  inside a ``blackout`` range; the whole frame is blacked out

The default for anything uncertain is to hide more, never less.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import box_quad, project
from .job import Job

OK, FILLED, KEPT, LOST, BLACKOUT = range(5)
STATUS_NAMES = ("ok", "filled", "kept", "lost", "blackout")
JUMP_WINDOW = 5


@dataclass
class Plan:
    status: np.ndarray            # (n,) int8
    polys: list                   # per frame: list of (k, 2) float polygons in frame pixels
    pad: np.ndarray               # (n,) float, extra frame-space padding per frame
    score: np.ndarray             # (n,) tracking confidence, 0 when untracked
    fps: float

    def __len__(self) -> int:
        return len(self.status)

    def counts(self) -> dict:
        return {name: int((self.status == k).sum()) for k, name in enumerate(STATUS_NAMES)}

    def runs(self, statuses) -> list[tuple[int, int]]:
        """Inclusive (first, last) frame runs whose status is in ``statuses``."""
        sel = np.isin(self.status, list(statuses))
        out, i, n = [], 0, len(sel)
        while i < n:
            if sel[i]:
                j = i
                while j + 1 < n and sel[j + 1]:
                    j += 1
                out.append((i, j))
                i = j + 1
            else:
                i += 1
        return out

    def bbox(self, i: int):
        """Bounding box (x0, y0, x1, y1) of frame i's regions, or None."""
        if not self.polys[i]:
            return None
        pts = np.vstack(self.polys[i])
        r = self.pad[i]
        return (pts[:, 0].min() - r, pts[:, 1].min() - r, pts[:, 0].max() + r, pts[:, 1].max() + r)


def _in_ranges(t: float, ranges) -> bool:
    return any(s <= t <= e for s, e in ranges)


def _nearest_good(good: np.ndarray):
    n = len(good)
    prev = np.full(n, -10**9, dtype=np.int64)
    nxt = np.full(n, 10**9, dtype=np.int64)
    last = -10**9
    for i in range(n):
        if good[i]:
            last = i
        prev[i] = last
    last = 10**9
    for i in range(n - 1, -1, -1):
        if good[i]:
            last = i
        nxt[i] = last
    return prev, nxt


def _reject_jumps(corners: np.ndarray, good: np.ndarray, max_jump: float) -> np.ndarray:
    """Drop tracked frames that disagree with their tracked neighbours."""
    good = good.copy()
    for _ in range(2):
        idx = np.flatnonzero(good)
        bad = []
        for k in range(1, len(idx) - 1):
            p, i, q = idx[k - 1], idx[k], idx[k + 1]
            if i - p > JUMP_WINDOW or q - i > JUMP_WINDOW:
                continue
            pred = corners[p] + (corners[q] - corners[p]) * ((i - p) / (q - p))
            if np.linalg.norm(corners[i] - pred, axis=1).max() > max_jump:
                bad.append(i)
        if not bad:
            break
        good[bad] = False
    return good


def _velocity(corners: np.ndarray, good: np.ndarray, s: int, direction: int) -> np.ndarray:
    """Per-corner motion per frame at the edge of a tracked run, pointing into the gap.

    ``direction`` is +1 when the gap follows frame ``s`` and -1 when it precedes it.
    """
    # nearest tracked frame on the side away from the gap
    for r in range(s - direction, s - direction * (JUMP_WINDOW + 1), -direction):
        if 0 <= r < len(good) and good[r]:
            return (corners[s] - corners[r]) / abs(s - r)
    return np.zeros_like(corners[s])


def build_plan(job: Job, track: dict, fps: float) -> Plan:
    opt = job.options
    H = track["H"]
    score = track["score"].astype(np.float32)
    inliers = track["inliers"]
    n = len(H)
    quads = [box_quad(r.box, job.padding) for r in job.redact]
    k = len(quads)
    allpts = np.vstack(quads)

    has_h = np.isfinite(H).all(axis=(1, 2)) & (inliers >= opt.min_inliers)
    corners = np.full((n, 4 * k, 2), np.nan)
    for i in np.flatnonzero(has_h):
        corners[i] = project(H[i], allpts)
    has_h &= np.isfinite(corners).all(axis=(1, 2))
    good = has_h & (score >= opt.min_score)
    # The alignment score is measured around the regions, so it is meaningless when they
    # are out of view. Such frames are trusted if the feature match is strong and the
    # regions sit well outside the frame: there is nothing on screen to hide.
    w, h = float(track["width"]), float(track["height"])
    m = opt.offscreen_margin
    outside = np.zeros(n, bool)
    idx = np.flatnonzero(has_h)
    if len(idx):
        xs, ys = corners[idx, :, 0], corners[idx, :, 1]
        outside[idx] = ((xs.max(axis=1) < -m) | (xs.min(axis=1) > w + m)
                        | (ys.max(axis=1) < -m) | (ys.min(axis=1) > h + m))
    good |= has_h & (inliers >= opt.offscreen_min_inliers) & outside
    good = _reject_jumps(corners, good, opt.max_jump)
    prev, nxt = _nearest_good(good)

    status = np.zeros(n, np.int8)
    polys: list = [[] for _ in range(n)]
    pad = np.zeros(n, np.float32)
    for i in range(n):
        t = i / fps
        if _in_ranges(t, job.blackout):
            status[i] = BLACKOUT
            continue
        if good[i]:
            status[i] = OK
            polys[i] = [corners[i, 4 * j:4 * j + 4] for j in range(k)]
            pad[i] = opt.frame_padding
            continue
        dp, dn = i - prev[i], nxt[i] - i
        sides = [s for s, d in ((prev[i], dp), (nxt[i], dn)) if d <= opt.max_gap]
        if sides:
            status[i] = FILLED
            if len(sides) == 2:
                # hull of both neighbours covers any motion between them
                anchors = [corners[sides[0]], corners[sides[1]]]
                d = min(dp, dn)
            else:
                # one neighbour: also cover where it was heading at its last speed
                s = sides[0]
                d = dp if s == prev[i] else dn
                v = _velocity(corners, good, s, direction=1 if s == prev[i] else -1)
                anchors = [corners[s], corners[s] + v * d]
            for j in range(k):
                pts = np.vstack([a[4 * j:4 * j + 4] for a in anchors]).astype(np.float32)
                polys[i].append(cv2.convexHull(pts).reshape(-1, 2).astype(np.float64))
            pad[i] = opt.frame_padding + opt.gap_padding + opt.gap_growth * d
        elif _in_ranges(t, job.keep):
            status[i] = KEPT
        else:
            status[i] = LOST
    return Plan(status, polys, pad, np.where(good, score, 0.0).astype(np.float32), fps)
