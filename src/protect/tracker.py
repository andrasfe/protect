"""Locate the reference surface in every frame.

For each frame we estimate a homography H that maps reference-image pixels to
frame pixels:

1. SIFT features from the static parts of the reference (the ``track``
   regions) are matched against the frame and a homography is fitted with
   RANSAC.
2. The estimate is refined to sub-pixel accuracy with ECC image alignment on
   the area around the regions that will be blacked out. The ECC correlation
   coefficient (0..1) is kept as a per-frame confidence score.

Frames are independent, so they are processed in parallel. Temporal checks and
the fail-closed policy live in ``timeline``.
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

from . import video
from .geometry import box_quad, boxes_bbox, homography_is_plausible, project, translation
from .job import Job, Options

BATCH = 8


@dataclass
class FrameEstimate:
    H: np.ndarray | None
    inliers: int
    score: float


class Reference:
    """Everything a worker needs to localise the reference surface in a frame."""

    def __init__(self, ref_gray: np.ndarray, track: list, redact_boxes: list,
                 search, options: Options, refine=None):
        self.opt = options
        self.search = tuple(int(round(v)) for v in search) if search else None
        h, w = ref_gray.shape
        self.sift = cv2.SIFT_create(nfeatures=options.max_features)

        mask = np.zeros((h, w), np.uint8)
        for x, y, bw, bh in track:
            x0, y0 = max(0, int(x)), max(0, int(y))
            x1, y1 = min(w, int(np.ceil(x + bw))), min(h, int(np.ceil(y + bh)))
            mask[y0:y1, x0:x1] = 255
        ref_sift = cv2.SIFT_create()
        kps, des = ref_sift.detectAndCompute(ref_gray, mask)
        if des is None or len(kps) < options.min_inliers:
            raise ValueError(
                f"only {0 if des is None else len(kps)} features in the track regions; "
                "mark larger or more textured static areas")
        self.ref_pts = np.float32([k.pt for k in kps])
        self.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
        self.matcher.add([des])
        self.matcher.train()
        self.n_ref_features = len(kps)

        # plausibility is judged on the outline of the tracked area
        self.track_quad = box_quad(boxes_bbox(track))

        # Pixels outside the search area (letterbox bars, burnt-in borders) are not
        # picture content and must never be compared.
        ax0, ay0, ax1, ay1 = 0, 0, w, h
        if self.search:
            sx, sy, sw, sh = self.search
            ax0, ay0, ax1, ay1 = max(0, sx), max(0, sy), min(w, sx + sw), min(h, sy + sh)
        self.area = (ax0, ay0, ax1, ay1)
        self._valid_cache: dict = {}

        # refinement band: the explicit refine box, or the redact regions plus
        # context; clipped to the picture area
        if refine is not None:
            bx, by, bw, bh = refine
            m = 0.0
        else:
            bx, by, bw, bh = boxes_bbox(redact_boxes)
            m = options.refine_margin
        x0, y0 = max(ax0, int(bx - m)), max(ay0, int(by - m))
        x1, y1 = min(ax1, int(np.ceil(bx + bw + m))), min(ay1, int(np.ceil(by + bh + m)))
        if x1 - x0 < 16 or y1 - y0 < 16:
            raise ValueError("redact regions lie outside the reference image or search area")
        self.band = (x0, y0, x1 - x0, y1 - y0)
        area = (x1 - x0) * (y1 - y0)
        self.band_scale = min(1.0, float(np.sqrt(options.max_refine_pixels / area)))
        template = ref_gray[y0:y1, x0:x1]
        if self.band_scale < 1.0:
            template = cv2.resize(template, None, fx=self.band_scale, fy=self.band_scale,
                                  interpolation=cv2.INTER_AREA)
        self.template = template.astype(np.float32)

    # ------------------------------------------------------------------
    def estimate(self, gray: np.ndarray) -> FrameEstimate:
        H, inliers = self._match(gray)
        if H is None:
            return FrameEstimate(None, inliers, 0.0)
        H, score = self._refine(gray, H)
        if not homography_is_plausible(H, self.track_quad, self.opt.min_scale, self.opt.max_scale):
            return FrameEstimate(None, inliers, 0.0)
        return FrameEstimate(H, inliers, score)

    def _match(self, gray: np.ndarray):
        ox = oy = 0
        img = gray
        if self.search:
            sx, sy, sw, sh = self.search
            ox, oy = max(0, sx), max(0, sy)
            img = gray[oy:sy + sh, ox:sx + sw]
        kps, des = self.sift.detectAndCompute(img, None)
        if des is None or len(kps) < self.opt.min_inliers:
            return None, 0
        pairs = self.matcher.knnMatch(des, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < self.opt.ratio * p[1].distance]
        if len(good) < self.opt.min_inliers:
            return None, len(good)
        src = self.ref_pts[[g.trainIdx for g in good]]
        dst = np.float32([kps[g.queryIdx].pt for g in good]) + np.float32([ox, oy])
        H, inl = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, self.opt.ransac_px,
                                    confidence=0.999, maxIters=5000)
        n = int(inl.sum()) if inl is not None else 0
        if H is None or n < self.opt.min_inliers:
            return None, n
        if not homography_is_plausible(H, self.track_quad, self.opt.min_scale, self.opt.max_scale):
            return None, n
        return H, n

    def _valid_source(self, shape) -> np.ndarray:
        """255 where a frame of this shape has picture content (inside the search area)."""
        v = self._valid_cache.get(shape)
        if v is None:
            v = np.zeros(shape, np.uint8)
            if self.search:
                sx, sy, sw, sh = self.search
                v[max(0, sy):max(0, sy + sh), max(0, sx):max(0, sx + sw)] = 255
            else:
                v[:] = 255
            self._valid_cache[shape] = v
        return v

    def _refine(self, gray: np.ndarray, H: np.ndarray):
        """ECC-align the frame to the reference band. Returns (H, correlation)."""
        x0, y0, bw, bh = self.band
        s = self.band_scale
        # maps band pixels (scaled) -> frame pixels
        S = np.diag([1.0 / s, 1.0 / s, 1.0])
        M = H @ translation(x0, y0) @ S
        out_size = self.template.shape[1], self.template.shape[0]
        warped = cv2.warpPerspective(gray, M, out_size,
                                     flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid = cv2.warpPerspective(self._valid_source(gray.shape), M, out_size,
                                    flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if cv2.countNonZero(valid) < 0.25 * valid.size:
            # most of the band is outside the frame: nothing to refine against,
            # and nothing much to hide either. Keep the feature estimate.
            return H, _ncc(self.template, warped.astype(np.float32), valid)
        W = np.eye(3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)
        try:
            rho, W = cv2.findTransformECC(self.template, warped.astype(np.float32), W,
                                          cv2.MOTION_HOMOGRAPHY, criteria, valid, 3)
        except cv2.error:
            return H, _ncc(self.template, warped.astype(np.float32), valid)
        H2 = M @ W.astype(np.float64) @ np.linalg.inv(S) @ translation(-x0, -y0)
        H2 /= H2[2, 2]
        # ECC can wander off on low-texture frames; accept only small corrections
        corners = box_quad(self.band)
        shift = np.linalg.norm(project(H2, corners) - project(H, corners), axis=1).max()
        if not np.isfinite(shift) or shift > 4 * self.opt.ransac_px + 4:
            return H, _ncc(self.template, warped.astype(np.float32), valid)
        return H2, float(rho)


def _ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    m = mask > 0
    if m.sum() < 16:
        return 0.0
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / d) if d > 0 else 0.0


# ---------------------------------------------------------------------------
# parallel driver

_REF: Reference | None = None


def _init_worker(ref_gray, track, redact_boxes, search, options, refine):
    global _REF
    cv2.setNumThreads(1)
    _REF = Reference(ref_gray, track, redact_boxes, search, options, refine)


def _work(batch):
    start, frames = batch
    out = []
    for g in frames:
        e = _REF.estimate(g)
        out.append((e.H, e.inliers, e.score))
    return start, out


def load_reference_gray(job: Job) -> np.ndarray:
    img = cv2.imread(str(job.reference), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"cannot read reference image {job.reference}")
    return img


def track(job: Job, workers: int | None = None, progress=True) -> dict:
    """Estimate a homography for every frame and save it next to the job."""
    job.validate()
    info = video.probe(job.input)
    ref_gray = load_reference_gray(job)
    if ref_gray.shape != (info.height, info.width):
        raise ValueError(
            f"reference image is {ref_gray.shape[1]}x{ref_gray.shape[0]} but the video is "
            f"{info.width}x{info.height}; re-create it with 'protect init'")
    # build once here so configuration problems surface before spawning workers
    probe_ref = Reference(ref_gray, job.track, [r.box for r in job.redact], job.search,
                          job.options, job.refine)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)

    cap = info.frame_count + 64
    Hs = np.full((cap, 3, 3), np.nan)
    inliers = np.zeros(cap, np.int32)
    scores = np.zeros(cap, np.float32)

    t0 = time.time()
    n = 0
    pending: deque = deque()
    args = (ref_gray, job.track, [r.box for r in job.redact], job.search, job.options, job.refine)

    def collect(fut):
        start, res = fut.result()
        for k, (H, ni, sc) in enumerate(res):
            if H is not None:
                Hs[start + k] = H
            inliers[start + k] = ni
            scores[start + k] = sc

    def grow(need):
        nonlocal Hs, inliers, scores, cap
        if need <= cap:
            return
        extra = max(need - cap, 1024)
        Hs = np.concatenate([Hs, np.full((extra, 3, 3), np.nan)])
        inliers = np.concatenate([inliers, np.zeros(extra, np.int32)])
        scores = np.concatenate([scores, np.zeros(extra, np.float32)])
        cap += extra

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=args) as pool:
        batch, start = [], 0
        for g in video.iter_frames(job.input, info, gray=True):
            batch.append(g)
            if len(batch) == BATCH:
                pending.append(pool.submit(_work, (start, batch)))
                start += len(batch)
                grow(start)
                batch = []
                while len(pending) >= 3 * workers:
                    collect(pending.popleft())
                if progress and start % (BATCH * 64) == 0:
                    _progress(start, info.frame_count, t0)
        if batch:
            pending.append(pool.submit(_work, (start, batch)))
            start += len(batch)
            grow(start)
        while pending:
            collect(pending.popleft())
        n = start
    if progress:
        _progress(n, n, t0, end="\n")

    result = {
        "H": Hs[:n], "inliers": inliers[:n], "score": scores[:n],
        "frame_count": np.int64(n), "width": np.int64(info.width), "height": np.int64(info.height),
        "fingerprint": np.str_(job.tracking_fingerprint()),
        "n_ref_features": np.int64(probe_ref.n_ref_features),
    }
    np.savez_compressed(job.track_path, **result)
    return result


def load_track(job: Job, check: bool = True) -> dict:
    if not job.track_path.exists():
        raise FileNotFoundError(f"no tracking data at {job.track_path}; run 'protect track' first")
    with np.load(job.track_path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    if check and str(data["fingerprint"]) != job.tracking_fingerprint():
        raise ValueError("tracking data is out of date for this job (video, reference, regions "
                         "or tracking options changed); run 'protect track' again")
    return data


def _progress(done: int, total: int, t0: float, end: str = "") -> None:
    el = time.time() - t0
    fps = done / el if el > 0 else 0.0
    eta = (total - done) / fps if fps > 0 and total > done else 0.0
    sys.stderr.write(f"\r  tracked {done}/{total} frames  {fps:5.1f} fps  eta {eta / 60:4.1f} min   {end}")
    sys.stderr.flush()
