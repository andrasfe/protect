"""Small geometry helpers: boxes, homographies and polygon sanity checks."""

from __future__ import annotations

import numpy as np


def box_quad(box, pad: float = 0.0) -> np.ndarray:
    """Corners of an (x, y, w, h) box grown by ``pad`` on every side, clockwise."""
    x, y, w, h = (float(v) for v in box)
    x0, y0, x1, y1 = x - pad, y - pad, x + w + pad, y + h + pad
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to an (N, 2) array of points."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    w = hom[:, 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = hom[:, :2] / w
    out[~np.isfinite(out)] = np.nan
    return out


def polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def is_convex(poly: np.ndarray) -> bool:
    d = np.roll(poly, -1, axis=0) - poly
    cross = d[:, 0] * np.roll(d[:, 1], -1) - d[:, 1] * np.roll(d[:, 0], -1)
    return bool(np.all(cross > 0) or np.all(cross < 0))


def homography_is_plausible(H: np.ndarray, quad: np.ndarray,
                            min_scale: float = 0.1, max_scale: float = 10.0) -> bool:
    """Reject homographies that fold, flip or wildly rescale ``quad``."""
    if H is None or not np.all(np.isfinite(H)):
        return False
    hom = np.hstack([quad, np.ones((len(quad), 1))]) @ H.T
    if np.any(hom[:, 2] <= 1e-9):          # a corner maps behind the camera
        return False
    proj = hom[:, :2] / hom[:, 2:3]
    if not is_convex(proj):
        return False
    a0, a1 = polygon_area(quad), polygon_area(proj)
    if a0 == 0 or np.sign(a0) != np.sign(a1):   # mirrored
        return False
    ratio = abs(a1 / a0)
    return min_scale ** 2 <= ratio <= max_scale ** 2


def boxes_bbox(boxes) -> tuple[float, float, float, float]:
    """Bounding (x, y, w, h) of several (x, y, w, h) boxes."""
    arr = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    x0, y0 = arr[:, 0].min(), arr[:, 1].min()
    x1, y1 = (arr[:, 0] + arr[:, 2]).max(), (arr[:, 1] + arr[:, 3]).max()
    return x0, y0, x1 - x0, y1 - y0


def translation(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])
