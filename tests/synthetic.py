"""Synthetic footage: a textured panel moving over a background under known homographies."""

from __future__ import annotations

import math

import cv2
import numpy as np

FRAME_W, FRAME_H = 960, 540
PANEL_W, PANEL_H = 560, 340
# where the panel sits in the reference frame (frame 0)
PANEL_ORIGIN = (200.0, 100.0)
# the line that must be hidden, in panel coordinates
SECRET_BOX_PANEL = (40, 52, 300, 26)

WORDS = ["ALPHA", "BRAVO", "DELTA", "OMEGA", "SIGMA", "KAPPA", "THETA", "GAMMA"]


def make_panel(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = np.full((PANEL_H, PANEL_W, 3), (40, 30, 25), np.uint8)
    cv2.rectangle(p, (0, 0), (PANEL_W - 1, PANEL_H - 1), (200, 200, 200), 6)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(p, "HEADER LINE ONE", (40, 40), font, 0.7, (255, 200, 80), 2, cv2.LINE_AA)
    cv2.putText(p, "SAMPLE PERSON 4471", (40, 72), font, 0.7, (255, 200, 80), 2, cv2.LINE_AA)
    for row in range(5):
        y = 110 + row * 26
        for col in range(3):
            w = WORDS[int(rng.integers(len(WORDS)))]
            n = int(rng.integers(10, 99))
            cv2.putText(p, f"{w} {n}", (40 + col * 170, y), font, 0.55, (230, 180, 90), 1, cv2.LINE_AA)
    for _ in range(12):
        c = tuple(int(v) for v in rng.integers(60, 255, 3))
        x, y = int(rng.integers(20, PANEL_W - 60)), int(rng.integers(250, PANEL_H - 40))
        cv2.circle(p, (x, y), int(rng.integers(6, 20)), c, -1)
    return p


def make_background(seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bg = rng.integers(150, 200, (FRAME_H // 8, FRAME_W // 8, 3)).astype(np.uint8)
    bg = cv2.resize(bg, (FRAME_W, FRAME_H), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(bg, (0, 0), 3)


def panel_to_frame(t: float) -> np.ndarray:
    """Homography from panel pixels to frame pixels at time parameter t (0 = reference)."""
    ox, oy = PANEL_ORIGIN
    a = 0.06 * math.sin(1.3 * t)
    s = 1.0 + 0.15 * math.sin(0.7 * t)
    tx = ox + 60 * math.sin(0.9 * t)
    ty = oy + 35 * math.sin(1.1 * t)
    persp = 1.5e-4 * math.sin(0.5 * t)
    c, si = math.cos(a) * s, math.sin(a) * s
    H = np.array([[c, -si, tx], [si, c, ty], [persp, 0.0, 1.0]])
    return H


def render_frame(panel: np.ndarray, bg: np.ndarray, H: np.ndarray | None) -> np.ndarray:
    frame = bg.copy()
    if H is None:
        return frame
    warped = cv2.warpPerspective(panel, H, (FRAME_W, FRAME_H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full(panel.shape[:2], 255, np.uint8), H, (FRAME_W, FRAME_H),
                               flags=cv2.INTER_NEAREST)
    frame[mask > 0] = warped[mask > 0]
    return frame


def ref_box(panel_box) -> tuple[float, float, float, float]:
    """A panel-space box expressed in reference-frame pixels (frame 0 has no rotation)."""
    H0 = panel_to_frame(0.0)
    x, y, w, h = panel_box
    p = np.array([[x, y, 1.0], [x + w, y + h, 1.0]]) @ H0.T
    p = p[:, :2] / p[:, 2:3]
    return float(p[0, 0]), float(p[0, 1]), float(p[1, 0] - p[0, 0]), float(p[1, 1] - p[0, 1])
