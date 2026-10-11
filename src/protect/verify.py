"""Check a rendered video against its plan, and scan it for leaked text.

Three independent checks:

* every frame: the planned regions are solid black in the output, and frames
  planned as fully black are black;
* sampled frames: OCR (tesseract) around the regions in both the original and
  the output, searching for user-supplied terms. Hits in the original measure
  how sensitive the OCR is on this footage; any hit in the output is a failure;
* contact sheets of the output around the regions, for a human to look at.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np

from . import video
from .job import Job
from .render import frame_polygons
from .timeline import BLACKOUT, FILLED, KEPT, LOST, OK, STATUS_NAMES, Plan

BLACK_LEVEL = 24       # max pixel value accepted as "black" after lossy encoding


# ---------------------------------------------------------------------------
# term matching

def load_terms(path: str | os.PathLike) -> list[str]:
    terms = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            terms.append(line)
    return terms


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def match_terms(text: str, terms: list[str], threshold: float = 0.75) -> list[tuple[str, str]]:
    """Fuzzy-find terms in OCR output. Returns (term, matched_text) pairs."""
    tokens = [t for t in re.findall(r"[A-Z0-9]+", text.upper()) if len(t) >= 3]
    joined = "".join(tokens)
    hits = []
    for term in terms:
        t = _norm(term)
        if len(t) < 3:
            continue
        if t in joined:
            hits.append((term, t))
            continue
        # a fuzzy match must cover most of the term: short fragments such as
        # "PER" are everywhere in OCR noise and say nothing about "PERSON"
        min_len = max(4, -(-3 * len(t) // 4))
        max_len = len(t) + max(2, len(t) // 4)
        for tok in tokens:
            if min_len <= len(tok) <= max_len and SequenceMatcher(None, tok, t).ratio() >= threshold:
                hits.append((term, tok))
                break
    return hits


# ---------------------------------------------------------------------------
# OCR

def have_tesseract() -> bool:
    return shutil.which("tesseract") is not None


def ocr(img: np.ndarray, scale: float = 3.0) -> str:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if scale != 1.0:
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    ok, png = cv2.imencode(".png", 255 - g)    # tesseract prefers dark text on light
    if not ok:
        return ""
    env = dict(os.environ, OMP_THREAD_LIMIT="1")
    res = subprocess.run(["tesseract", "stdin", "stdout", "--psm", "6"], input=png.tobytes(),
                         capture_output=True, env=env)
    return res.stdout.decode(errors="replace")


# ---------------------------------------------------------------------------
# helpers

def _crop_box(plan: Plan, i: int, size) -> tuple[int, int, int, int] | None:
    bb = plan.bbox(i)
    if bb is None:
        return None
    w, h = size
    x0, y0, x1, y1 = bb
    mx = max(40.0, 0.5 * (x1 - x0))
    my = max(40.0, 1.0 * (y1 - y0))
    x0, y0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
    x1, y1 = int(min(w, x1 + mx)), int(min(h, y1 + my))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return x0, y0, x1, y1


def _mask_is_black(frame: np.ndarray, plan: Plan, i: int) -> bool:
    """True if the interior of every planned polygon is black in ``frame``."""
    h, w = frame.shape[:2]
    for poly in frame_polygons(plan, i, (w, h)):
        x0, y0 = max(0, poly[:, 0].min()), max(0, poly[:, 1].min())
        x1, y1 = min(w, poly[:, 0].max() + 1), min(h, poly[:, 1].max() + 1)
        if x1 <= x0 or y1 <= y0:
            continue                      # entirely outside the frame
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(m, [poly - [x0, y0]], 255)
        m = cv2.erode(m, np.ones((3, 3), np.uint8))   # ignore chroma bleed at the edge
        if not m.any():
            continue
        roi = frame[y0:y1, x0:x1]
        if roi[m > 0].max() > BLACK_LEVEL:
            return False
    return True


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = cv2.copyMakeBorder(img, 18, 0, 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    cv2.putText(out, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def contact_sheets(tiles: list[tuple[str, np.ndarray]], out_prefix: Path,
                   tile_h: int = 120, max_w: int = 560, cols: int = 3, rows: int = 12) -> list[Path]:
    paths = []
    per = cols * rows
    for s in range(0, len(tiles), per):
        chunk = tiles[s:s + per]
        cells = []
        for text, img in chunk:
            f = tile_h / img.shape[0]
            if img.shape[1] * f > max_w:
                f = max_w / img.shape[1]
            im = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
            cell = np.full((tile_h, max_w, 3), 255, np.uint8)
            cell[:im.shape[0], :im.shape[1]] = im
            cells.append(_label(cell, text))
        while len(cells) % cols:
            cells.append(np.full_like(cells[0], 255))
        grid = [np.hstack(cells[r:r + cols]) for r in range(0, len(cells), cols)]
        sheet = np.vstack(grid)
        path = out_prefix.with_name(f"{out_prefix.name}-{s // per + 1:03d}.jpg")
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------

def verify(job: Job, plan: Plan, output: Path | None = None, terms: list[str] | None = None,
           every: float = 1.0, progress: bool = True) -> dict:
    output = Path(output or job.output)
    info = video.probe(job.input)
    out_info = video.probe(output)
    size = info.size
    review = job.review_dir
    review.mkdir(parents=True, exist_ok=True)
    for old in review.glob("output-*.jpg"):
        old.unlink()
    for old in review.glob("kept-*.jpg"):
        old.unlink()

    n = len(plan)
    step = max(1, int(round(every * plan.fps)))
    sample = set(range(0, n, step))
    ok_idx = np.flatnonzero(plan.status == OK)
    worst = ok_idx[np.argsort(plan.score[ok_idx])[:40]] if len(ok_idx) else []
    filled = np.flatnonzero(plan.status == FILLED)
    if len(filled) > 40:
        filled = filled[np.linspace(0, len(filled) - 1, 40).astype(int)]
    extra = set(int(i) for i in worst) | set(int(i) for i in filled)
    sample |= extra

    use_ocr = bool(terms) and have_tesseract()
    jobs_orig, jobs_out = [], []           # (frame, image)
    tiles, kept_tiles = [], []
    mask_fail, black_fail = [], []
    i = 0
    t0 = time.time()
    for orig, out in zip(video.iter_frames(job.input, info), video.iter_frames(output, out_info)):
        st = plan.status[i]
        if st in (LOST, BLACKOUT):
            if out.max() > BLACK_LEVEL:
                black_fail.append(i)
        elif st in (OK, FILLED):
            if not _mask_is_black(out, plan, i):
                mask_fail.append(i)
        if i in sample:
            tag = f"#{i}  {i / plan.fps:7.1f}s  {STATUS_NAMES[st]}"
            if st in (OK, FILLED):
                cb = _crop_box(plan, i, size)
                if cb:
                    x0, y0, x1, y1 = cb
                    oc = out[y0:y1, x0:x1].copy()
                    if use_ocr:
                        jobs_orig.append((i, orig[y0:y1, x0:x1].copy()))
                        jobs_out.append((i, oc))
                    if st == OK:
                        tag += f" {plan.score[i]:.2f}"
                    tiles.append((tag, oc))
            elif st == KEPT:
                if use_ocr:
                    jobs_out.append((i, out.copy()))
                kept_tiles.append((tag, cv2.resize(out, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)))
        i += 1
        if progress and i % 1000 == 0:
            sys.stderr.write(f"\r  checked {i}/{n} frames   ")
            sys.stderr.flush()
    if progress:
        sys.stderr.write(f"\r  checked {i}/{n} frames   \n")
    if i != n:
        raise video.VideoError(f"output/original have {i} comparable frames, plan has {n}")

    report: dict = {
        "frames": n,
        "status": plan.counts(),
        "mask_check": {"failed_frames": mask_fail[:200], "failed": len(mask_fail)},
        "black_check": {"failed_frames": black_fail[:200], "failed": len(black_fail)},
    }

    if use_ocr:
        def run(items):
            with ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
                texts = list(ex.map(lambda it: ocr(it[1], 1.0 if it[1].shape[1] > 1200 else 3.0), items))
            return [(fi, match_terms(t, terms)) for (fi, _), t in zip(items, texts)]
        if progress:
            sys.stderr.write(f"  OCR on {len(jobs_orig)} original and {len(jobs_out)} output samples...\n")
        orig_hits = run(jobs_orig)
        out_hits = run(jobs_out)
        per_term = {}
        for idx, term in enumerate(terms):
            label = f"term{idx + 1}"     # never write the terms themselves to disk
            per_term[label] = {
                "original_frames": sum(1 for _, h in orig_hits if any(t == term for t, _ in h)),
                "output_frames": sum(1 for _, h in out_hits if any(t == term for t, _ in h)),
            }
        leaks = [{"frame": fi, "time": round(fi / plan.fps, 2),
                  "terms": sorted({f"term{terms.index(t) + 1}" for t, _ in h})}
                 for fi, h in out_hits if h]
        report["ocr"] = {
            "original_samples": len(jobs_orig),
            "output_samples": len(jobs_out),
            "original_frames_with_hits": sum(1 for _, h in orig_hits if h),
            "output_frames_with_hits": len(leaks),
            "per_term": per_term,
            "output_hits": leaks,
        }
    elif terms:
        report["ocr"] = {"skipped": "tesseract not found"}

    sheets = contact_sheets(tiles, review / "output") if tiles else []
    sheets += contact_sheets(kept_tiles, review / "kept", tile_h=270, max_w=480, cols=4, rows=6) if kept_tiles else []
    report["sheets"] = [str(p) for p in sheets]
    report["elapsed_s"] = round(time.time() - t0, 1)
    report["passed"] = (not mask_fail and not black_fail
                        and not (use_ocr and report["ocr"]["output_frames_with_hits"]))
    (review / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def untracked_sheets(job: Job, plan: Plan, every: float = 0.5, progress: bool = True) -> list[Path]:
    """Contact sheets of the original frames that tracking could not cover.

    Use these to decide which time ranges are safe to list under ``keep``
    (nothing sensitive visible) and which should stay blacked out.
    """
    runs = plan.runs([LOST, KEPT])
    if not runs:
        return []
    step = max(1, int(round(every * plan.fps)))
    want: dict[int, str] = {}
    for a, b in runs:
        for f in sorted({a, b, *range(a, b + 1, step)}):
            want[f] = f"#{f} {f / plan.fps:7.1f}s  run {a / plan.fps:.1f}-{b / plan.fps:.1f}s"
    info = video.probe(job.input)
    tiles = []
    for i, frame in enumerate(video.iter_frames(job.input, info)):
        if i in want:
            tiles.append((want[i], cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)))
        if i > max(want):
            break
    review = job.review_dir
    review.mkdir(parents=True, exist_ok=True)
    for old in review.glob("untracked-*.jpg"):
        old.unlink()
    return contact_sheets(tiles, review / "untracked", tile_h=270, max_w=480, cols=4, rows=6)
