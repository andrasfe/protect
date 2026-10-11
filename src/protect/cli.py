"""Command line interface: ``protect <command> ...``."""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

import cv2
import numpy as np

from . import __version__, job as jobmod, render, timeline, tracker, verify, video
from .geometry import box_quad


def _plan(job: jobmod.Job) -> timeline.Plan:
    data = tracker.load_track(job)
    info = video.probe(job.input)
    return timeline.build_plan(job, data, float(info.fps))


def _fmt_runs(plan: timeline.Plan, statuses, limit: int = 40) -> list[str]:
    runs = plan.runs(statuses)
    lines = [f"    {a / plan.fps:8.2f}s - {b / plan.fps:8.2f}s  ({b - a + 1} frames)" for a, b in runs[:limit]]
    if len(runs) > limit:
        lines.append(f"    ... {len(runs) - limit} more")
    return lines


def _print_plan(plan: timeline.Plan) -> None:
    c = plan.counts()
    n = len(plan)
    print(f"  frames: {n}")
    for name in timeline.STATUS_NAMES:
        if c[name]:
            print(f"    {name:9s} {c[name]:7d}  ({100 * c[name] / n:5.1f}%)")
    if c["lost"]:
        print("  fully blacked out (untracked, not in a 'keep' range):")
        print("\n".join(_fmt_runs(plan, [timeline.LOST])))
    if c["kept"]:
        print("  left untouched (untracked, inside a 'keep' range):")
        print("\n".join(_fmt_runs(plan, [timeline.KEPT])))


# ---------------------------------------------------------------------------
# commands

def cmd_init(a) -> int:
    src = Path(a.video).resolve()
    info = video.probe(src)
    job_path = Path(a.job).resolve() if a.job else src.with_name(src.stem + ".job.json")
    if job_path.exists() and not a.force:
        print(f"{job_path} exists; use --force to overwrite", file=sys.stderr)
        return 1
    ref_path = job_path.with_suffix(".ref.png")
    frame = video.extract_frame(src, a.at, info)
    cv2.imwrite(str(ref_path), frame)
    job = jobmod.Job(path=job_path, input=src,
                     output=Path(a.output).resolve() if a.output else src.with_name(src.stem + ".redacted.mp4"),
                     reference=ref_path)
    job.save()
    print(f"wrote {job_path}\nwrote {ref_path}  ({info.width}x{info.height}, t={a.at}s)")
    print("next: mark regions on the reference image with 'protect select JOB track' and "
          "'protect select JOB redact' (or edit the JSON), then 'protect preview JOB'")
    return 0


def cmd_select(a) -> int:
    job = jobmod.load(a.job)
    img = cv2.imread(str(job.reference))
    if img is None:
        print(f"cannot read {job.reference}", file=sys.stderr)
        return 1
    scale = min(1.0, a.max_width / img.shape[1])
    shown = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
    title = {"track": "track: drag boxes over STATIC texture on the surface, ENTER after each, ESC when done",
             "redact": "redact: drag boxes over what must be hidden, ENTER after each, ESC when done"}[a.kind]
    try:
        rois = cv2.selectROIs(title, shown, showCrosshair=False)
    except cv2.error as exc:
        print(f"cannot open a window ({exc}); edit the job JSON instead", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()
    boxes = [tuple(round(float(v) / scale, 1) for v in r) for r in np.asarray(rois).reshape(-1, 4) if r[2] > 0 and r[3] > 0]
    if a.kind == "track":
        job.track = ([] if a.replace else job.track) + boxes
    else:
        start = 0 if a.replace else len(job.redact)
        new = [jobmod.Region(f"region{start + k + 1}", b) for k, b in enumerate(boxes)]
        job.redact = ([] if a.replace else job.redact) + new
    job.save()
    print(f"added {len(boxes)} {a.kind} region(s) to {job.path}")
    return 0


def cmd_preview(a) -> int:
    job = jobmod.load(a.job)
    img = cv2.imread(str(job.reference))
    over = img.copy()
    for r in job.redact:
        q = box_quad(r.box, job.padding).round().astype(np.int32)
        cv2.fillPoly(over, [q], (0, 0, 255))
    img = cv2.addWeighted(over, 0.45, img, 0.55, 0)
    for r in job.redact:
        q = box_quad(r.box, job.padding).round().astype(np.int32)
        cv2.polylines(img, [q], True, (0, 0, 255), 1)
        cv2.putText(img, r.label, (int(q[0, 0]), int(q[0, 1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    for b in job.track:
        cv2.polylines(img, [box_quad(b).round().astype(np.int32)], True, (0, 255, 0), 1)
    if job.search:
        cv2.polylines(img, [box_quad(job.search).round().astype(np.int32)], True, (255, 128, 0), 1)
    if job.refine:
        cv2.polylines(img, [box_quad(job.refine).round().astype(np.int32)], True, (0, 220, 255), 1)
    out = Path(a.output) if a.output else job.review_dir / "preview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    print(f"wrote {out}  (green: track, red: redact incl. padding, blue: search area, yellow: refine)")
    return 0


def cmd_track(a) -> int:
    job = jobmod.load(a.job)
    print(f"tracking {job.input.name}")
    tracker.track(job, workers=a.workers)
    _print_plan(_plan(job))
    return 0


def cmd_plan(a) -> int:
    job = jobmod.load(a.job)
    plan = _plan(job)
    _print_plan(plan)
    if a.sheets:
        paths = verify.untracked_sheets(job, plan)
        for p in paths:
            print(f"  wrote {p}")
        if not paths:
            print("  no untracked frames")
    return 0


def cmd_render(a) -> int:
    job = jobmod.load(a.job)
    job.validate()
    plan = _plan(job)
    _print_plan(plan)
    out = render.render(job, plan, Path(a.output) if a.output else None)
    print(f"wrote {out}")
    return 0


def _print_report(rep: dict) -> None:
    print(f"  mask check:  {rep['mask_check']['failed']} frame(s) failed")
    print(f"  black check: {rep['black_check']['failed']} frame(s) failed")
    o = rep.get("ocr")
    if o and "skipped" in o:
        print(f"  OCR: skipped ({o['skipped']})")
    elif o:
        print(f"  OCR: terms found in {o['original_frames_with_hits']}/{o['original_samples']} original samples "
              f"(sensitivity on this footage), {o['output_frames_with_hits']}/{o['output_samples']} output samples")
        for h in o["output_hits"][:20]:
            print(f"    possible leak at frame {h['frame']} ({h['time']}s): {', '.join(h['terms'])}")
    for s in rep["sheets"]:
        print(f"  review: {s}")
    print("  PASSED" if rep["passed"] else "  FAILED")


def cmd_verify(a) -> int:
    job = jobmod.load(a.job)
    plan = _plan(job)
    terms = verify.load_terms(a.terms) if a.terms else None
    rep = verify.verify(job, plan, Path(a.output) if a.output else None, terms, every=a.every)
    _print_report(rep)
    return 0 if rep["passed"] else 2


def cmd_run(a) -> int:
    job = jobmod.load(a.job)
    job.validate()
    try:
        tracker.load_track(job)
        print("tracking data is up to date")
    except (FileNotFoundError, ValueError):
        print(f"tracking {job.input.name}")
        tracker.track(job, workers=a.workers)
    plan = _plan(job)
    _print_plan(plan)
    out = render.render(job, plan)
    print(f"wrote {out}\nverifying")
    terms = verify.load_terms(a.terms) if a.terms else None
    rep = verify.verify(job, plan, out, terms, every=a.every)
    _print_report(rep)
    return 0 if rep["passed"] else 2


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="protect", description=(
        "Black out sensitive regions on a screen or other flat surface that moves around in a video."))
    p.add_argument("--version", action="version", version=f"protect {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="create a job and its reference image from a video")
    s.add_argument("video")
    s.add_argument("--at", type=float, required=True, help="time (s) of a sharp frame showing the surface")
    s.add_argument("--job", help="job file to create (default: <video>.job.json next to the video)")
    s.add_argument("--output", help="redacted video to write (default: <video>.redacted.mp4)")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("select", help="draw track or redact regions on the reference image")
    s.add_argument("job")
    s.add_argument("kind", choices=["track", "redact"])
    s.add_argument("--replace", action="store_true", help="replace existing regions instead of adding")
    s.add_argument("--max-width", type=int, default=1600)
    s.set_defaults(func=cmd_select)

    s = sub.add_parser("preview", help="draw the job's regions on the reference image")
    s.add_argument("job")
    s.add_argument("-o", "--output")
    s.set_defaults(func=cmd_preview)

    s = sub.add_parser("track", help="locate the surface in every frame")
    s.add_argument("job")
    s.add_argument("--workers", type=int)
    s.set_defaults(func=cmd_track)

    s = sub.add_parser("plan", help="summarise what will be hidden; --sheets shows untracked frames")
    s.add_argument("job")
    s.add_argument("--sheets", action="store_true")
    s.set_defaults(func=cmd_plan)

    s = sub.add_parser("render", help="write the redacted video")
    s.add_argument("job")
    s.add_argument("-o", "--output")
    s.set_defaults(func=cmd_render)

    for name, func, hlp in (("verify", cmd_verify, "check the redacted video"),
                            ("run", cmd_run, "track (if needed), render and verify")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("job")
        s.add_argument("--terms", help="text file with words that must not be readable, one per line")
        s.add_argument("--every", type=float, default=1.0, help="OCR sampling interval in seconds")
        if name == "verify":
            s.add_argument("-o", "--output", help="video to check (default: the job's output)")
        else:
            s.add_argument("--workers", type=int)
        s.set_defaults(func=func)

    a = p.parse_args(argv)
    # turn SIGTERM into an exception so partial outputs are removed on the way out
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        return a.func(a)
    except (jobmod.JobError, video.VideoError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
