"""Job files: which video to process, what to track and what to black out.

A job is a small JSON file. Paths inside it are resolved relative to the job
file, and all coordinates are pixels in the job's reference image (a still
frame taken from the video).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

Box = tuple[float, float, float, float]
Range = tuple[float, float]


class JobError(ValueError):
    pass


@dataclass
class Region:
    label: str
    box: Box


@dataclass
class Options:
    # feature matching
    ratio: float = 0.8              # Lowe ratio test
    ransac_px: float = 3.0          # RANSAC reprojection threshold (frame pixels)
    min_inliers: int = 15           # fewer inliers than this = not found
    max_features: int = 8000        # cap on keypoints detected per frame
    min_scale: float = 0.1          # allowed size of the surface relative to the reference
    max_scale: float = 10.0
    # sub-pixel refinement around the redacted regions
    refine_margin: float = 24.0     # context around the redact regions (reference pixels)
    max_refine_pixels: int = 600_000
    min_score: float = 0.5          # alignment correlation below this = not found
    # temporal consistency and fail-closed behaviour
    max_jump: float = 20.0          # max deviation (frame px) from neighbouring frames
    offscreen_min_inliers: int = 30 # low-score frames still count if the regions are confidently
    offscreen_margin: float = 40.0  #   off-screen: this many matches and this far outside (frame px)
    max_gap: int = 15               # untracked frames bridged from tracked neighbours
    gap_padding: float = 6.0        # extra padding for bridged frames (frame px)
    gap_growth: float = 3.0         # more padding per frame away from a tracked frame
    frame_padding: int = 2          # padding added in frame space to every region
    # encoding
    crf: int = 18
    preset: str = "medium"

    # options that change tracking results (anything else only affects rendering)
    TRACKING_KEYS = ("ratio", "ransac_px", "min_inliers", "max_features", "min_scale",
                     "max_scale", "refine_margin", "max_refine_pixels")


@dataclass
class Job:
    path: Path
    input: Path
    output: Path
    reference: Path
    track: list[Box] = field(default_factory=list)
    redact: list[Region] = field(default_factory=list)
    padding: float = 4.0
    search: Box | None = None
    refine: Box | None = None
    keep: list[Range] = field(default_factory=list)
    blackout: list[Range] = field(default_factory=list)
    audio: bool = False
    options: Options = field(default_factory=Options)

    # ---- derived paths -------------------------------------------------
    @property
    def track_path(self) -> Path:
        return self.path.with_suffix(".track.npz")

    @property
    def review_dir(self) -> Path:
        return self.path.with_suffix(".review")

    # ---- validation ----------------------------------------------------
    def validate(self, need_track: bool = True, need_redact: bool = True) -> None:
        if need_track and not self.track:
            raise JobError("job has no 'track' regions; mark static texture on the surface first")
        if need_redact and not self.redact:
            raise JobError("job has no 'redact' regions")
        if not self.input.exists():
            raise JobError(f"input video not found: {self.input}")
        if not self.reference.exists():
            raise JobError(f"reference image not found: {self.reference}")
        if self.output.resolve() == self.input.resolve():
            raise JobError("output must not overwrite the input")

    def tracking_fingerprint(self) -> str:
        """Changes whenever stored tracking results would be invalid."""
        st = self.input.stat()
        payload = {
            "input": [str(self.input.resolve()), st.st_size, int(st.st_mtime)],
            "reference": hashlib.sha256(self.reference.read_bytes()).hexdigest(),
            "track": [list(map(float, b)) for b in self.track],
            "redact": [list(map(float, r.box)) for r in self.redact],
            "search": list(map(float, self.search)) if self.search else None,
            "refine": list(map(float, self.refine)) if self.refine else None,
            "options": {k: getattr(self.options, k) for k in Options.TRACKING_KEYS},
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    # ---- (de)serialisation --------------------------------------------
    def to_dict(self) -> dict:
        base = self.path.parent
        defaults = Options()
        opts = {f.name: getattr(self.options, f.name) for f in fields(Options)
                if getattr(self.options, f.name) != getattr(defaults, f.name)}
        return {
            "input": _rel(self.input, base),
            "output": _rel(self.output, base),
            "reference": _rel(self.reference, base),
            "track": [list(b) for b in self.track],
            "redact": [asdict(r) | {"box": list(r.box)} for r in self.redact],
            "padding": self.padding,
            "search": list(self.search) if self.search else None,
            "refine": list(self.refine) if self.refine else None,
            "keep": [list(r) for r in self.keep],
            "blackout": [list(r) for r in self.blackout],
            "audio": self.audio,
            "options": opts,
        }

    def save(self) -> None:
        self.path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")


def _rel(p: Path, base: Path) -> str:
    try:
        return str(p.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(p)


def _box(value, what: str) -> Box:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise JobError(f"{what}: expected [x, y, w, h], got {value!r}")
    x, y, w, h = (float(v) for v in value)
    if w <= 0 or h <= 0:
        raise JobError(f"{what}: width and height must be positive")
    return x, y, w, h


def _range(value, what: str) -> Range:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise JobError(f"{what}: expected [start_seconds, end_seconds], got {value!r}")
    s, e = float(value[0]), float(value[1])
    if e < s:
        raise JobError(f"{what}: end is before start")
    return s, e


def load(path: str | Path) -> Job:
    path = Path(path).resolve()
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        raise JobError(f"job file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise JobError(f"{path}: invalid JSON ({exc})") from None
    base = path.parent

    def p(key: str, required: bool = True) -> Path | None:
        if key not in data or data[key] in (None, ""):
            if required:
                raise JobError(f"job is missing '{key}'")
            return None
        q = Path(data[key]).expanduser()
        return q if q.is_absolute() else (base / q)

    known = {f.name for f in fields(Options)}
    raw_opts = data.get("options") or {}
    unknown = set(raw_opts) - known
    if unknown:
        raise JobError(f"unknown options: {', '.join(sorted(unknown))}")
    options = Options(**raw_opts)

    redact = []
    for i, r in enumerate(data.get("redact") or []):
        if isinstance(r, dict):
            redact.append(Region(str(r.get("label") or f"region{i + 1}"), _box(r.get("box"), f"redact[{i}]")))
        else:
            redact.append(Region(f"region{i + 1}", _box(r, f"redact[{i}]")))

    output = p("output", required=False)
    inp = p("input")
    if output is None:
        output = inp.with_name(inp.stem + ".redacted.mp4")
    return Job(
        path=path,
        input=inp,
        output=output,
        reference=p("reference"),
        track=[_box(b, f"track[{i}]") for i, b in enumerate(data.get("track") or [])],
        redact=redact,
        padding=float(data.get("padding", 4.0)),
        search=_box(data["search"], "search") if data.get("search") else None,
        refine=_box(data["refine"], "refine") if data.get("refine") else None,
        keep=[_range(r, f"keep[{i}]") for i, r in enumerate(data.get("keep") or [])],
        blackout=[_range(r, f"blackout[{i}]") for i, r in enumerate(data.get("blackout") or [])],
        audio=bool(data.get("audio", False)),
        options=options,
    )
