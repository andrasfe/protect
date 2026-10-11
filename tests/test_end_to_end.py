import json
import shutil
import subprocess
from fractions import Fraction

import cv2
import numpy as np
import pytest

from protect import geometry, render, timeline, tracker, verify, video
from protect import job as jobmod

import synthetic as syn

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

N_FRAMES = 80
ABSENT = range(50, 80)          # frames where the surface is out of view
GAP = 15                        # default max_gap: absent frames 50-64 are bridged, 65-79 lost


def _make_video(path, panel, bg):
    raw = path.with_name("raw.mp4")
    w = video.VideoWriter(raw, (syn.FRAME_W, syn.FRAME_H), Fraction(30), crf=10, preset="ultrafast")
    for i in range(N_FRAMES):
        H = None if i in ABSENT else syn.panel_to_frame(i / 12.0)
        w.write(syn.render_frame(panel, bg, H))
    w.close()
    # add an audio track and identifying metadata that must not survive
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=5", "-shortest",
                    "-metadata", "title=private title", "-metadata", "location=+00.0000+000.0000/",
                    "-c:v", "copy", "-c:a", "aac", str(path)], check=True)


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    d = tmp_path_factory.mktemp("e2e")
    panel, bg = syn.make_panel(), syn.make_background()
    src = d / "input.mp4"
    _make_video(src, panel, bg)
    ref = d / "job.ref.png"
    cv2.imwrite(str(ref), video.extract_frame(src, 0.0))
    secret = syn.ref_box(syn.SECRET_BOX_PANEL)
    (d / "job.json").write_text(json.dumps({
        "input": "input.mp4", "output": "output.mp4", "reference": "job.ref.png",
        "track": [syn.ref_box((0, 0, syn.PANEL_W, syn.PANEL_H))],
        "redact": [{"label": "line", "box": secret}],
        "padding": 3,
    }))
    job = jobmod.load(d / "job.json")
    data = tracker.track(job, workers=2, progress=False)
    plan = timeline.build_plan(job, data, 30.0)
    out = render.render(job, plan, progress=False)
    return job, plan, out, secret


def test_every_frame_accounted_for(processed):
    job, plan, out, _ = processed
    assert len(plan) == N_FRAMES
    assert video.probe(out).frame_count == N_FRAMES
    assert all(plan.status[i] == timeline.FILLED for i in range(50, 50 + GAP))
    assert all(plan.status[i] == timeline.LOST for i in range(50 + GAP, 80))
    assert (plan.status[:50] == timeline.OK).sum() >= 48


def test_secret_line_is_black_in_every_frame(processed):
    job, plan, out, secret = processed
    H0inv = np.linalg.inv(syn.panel_to_frame(0.0))
    corners = geometry.box_quad(secret)
    for i, frame in enumerate(video.iter_frames(out)):
        if i in ABSENT:
            if i >= 50 + GAP:
                assert frame.max() <= verify.BLACK_LEVEL, f"frame {i} should be fully black"
            continue
        truth = geometry.project(syn.panel_to_frame(i / 12.0) @ H0inv, corners)
        m = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillPoly(m, [np.round(truth).astype(np.int32)], 255)
        assert frame[m > 0].max() <= verify.BLACK_LEVEL, f"secret visible in frame {i}"


def test_audio_and_metadata_removed(processed):
    _, _, out, _ = processed
    tags = video.container_tags(out)
    assert [s["codec_type"] for s in tags["streams"]] == ["video"]
    fmt = tags.get("format", {}).get("tags", {})
    assert "title" not in fmt and "location" not in fmt


def test_verify_passes(processed):
    job, plan, out, _ = processed
    terms = ["SAMPLE", "PERSON", "4471"] if verify.have_tesseract() else None
    rep = verify.verify(job, plan, out, terms, every=0.5, progress=False)
    assert rep["mask_check"]["failed"] == 0
    assert rep["black_check"]["failed"] == 0
    assert rep["passed"]
    if terms:
        assert rep["ocr"]["original_frames_with_hits"] > 0      # the check can see the text...
        assert rep["ocr"]["output_frames_with_hits"] == 0       # ...and it is gone


def test_verify_catches_unredacted_output(processed):
    job, plan, _, _ = processed
    rep = verify.verify(job, plan, job.input, None, every=1.0, progress=False)
    assert rep["mask_check"]["failed"] > 0 and rep["black_check"]["failed"] > 0
    assert not rep["passed"]
