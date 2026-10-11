import numpy as np
import pytest

from protect import geometry, timeline
from protect.job import Job, Options, Region
from protect.tracker import Reference
from protect.verify import match_terms

import synthetic as syn


# ---------------------------------------------------------------- geometry

def test_box_quad_and_projection():
    q = geometry.box_quad((10, 20, 30, 40), pad=2)
    assert q.tolist() == [[8, 18], [42, 18], [42, 62], [8, 62]]
    H = geometry.translation(5, -3)
    assert np.allclose(geometry.project(H, q), q + [5, -3])


def test_plausibility_rejects_flips_and_collapse():
    q = geometry.box_quad((0, 0, 100, 50))
    assert geometry.homography_is_plausible(np.eye(3), q)
    mirror = np.diag([-1.0, 1.0, 1.0])
    assert not geometry.homography_is_plausible(mirror, q)
    tiny = np.diag([0.01, 0.01, 1.0])
    assert not geometry.homography_is_plausible(tiny, q)
    assert not geometry.homography_is_plausible(np.full((3, 3), np.nan), q)


# ---------------------------------------------------------------- tracking

@pytest.fixture(scope="module")
def scene():
    panel = syn.make_panel()
    bg = syn.make_background()
    ref = syn.render_frame(panel, bg, syn.panel_to_frame(0.0))
    gray = lambda im: np.ascontiguousarray(im[..., 1]) if im.ndim == 3 else im
    import cv2
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    track = [syn.ref_box((0, 0, syn.PANEL_W, syn.PANEL_H))]
    secret = syn.ref_box(syn.SECRET_BOX_PANEL)
    reference = Reference(ref_gray, track, [secret], None, Options())
    return panel, bg, reference, secret


@pytest.mark.parametrize("t", [0.4, 1.0, 1.9, 2.7, 3.5])
def test_tracker_recovers_known_motion(scene, t):
    import cv2
    panel, bg, reference, secret = scene
    Hp = syn.panel_to_frame(t)
    frame = cv2.cvtColor(syn.render_frame(panel, bg, Hp), cv2.COLOR_BGR2GRAY)
    est = reference.estimate(frame)
    assert est.H is not None
    assert est.score > 0.8
    truth = Hp @ np.linalg.inv(syn.panel_to_frame(0.0))
    corners = geometry.box_quad(secret)
    err = np.linalg.norm(geometry.project(est.H, corners) - geometry.project(truth, corners), axis=1)
    assert err.max() < 1.0, err


@pytest.mark.parametrize("t", [1.0, 2.0])
def test_letterbox_bars_are_not_compared(scene, t):
    """Black bars must not count as picture content when scoring the alignment."""
    import cv2
    panel, bg, _, secret = scene
    left, right = 260, 860            # the panel starts under the left bar in the reference

    def boxed(img):
        img = img.copy()
        img[:, :left] = 0
        img[:, right:] = 0
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ref = boxed(syn.render_frame(panel, bg, syn.panel_to_frame(0.0)))
    reference = Reference(ref, [syn.ref_box((0, 0, syn.PANEL_W, syn.PANEL_H))], [secret],
                          (left, 0, right - left, syn.FRAME_H), Options())
    Hp = syn.panel_to_frame(t)
    est = reference.estimate(boxed(syn.render_frame(panel, bg, Hp)))
    assert est.H is not None and est.score > 0.95, est.score   # ~0.87 if bars are compared
    truth = Hp @ np.linalg.inv(syn.panel_to_frame(0.0))
    corners = geometry.box_quad(secret)
    err = np.linalg.norm(geometry.project(est.H, corners) - geometry.project(truth, corners), axis=1)
    assert err.max() < 1.0, err


def test_explicit_refine_box_sets_alignment_band(scene):
    _, _, _, secret = scene
    import cv2
    panel, bg = syn.make_panel(), syn.make_background()
    ref = cv2.cvtColor(syn.render_frame(panel, bg, syn.panel_to_frame(0.0)), cv2.COLOR_BGR2GRAY)
    r = Reference(ref, [syn.ref_box((0, 0, syn.PANEL_W, syn.PANEL_H))], [secret], None, Options(),
                  refine=(250, 130, 400, 40))
    assert r.band == (250, 130, 400, 40)


def test_tracker_reports_absent_surface(scene):
    import cv2
    panel, bg, reference, _ = scene
    frame = cv2.cvtColor(syn.render_frame(panel, bg, None), cv2.COLOR_BGR2GRAY)
    assert reference.estimate(frame).H is None


# ---------------------------------------------------------------- timeline

def _job(tmp_path, **kw):
    return Job(path=tmp_path / "j.json", input=tmp_path / "in.mp4", output=tmp_path / "out.mp4",
               reference=tmp_path / "ref.png", track=[(0, 0, 100, 100)],
               redact=[Region("r", (10, 10, 20, 10))], padding=0, **kw)


def _track(n, fail=(), glitch=()):
    H = np.repeat(np.eye(3)[None], n, axis=0)
    H[:, 0, 2] = np.arange(n) * 2.0           # smooth motion, 2 px/frame
    score = np.full(n, 0.95, np.float32)
    for i in fail:
        H[i] = np.nan
        score[i] = 0
    for i in glitch:
        H[i, 0, 2] += 200                      # confident but wrong
    return {"H": H, "score": score, "inliers": np.full(n, 100, np.int32), "width": 640, "height": 480}


def test_plan_statuses(tmp_path):
    fps = 10.0
    fail = list(range(10, 13)) + list(range(40, 80))
    job = _job(tmp_path, keep=[(6.0, 6.5)], blackout=[(9.0, 9.4)])
    plan = timeline.build_plan(job, _track(100, fail=fail, glitch=[25]), fps)
    st = plan.status
    assert all(st[i] == timeline.FILLED for i in range(10, 13))
    assert st[25] == timeline.FILLED                       # glitch rejected and bridged
    assert all(st[i] == timeline.FILLED for i in range(40, 40 + job.options.max_gap))
    assert all(st[i] == timeline.FILLED for i in range(80 - job.options.max_gap, 80))
    assert all(st[i] == timeline.KEPT for i in range(60, 65))   # 65+ is bridged from frame 80
    assert st[56] == timeline.LOST and st[59] == timeline.LOST
    assert all(st[i] == timeline.BLACKOUT for i in range(90, 95))
    assert st[0] == timeline.OK and st[99] == timeline.OK
    # a bridged frame covers where the region actually was, with padding
    i = 11
    true_quad = geometry.project(geometry.translation(i * 2.0, 0), geometry.box_quad((10, 10, 20, 10)))
    hull = plan.polys[i][0]
    import cv2
    for x, y in true_quad:
        assert cv2.pointPolygonTest(hull.astype(np.float32), (float(x), float(y)), True) >= -plan.pad[i]


def test_plan_without_any_tracking_blacks_everything(tmp_path):
    plan = timeline.build_plan(_job(tmp_path), _track(30, fail=range(30)), 30.0)
    assert (plan.status == timeline.LOST).all()


# ---------------------------------------------------------------- OCR matching

def test_match_terms_is_fuzzy_but_specific():
    assert match_terms("xx SAMPIE PERS0N 4471 yy", ["SAMPLE"])
    assert match_terms("SAMPLEPERSON", ["sample"])
    assert not match_terms("ALPHA BRAVO 12 %", ["SAMPLE", "PERSON"])
    assert not match_terms("Per ee Ind Ton", ["SAMPLE", "PERSON"])     # fragments are not matches
    assert match_terms("PERSN", ["PERSON"])                            # one letter dropped


def test_offscreen_regions_need_strong_match(tmp_path):
    t = _track(10)
    t["H"][:, 1, 2] = -200                 # regions projected far above the frame
    t["score"][:] = 0.0                    # nothing to align against
    plan = timeline.build_plan(_job(tmp_path), t, 30.0)
    assert (plan.status == timeline.OK).all()
    t["inliers"][:] = 20                   # weak match: not trusted
    plan = timeline.build_plan(_job(tmp_path), t, 30.0)
    assert (plan.status == timeline.LOST).all()
    t["inliers"][:] = 100
    t["H"][:, 1, 2] = -30                  # just above the edge: inside the safety margin
    plan = timeline.build_plan(_job(tmp_path), t, 30.0)
    assert (plan.status == timeline.LOST).all()
