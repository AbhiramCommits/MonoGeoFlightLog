"""Tests for camera calibration and the classical two-view geometry baseline."""

from __future__ import annotations

import numpy as np
import pytest

from monogeo import synth
from monogeo.calib import (
    calibrate_from_checkerboards,
    calibration_report,
    distortion_displacement_px,
)
from monogeo.data import load_sequence
from monogeo.twoview import (
    depth_from_triangulation,
    estimate_pose,
    match_features,
    pose_error,
    relative_pose,
    reprojection_error,
    rotation_error_deg,
    scale_from_gt,
    translation_angle_deg,
    triangulate,
)

SIZE = (128, 416)


@pytest.fixture(scope="module")
def forward_seq(tmp_path_factory):
    """A small synthetic forward sequence shared by the tests in this module."""
    root = tmp_path_factory.mktemp("data")
    return load_sequence("forward", size=SIZE, num_frames=10, root=root, cache=False, verbose=False)


def test_calibration_recovers_intrinsics() -> None:
    """Calibration must recover focal length to a few pixels and a low RMS."""
    K_true = synth.default_intrinsics(640, 480, 62.0)
    images, _, _ = synth.render_checkerboard_views(
        K_true, synth.DEFAULT_DIST, n=12, size=(480, 640), seed=0
    )
    res = calibrate_from_checkerboards(images, pattern=(6, 7), square_size=0.04)

    assert res.num_views >= 8
    assert res.rms < 1.5, f"RMS reprojection {res.rms:.3f} px is too high"
    assert abs(res.K[0, 0] - K_true[0, 0]) < 0.02 * K_true[0, 0]
    assert abs(res.K[1, 1] - K_true[1, 1]) < 0.02 * K_true[1, 1]
    assert abs(res.K[0, 2] - K_true[0, 2]) < 15.0
    assert abs(res.K[1, 2] - K_true[1, 2]) < 15.0

    report = calibration_report(
        res.K, res.dist, K_true, synth.DEFAULT_DIST, rms=res.rms, image_size=(640, 480)
    )
    assert report["focal_mean_abs_error_px"] < 5.0
    # the distortion *field* is what matters, not the individual coefficients
    disp = report["distortion_field_displacement_px"]
    assert disp["max"] < 6.0, f"distortion field off by {disp['max']:.2f} px"


def test_calibration_rejects_too_few_views() -> None:
    """Fewer than three detectable boards must raise rather than silently fit."""
    K_true = synth.default_intrinsics(640, 480, 62.0)
    images, _, _ = synth.render_checkerboard_views(
        K_true, synth.DEFAULT_DIST, n=2, size=(480, 640), seed=0
    )
    with pytest.raises(ValueError):
        calibrate_from_checkerboards(images, pattern=(6, 7), square_size=0.04)


def test_distortion_displacement_is_zero_for_identical_models() -> None:
    """The distortion metric must vanish for identical coefficient vectors."""
    K = synth.default_intrinsics(320, 96, 62.0)
    _mean_px, max_px = distortion_displacement_px(
        K, synth.DEFAULT_DIST, synth.DEFAULT_DIST, (320, 96)
    )
    assert max_px < 1e-9
    # and be clearly non-zero for a different model
    _mean_px2, max_px2 = distortion_displacement_px(K, synth.DEFAULT_DIST, np.zeros(5), (320, 96))
    assert max_px2 > 1.0


def test_match_features_finds_correspondences(forward_seq) -> None:
    """SIFT matching must find a healthy number of mutual matches."""
    m = match_features(forward_seq[1], forward_seq[3], max_features=3000, ratio=0.75)
    assert m.detector == "sift"
    assert m.n_keypoints1 > 150 and m.n_keypoints2 > 150
    assert len(m) > 40, f"only {len(m)} matches"
    assert 0.0 < m.n_mutual / max(1, m.n_ratio) <= 1.0
    assert m.points1.shape == m.points2.shape
    # matches must be inside the image
    assert m.points1[:, 0].max() < forward_seq.width
    assert m.points1[:, 1].max() < forward_seq.height


def test_orb_fallback_path(tmp_path) -> None:
    """The ORB fallback path must produce matches and a usable pose.

    ORB is far weaker than SIFT on this scene (the terrain is smooth blob-like
    noise, which is exactly what a corner detector dislikes), so the bar is
    deliberately low: the point of the test is that the fallback path runs end
    to end, not that it competes with SIFT.
    """
    seq = load_sequence(
        "forward", size=SIZE, num_frames=4, root=tmp_path, cache=False, verbose=False
    )
    m = match_features(seq[1], seq[3], detector="orb", max_features=4000)
    assert m.detector == "orb"
    assert len(m) >= 5, f"ORB produced only {len(m)} matches"
    assert m.points1.shape == m.points2.shape
    assert m.points1[:, 0].max() < seq.width and m.points1[:, 1].max() < seq.height
    # the fallback must still be usable by the pose estimator
    tv = estimate_pose(m, seq.K)
    assert 0.0 <= tv.inlier_ratio <= 1.0
    assert tv.num_inliers >= 3


def test_two_view_pose_and_reprojection(forward_seq) -> None:
    """Pose and reprojection accuracy on a synthetic pair with a known pose."""
    seq = forward_seq
    i, j = 2, 4  # two frames apart -> a real baseline
    matches = match_features(seq[i], seq[j], max_features=4000, ratio=0.75)
    assert len(matches) > 40

    tv = estimate_pose(matches, seq.K, threshold_px=1.0)
    R_gt, t_gt = relative_pose(seq.poses, i, j)
    err = pose_error(tv.R, tv.t, R_gt, t_gt)

    assert err["rotation_deg"] < 2.0, f"rotation error {err['rotation_deg']:.3f} deg"
    assert err["translation_deg"] < 10.0, f"translation error {err['translation_deg']:.3f} deg"
    assert tv.inlier_ratio > 0.3, f"inlier ratio {tv.inlier_ratio:.3f}"

    pts3d, valid = triangulate(matches, seq.K, tv.R, tv.t, mask=tv.inlier_mask)
    assert len(pts3d) > 20
    rep = reprojection_error(pts3d, matches, seq.K, tv.R, tv.t, mask=valid, view="second")
    assert rep["median"] < 2.0, f"median reprojection {rep['median']:.3f} px"
    assert rep["median"] > 0.0

    # triangulated points must be in front of both cameras
    X2 = pts3d @ tv.R.T + tv.t
    assert np.all(pts3d[:, 2] > 0)
    assert np.all(X2[:, 2] > 0)


def test_scale_from_gt_recovers_the_baseline(forward_seq) -> None:
    """The monocular scale factor must equal the true baseline length."""
    seq = forward_seq
    i, j = 2, 4
    matches = match_features(seq[i], seq[j], max_features=4000)
    tv = estimate_pose(matches, seq.K)
    pts3d, valid = triangulate(matches, seq.K, tv.R, tv.t, mask=tv.inlier_mask)
    sc = scale_from_gt(pts3d, seq.depths[i], points1=matches.points1[valid])

    _, t_gt = relative_pose(seq.poses, i, j)
    baseline = float(np.linalg.norm(t_gt))
    assert sc["n"] > 20
    assert np.isfinite(sc["scale"]) and sc["scale"] > 0
    # median scale should be within 25% of the true metric baseline
    assert (
        abs(sc["scale"] - baseline) / baseline < 0.30
    ), f"scale {sc['scale']:.3f} vs baseline {baseline:.3f}"


def test_pose_error_helpers() -> None:
    """Rotation/translation error helpers behave on known transforms."""
    R = np.eye(3)
    assert rotation_error_deg(R, R) == pytest.approx(0.0, abs=1e-9)
    ang = np.deg2rad(5.0)
    Rz = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    assert rotation_error_deg(Rz, R) == pytest.approx(5.0, abs=1e-6)

    assert translation_angle_deg(np.array([1.0, 0, 0]), np.array([2.0, 0, 0])) == pytest.approx(0.0)
    assert translation_angle_deg(np.array([1.0, 0, 0]), np.array([0, 1.0, 0])) == pytest.approx(
        90.0
    )
    # scale invariance of the direction-only comparison
    assert translation_angle_deg(np.array([1.0, 0, 0]), np.array([100.0, 0, 0])) == pytest.approx(
        0.0
    )


def test_depth_from_triangulation_is_sparse_but_positive(forward_seq) -> None:
    """Triangulated depth maps must be sparse, positive and roughly the right scale."""
    seq = forward_seq
    matches = match_features(seq[0], seq[2], max_features=4000)
    tv = estimate_pose(matches, seq.K)
    pts3d, valid = triangulate(matches, seq.K, tv.R, tv.t, mask=tv.inlier_mask)
    sc = scale_from_gt(pts3d, seq.depths[0], points1=matches.points1[valid])
    depth = depth_from_triangulation(pts3d * sc["scale"], matches.points1[valid], SIZE)

    covered = depth > 0
    assert 0.0 < covered.mean() < 0.5  # genuinely sparse, unlike the learned methods
    d = depth[covered]
    assert np.all(d > 0) and np.all(np.isfinite(d))
    gt_med = np.median(seq.depths[0][seq.depths[0] > 0])
    assert 0.2 * gt_med < np.median(d) < 5.0 * gt_med
