"""Tests for the deterministic synthetic flight-log renderer and data layer."""

from __future__ import annotations

import numpy as np
import pytest

from monogeo import synth
from monogeo.data import SYNTHETIC_KINDS, SequenceDataset, load_sequence

SMALL = (64, 192)


def test_scene_is_deterministic() -> None:
    """Two scenes built from the same seed must be identical."""
    a = synth.make_scene(3)
    b = synth.make_scene(3)
    c = synth.make_scene(4)
    assert a.points.shape == b.points.shape
    np.testing.assert_array_equal(a.points, b.points)
    np.testing.assert_array_equal(a.colors, b.colors)
    assert len(a) > 10_000
    # a different seed must change the geometry (box layout differs, so compare
    # the deterministic ground plane prefix rather than the full array)
    n = min(len(a), len(c))
    assert not np.allclose(a.points[:n], c.points[:n])


def test_rendered_frames_are_not_degenerate() -> None:
    """Frames must carry real texture, not flat grey."""
    K = synth.default_intrinsics(SMALL[1], SMALL[0], 62.0)
    scene = synth.make_scene(0)
    poses = synth.make_trajectory("forward", n=4)
    frames, depths, out_poses = synth.render_sequence(scene, poses, K, synth.DEFAULT_DIST, SMALL)

    assert frames.shape == (4, SMALL[0], SMALL[1], 3)
    assert frames.dtype == np.uint8
    assert depths.shape == (4, SMALL[0], SMALL[1])
    assert out_poses.shape == (4, 4, 4)

    for i in range(len(frames)):
        assert frames[i].std() > 5.0, f"frame {i} is degenerate (std={frames[i].std():.2f})"
        # frames must not be a single constant value
        assert len(np.unique(frames[i])) > 16


def test_ground_truth_depth_is_positive_and_finite() -> None:
    """Every non-zero GT depth pixel must be finite and strictly positive."""
    K = synth.default_intrinsics(SMALL[1], SMALL[0], 62.0)
    scene = synth.make_scene(1)
    for kind in SYNTHETIC_KINDS:
        poses = synth.make_trajectory(kind, n=3)
        _, depths, _ = synth.render_sequence(scene, poses, K, synth.DEFAULT_DIST, SMALL)
        valid = depths > 0
        assert valid.mean() > 0.5, f"{kind}: only {valid.mean():.1%} of pixels have depth"
        d = depths[valid]
        assert np.all(np.isfinite(d)), f"{kind}: non-finite depth"
        assert np.all(d > 0.0), f"{kind}: non-positive depth"
        # a plausible aerial depth range, not a degenerate constant
        assert 2.0 < np.median(d) < 400.0, f"{kind}: implausible median depth {np.median(d)}"
        assert d.std() > 1.0, f"{kind}: depth is nearly constant"


def test_grid_coverage_is_nearly_complete() -> None:
    """The camera-aware ground sampling should leave almost no holes."""
    K = synth.default_intrinsics(SMALL[1], SMALL[0], 62.0)
    scene = synth.make_scene(0)
    poses = synth.make_trajectory("forward", n=2)
    _, depths, _ = synth.render_sequence(scene, poses, K, synth.DEFAULT_DIST, SMALL)
    assert (depths > 0).mean() > 0.9


def test_gt_projection_matches_renderer_pixels() -> None:
    """Reprojecting GT 3D points must land on the renderer's own pixels.

    This re-implements the pinhole + radial-tangential model *independently*
    of :mod:`monogeo.synth` internals, so it genuinely validates the renderer's
    geometry rather than restating it.
    """
    K = synth.default_intrinsics(320, 96, 62.0)
    dist = synth.DEFAULT_DIST
    scene = synth.make_scene(2)
    pose = synth.make_trajectory("forward", n=1)[0]

    _, _, pix, points, _ = synth.render_frame(scene, pose, K, dist, (96, 320), splat=2)
    assert points.shape[0] > 1000
    assert pix.shape[0] == points.shape[0]

    X = points.astype(np.float64)
    R = pose[:3, :3]
    t = pose[:3, 3]
    Xc = X @ R.T + t
    z = Xc[:, 2]
    assert np.all(z > 0)

    # forward radial-tangential model, written out from the OpenCV docs
    k1, k2, p1, p2, k3 = dist
    x = Xc[:, 0] / z
    y = Xc[:, 1] / z
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    u = K[0, 0] * xd + K[0, 2]
    v = K[1, 1] * yd + K[1, 2]
    proj = np.stack([u, v], 1)

    err = np.linalg.norm(proj - pix, axis=1)
    assert err.max() < 1e-6, f"continuous projection mismatch: max {err.max():.3e} px"

    # and within 1 px of the *quantised* pixel that the renderer splatted
    qerr = np.linalg.norm(proj - np.rint(pix), axis=1)
    assert qerr.max() < 1.0, f"quantised projection mismatch: max {qerr.max():.3f} px"


def test_long_sequence_keeps_near_full_depth_coverage() -> None:
    """Every frame of a long flight must still have terrain in it.

    Regression test: an earlier renderer pre-sampled the ground from a few fixed
    anchors, so a camera that had flown past them saw its *near* field full of
    holes and, past about frame 100, a completely empty frame. The per-frame
    terrain raycast removes the failure mode; this test keeps it removed.
    """
    K = synth.default_intrinsics(416, 128, 62.0)
    scene = synth.make_scene(0)
    for kind in ("forward", "descent"):
        poses = synth.make_trajectory(kind, n=40)
        frames, depths, _ = synth.render_sequence(scene, poses, K, synth.DEFAULT_DIST, (128, 416))
        coverage = np.array([(depths[i] > 0).mean() for i in range(len(depths))])
        assert coverage.min() > 0.9, f"{kind}: coverage dropped to {coverage.min():.3f}"
        for i in range(0, len(frames), 8):
            assert frames[i].std() > 5.0, f"{kind} frame {i} is degenerate"
            assert np.median(depths[i][depths[i] > 0]) > 2.0


def test_lookat_pose_is_rigid_and_correct() -> None:
    """lookat poses must be orthonormal with +Z pointing at the target."""
    pos = np.array([1.0, 9.0, 5.0])
    target = pos + np.array([0.0, -3.0, 11.0])
    pose = synth.lookat_pose(pos, target)
    R = pose[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
    assert np.linalg.det(R) > 0
    # the target must sit on the optical axis in camera coordinates
    cam = R @ target + pose[:3, 3]
    assert abs(cam[0]) < 1e-9 and abs(cam[1]) < 1e-9 and cam[2] > 0


def test_trajectories_move_and_reject_unknown_kind() -> None:
    """Forward must advance, orbit must circle, descent must lose altitude."""
    for kind in SYNTHETIC_KINDS:
        poses = synth.make_trajectory(kind, n=8)
        centers = np.stack([np.linalg.inv(p)[:3, 3] for p in poses])
        assert np.all(np.isfinite(centers))
        assert np.linalg.norm(centers[-1] - centers[0]) > 1.0, kind
    descend = np.stack([np.linalg.inv(p)[:3, 3] for p in synth.make_trajectory("descent", n=8)])
    assert descend[-1, 1] < descend[0, 1]
    with pytest.raises(ValueError):
        synth.make_trajectory("barrel-roll", n=4)


def test_render_checkerboard_views_are_detectable() -> None:
    """Most calibration views must give findChessboardCorners 42 corners."""
    import cv2

    K = synth.default_intrinsics(640, 480, 62.0)
    images, objp, centers = synth.render_checkerboard_views(K, synth.DEFAULT_DIST, n=6, seed=0)
    assert len(images) == 6
    assert objp.shape == (42, 3)
    assert centers.shape == (6, 3)
    found = 0
    for img in images:
        assert img.std() > 5.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = cv2.findChessboardCorners(
            gray, (7, 6), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if ok:
            assert corners.reshape(-1, 2).shape == (42, 2)
            found += 1
    assert found >= 4, f"only {found}/6 boards detectable"


@pytest.mark.parametrize("kind", SYNTHETIC_KINDS)
def test_load_sequence_and_dataset(kind: str, tmp_path) -> None:
    """load_sequence is indexable and the dataset yields well-formed triplets."""
    seq = load_sequence(kind, size=SMALL, num_frames=8, root=tmp_path, cache=False, verbose=False)
    assert seq.source == "synthetic"
    frame = seq[0]
    assert frame.shape == (SMALL[0], SMALL[1], 3), frame.shape

    ds = SequenceDataset(seq, frame_ids=(0, -1, 1))
    assert len(ds) == 6
    item = ds[1]
    assert item["target"].shape == (3, SMALL[0], SMALL[1])
    assert item["sources"].shape == (2, 3, SMALL[0], SMALL[1])
    assert item["K"].shape == (3, 3)
    assert item["inv_K"].shape == (3, 3)
    assert item["depth"].shape == (1, SMALL[0], SMALL[1])
    assert item["pose"].shape == (4, 4)
    assert item["source_poses"].shape == (2, 4, 4)
    assert item["rel_poses"].shape == (2, 4, 4)
    assert float(item["target"].max()) <= 1.0
    assert float(item["depth"].max()) > 0.0


def test_dataset_rejects_too_short_sequence(tmp_path) -> None:
    """A 2-frame sequence cannot supply +/-1 neighbours."""
    seq = load_sequence(
        "forward", size=SMALL, num_frames=2, root=tmp_path, cache=False, verbose=False
    )
    with pytest.raises(ValueError):
        SequenceDataset(seq, frame_ids=(0, -1, 1))
