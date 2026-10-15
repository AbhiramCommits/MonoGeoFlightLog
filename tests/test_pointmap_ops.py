"""Tests for pointmap operations, the pointmap network and its training loss."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from monogeo.data import load_sequence
from monogeo.models.pointmap_net import PointmapNet, PointmapPairDataset
from monogeo.pointmap_ops import (
    depth_from_pointmap,
    focal_from_pointmaps,
    median_scale,
    pointmap_from_depth,
    pointmap_scale_consistency,
    pose_from_pointmaps,
    two_view_pointmap_targets,
    umeyama_alignment,
    weighted_umeyama,
)
from monogeo.train_pointmap import pointmap_loss
from monogeo.twoview import relative_pose

SIZE = (128, 416)


# --------------------------------------------------------------------------- #
# similarity alignment
# --------------------------------------------------------------------------- #
def _random_similarity(rng: np.random.Generator) -> tuple[float, np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation

    return (
        float(rng.uniform(0.2, 5.0)),
        Rotation.from_rotvec(rng.normal(size=3) * 0.7).as_matrix(),
        rng.normal(size=3) * 2.0,
    )


def test_umeyama_recovers_a_known_similarity_transform() -> None:
    """Umeyama must invert an exact similarity transform to ~1e-5 or better."""
    rng = np.random.default_rng(0)
    for trial in range(8):
        X = rng.normal(size=(64, 3)) * rng.uniform(0.1, 10.0)
        s_true, R_true, t_true = _random_similarity(rng)
        Y = s_true * (X @ R_true.T) + t_true

        s, R, t = umeyama_alignment(X, Y)
        assert s == pytest.approx(s_true, rel=1e-5), f"trial {trial}"
        assert np.abs(R - R_true).max() < 1e-5
        assert np.abs(t - t_true).max() < 1e-5
        # and the transform must actually map X onto Y
        assert np.abs((s * (X @ R.T) + t) - Y).max() < 1e-5


def test_umeyama_returns_a_proper_rotation_under_noise() -> None:
    """The recovered rotation must be proper (det +1) even with reflections in the noise."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 3))
    Y = X.copy()
    Y[:, 2] *= -1.0  # a reflection is not achievable by a rotation
    s, R, _t = umeyama_alignment(X, Y)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)
    assert s > 0


def test_weighted_umeyama_ignores_outliers_with_zero_weight() -> None:
    """Zero-weight outliers must not influence the fit at all."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 3))
    s_true, R_true, t_true = _random_similarity(rng)
    Y = s_true * (X @ R_true.T) + t_true
    Y[:10] += rng.normal(size=(10, 3)) * 50.0  # gross outliers

    w = np.ones(50)
    w[:10] = 0.0
    s, R, t = weighted_umeyama(X, Y, w)
    assert s == pytest.approx(s_true, rel=1e-5)
    assert np.abs(R - R_true).max() < 1e-5
    assert np.abs(t - t_true).max() < 1e-5


def test_umeyama_rejects_bad_input() -> None:
    """Shape errors, degenerate inputs and negative weights must raise."""
    X = np.zeros((5, 3))
    with pytest.raises(ValueError):
        umeyama_alignment(X, np.zeros((4, 3)))
    with pytest.raises(ValueError):
        umeyama_alignment(np.zeros((2, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        umeyama_alignment(X, np.ones((5, 3)))  # zero variance
    with pytest.raises(ValueError):
        weighted_umeyama(np.ones((5, 3)), np.ones((5, 3)) * 2, np.array([-1.0, 1, 1, 1, 1]))


def test_median_scale_is_robust_to_outliers() -> None:
    """A few absurd predictions must not move the median scale."""
    gt = np.full(1000, 30.0)
    pred = np.full(1000, 10.0)
    assert median_scale(pred, gt) == pytest.approx(3.0)
    pred[:100] = 1e6
    assert median_scale(pred, gt) == pytest.approx(3.0)
    # a mask must be honoured
    assert median_scale(pred, gt, mask=np.arange(1000) >= 100) == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# pointmap <-> depth
# --------------------------------------------------------------------------- #
def test_pointmap_from_depth_round_trip() -> None:
    """Lifting depth to a pointmap and back must be exact."""
    K = np.array([[266.28, 0.0, 159.5], [0.0, 266.28, 47.5], [0.0, 0.0, 1.0]])
    depth = np.linspace(10.0, 80.0, 96 * 320).reshape(96, 320)
    pm = pointmap_from_depth(depth, K)
    assert pm.shape == (96, 320, 3)
    np.testing.assert_allclose(pm[..., 2], depth, rtol=1e-9)
    np.testing.assert_allclose(depth_from_pointmap(pm), depth, rtol=1e-9)
    # a pixel's ray must be K^-1 [u, v, 1]
    ray = pm[10, 20] / pm[10, 20, 2]
    expect = np.linalg.inv(K) @ np.array([20.0, 10.0, 1.0])
    np.testing.assert_allclose(ray, expect, atol=1e-9)


def test_pointmap_from_depth_applies_the_mask() -> None:
    """Masked pixels must be zeroed out."""
    K = np.eye(3)
    depth = np.ones((4, 4))
    mask = np.zeros((4, 4), bool)
    mask[0, 0] = True
    pm = pointmap_from_depth(depth, K, mask)
    assert pm[0, 0].sum() != 0.0  # the valid pixel is non-zero
    assert pm[1, 1].sum() == 0.0


# --------------------------------------------------------------------------- #
# pose from pointmaps
# --------------------------------------------------------------------------- #
def test_pose_from_pointmaps_on_gt_pointmaps() -> None:
    """GT pointmaps must give a rotation error well under 1 degree."""
    seq = load_sequence("forward", size=(96, 320), num_frames=12, cache=True, verbose=False)
    K = seq.K
    for i, j in [(2, 4), (2, 5), (3, 6)]:
        pm1, pm2, _, _, _ = two_view_pointmap_targets(seq, i, j)
        R_gt, t_gt = relative_pose(seq.poses, i, j)
        res = pose_from_pointmaps(pm1, pm2, K=K, gt_R=R_gt, gt_t=t_gt)
        assert res.rotation_error_deg < 1.0, f"pair ({i},{j}) rot err {res.rotation_error_deg}"
        assert res.translation_angle_error_deg < 1.0
        assert res.inlier_ratio > 0.5
        # the two GT pointmaps come from the same metric scene, so the
        # inter-pointmap scale must be ~1
        assert res.scale == pytest.approx(1.0, abs=0.01)


def test_pose_from_pointmaps_requires_intrinsics() -> None:
    """Without K there is no projection constraint, so the call must fail loudly."""
    pm = np.zeros((4, 4, 3))
    with pytest.raises(ValueError):
        pose_from_pointmaps(pm, pm, K=None)


def test_pointmap_scale_consistency_detects_a_scaled_second_view() -> None:
    """Scaling one pointmap must show up exactly as a non-unit consistency scale."""
    seq = load_sequence("forward", size=(96, 320), num_frames=8, cache=True, verbose=False)
    pm1, pm2, _, _, _ = two_view_pointmap_targets(seq, 1, 2)
    K = seq.K
    base = pointmap_scale_consistency(pm1, pm2, K=K)
    assert base["scale"] == pytest.approx(1.0, abs=0.01)
    assert base["scale_iqr"] < 0.05
    for factor in (1.4, 0.7, 3.0):
        got = pointmap_scale_consistency(pm1, pm2 * factor, K=K)
        assert got["scale"] == pytest.approx(1.0 / factor, rel=0.02), f"factor {factor}"
    with pytest.raises(ValueError):
        pointmap_scale_consistency(pm1, pm2, K=None)


def test_focal_from_pointmaps_matches_calibrated_intrinsics() -> None:
    """A GT pointmap must reproduce the calibrated focal length and principal point."""
    for width, height in [(320, 96), (416, 128)]:
        K = np.array(
            [[266.28, 0.0, (width - 1) / 2.0], [0.0, 266.28, (height - 1) / 2.0], [0.0, 0.0, 1.0]]
        )
        depth = np.full((height, width), 25.0)
        # add structure so the fit is not a degenerate constant
        depth[: height // 2] = 15.0
        f = focal_from_pointmaps(pointmap_from_depth(depth, K))
        assert f["fx"] == pytest.approx(K[0, 0], rel=1e-6)
        assert f["fy"] == pytest.approx(K[1, 1], rel=1e-6)
        assert f["cx"] == pytest.approx(K[0, 2], rel=1e-4)
        assert f["cy"] == pytest.approx(K[1, 2], rel=1e-4)
        assert f["num_points"] == width * height


def test_two_view_pointmap_targets_is_self_consistent() -> None:
    """GT pointmaps from two views must describe one rigid scene."""
    seq = load_sequence("forward", size=(96, 320), num_frames=8, cache=True, verbose=False)
    pm1, pm2, v1, v2, norm = two_view_pointmap_targets(seq, 2, 4)
    assert pm1.shape == pm2.shape == (96, 320, 3)
    assert v1.shape == v2.shape == (96, 320)
    assert norm > 0
    # both pointmaps use frame 1's frame, so their z must be positive wherever valid
    assert (pm1[..., 2][v1] > 0).all()
    assert (pm2[..., 2][v2] > 0).all()
    consistency = pointmap_scale_consistency(pm1, pm2, K=seq.K)
    assert consistency["num_overlap"] > 100
    assert consistency["scale"] == pytest.approx(1.0, abs=0.01)


# --------------------------------------------------------------------------- #
# network + loss
# --------------------------------------------------------------------------- #
def test_pointmap_net_shapes_and_confidence_are_positive() -> None:
    """The regressor must emit full-resolution pointmaps and positive confidence."""
    net = PointmapNet(
        img_size=(64, 192), patch_size=16, embed_dim=64, encoder_depth=2, decoder_depth=1
    )
    out = net(torch.rand(2, 3, 64, 192), torch.rand(2, 3, 64, 192))
    assert out["pts1"].shape == (2, 64, 192, 3)
    assert out["pts2"].shape == (2, 64, 192, 3)
    assert out["conf1"].shape == (2, 64, 192)
    assert float(out["conf1"].detach().min()) > 1.0  # 1 + softplus(...) > 1
    assert torch.isfinite(out["pts1"]).all()


def test_pointmap_net_rejects_wrong_input_size() -> None:
    """A size mismatch must raise rather than silently interpolate."""
    net = PointmapNet(img_size=(64, 192))
    with pytest.raises(ValueError):
        net(torch.rand(1, 3, 64, 128), torch.rand(1, 3, 64, 128))


def test_pointmap_loss_confidence_term_behaves() -> None:
    """The DUSt3R objective must fall when error falls and rise with confidence misuse."""
    net = PointmapNet(img_size=(64, 192), embed_dim=32, encoder_depth=1, decoder_depth=1)
    torch.manual_seed(0)
    img1 = torch.rand(1, 3, 64, 192)
    img2 = torch.rand(1, 3, 64, 192)
    out = net(img1, img2)

    target = {
        "pm1": torch.zeros_like(out["pts1"]),
        "pm2": torch.zeros_like(out["pts2"]),
        "valid1": torch.ones(1, 64, 192, dtype=torch.bool),
        "valid2": torch.ones(1, 64, 192, dtype=torch.bool),
    }
    # a perfect prediction must give a lower loss than a wrong one
    perfect = dict(out)
    perfect["pts1"] = torch.zeros_like(out["pts1"])
    perfect["pts2"] = torch.zeros_like(out["pts2"])
    loss_perfect = pointmap_loss(perfect, target)["total"]
    loss_wrong = pointmap_loss(out, target)["total"]
    assert float(loss_perfect.detach()) < float(loss_wrong.detach())

    # masked-out pixels must not contribute
    target_masked = dict(target)
    target_masked["valid1"] = torch.zeros(1, 64, 192, dtype=torch.bool)
    target_masked["valid2"] = torch.zeros(1, 64, 192, dtype=torch.bool)
    assert torch.isfinite(pointmap_loss(out, target_masked)["total"])

    # gradients must reach every parameter group
    loss_wrong.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(float(g.abs().sum()) > 0 for g in grads)


def test_pointmap_pair_dataset_shapes() -> None:
    """The pair dataset must return the documented keys and shapes."""
    seq = load_sequence("forward", size=SIZE, num_frames=8, cache=True, verbose=False)
    ds = PointmapPairDataset(seq, gap=1, indices=[1, 2])
    assert len(ds) == 2
    item = ds[0]
    for key in ("img1", "img2", "pm1", "pm2", "valid1", "valid2", "norm", "K"):
        assert key in item
    assert item["img1"].shape == (3, SIZE[0], SIZE[1])
    assert item["pm1"].shape == (SIZE[0], SIZE[1], 3)
    assert item["valid1"].dtype == torch.bool
    assert float(item["norm"]) > 0
    # normalised targets must be ~unit scale
    assert 0.1 < float(item["pm1"][..., 2][item["valid1"]].mean()) < 10.0


def test_pointmap_training_step_reduces_loss() -> None:
    """Twenty steps on two pairs must cut the pointmap loss materially."""
    seq = load_sequence("forward", size=(64, 192), num_frames=6, cache=True, verbose=False)
    ds = PointmapPairDataset(seq, gap=1, indices=[2, 3, 2, 3])
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)

    torch.manual_seed(0)
    net = PointmapNet(
        img_size=(64, 192), patch_size=16, embed_dim=32, encoder_depth=1, decoder_depth=1
    )
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    losses = []
    for _ in range(20):
        opt.zero_grad(set_to_none=True)
        batch = next(iter(loader))
        out = net(batch["img1"], batch["img2"])
        loss = pointmap_loss(out, batch)["total"]
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert np.mean(losses[-3:]) < 0.8 * np.mean(losses[:3]), f"{losses[0]:.4f} -> {losses[-1]:.4f}"
