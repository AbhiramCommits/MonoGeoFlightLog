"""Tests for the differentiable camera model, the losses, and depth training."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from monogeo import geometry
from monogeo.cli import load_config
from monogeo.data import SequenceDataset, load_sequence
from monogeo.geometry import BackprojectDepth, Project3D, pixel_grid, scale_intrinsics
from monogeo.losses import (
    automask_min_reprojection,
    disparity_smoothness,
    photometric_error,
    photometric_loss,
    ssim,
)
from monogeo.models import DepthNet, PoseNet, disp_to_depth, pose_vec_to_mat
from monogeo.models.pose_net import skew

H, W = 48, 160


# --------------------------------------------------------------------------- #
# geometry round trips
# --------------------------------------------------------------------------- #
def test_backproject_project_round_trip() -> None:
    """Backprojecting then reprojecting with the identity pose must be exact.

    This is the core invariant of the differentiable camera model: the
    ``K^-1`` / ``K`` pair has to compose back to the pixel grid, otherwise every
    photometric gradient is silently wrong.
    """
    K = torch.tensor(
        [[120.0, 0.0, (W - 1) / 2.0], [0.0, 120.0, (H - 1) / 2.0], [0.0, 0.0, 1.0]]
    ).unsqueeze(0)
    inv_K = torch.linalg.inv(K)
    depth = torch.full((1, 1, H, W), 25.0)

    points = BackprojectDepth(H, W)(depth, inv_K)
    assert points.shape == (1, H * W, 3)
    # a 25 m plane must backproject to a plane at z = 25
    assert torch.allclose(points[..., 2], torch.full_like(points[..., 2], 25.0), atol=1e-3)

    coords = Project3D(H, W)(points, K, torch.eye(4).unsqueeze(0))
    assert coords.shape == (1, H, W, 2)

    grid = pixel_grid(1, H, W, torch.device("cpu"))[0]
    u = (coords[0, ..., 0] + 1.0) / 2.0 * (W - 1)
    v = (coords[0, ..., 1] + 1.0) / 2.0 * (H - 1)
    u_ref = grid[0].reshape(H, W)
    v_ref = grid[1].reshape(H, W)
    assert float((u - u_ref).abs().max()) < 1e-3
    assert float((v - v_ref).abs().max()) < 1e-3


def test_scale_intrinsics_matches_resized_projection() -> None:
    """Rescaling K must agree with projecting a resized image grid."""
    h, w = 4, 8
    K = torch.tensor(
        [[200.0, 0.0, (w - 1) / 2.0], [0.0, 200.0, (h - 1) / 2.0], [0.0, 0.0, 1.0]]
    ).unsqueeze(0)
    s = 0.5
    Ks = scale_intrinsics(K, s)
    assert Ks[0, 0, 0] == pytest.approx(100.0)
    assert Ks[0, 0, 2] == pytest.approx((K[0, 0, 2] + 0.5) * s - 0.5)
    # the input must not be mutated
    assert K[0, 0, 0] == pytest.approx(200.0)

    # a point on the optical axis must land on the principal point
    depth = torch.full((1, 1, h, w), 3.0)
    pts = BackprojectDepth(h, w)(depth, torch.linalg.inv(Ks))
    coords = Project3D(h, w)(pts, Ks, torch.eye(4).unsqueeze(0))
    assert coords.shape == (1, h, w, 2)
    # the centre pixel row of pixels must map back to itself after the round trip
    grid = pixel_grid(1, h, w, torch.device("cpu"))[0]
    u = (coords[0, ..., 0] + 1.0) / 2.0 * (w - 1)
    v = (coords[0, ..., 1] + 1.0) / 2.0 * (h - 1)
    assert float((u - grid[0].reshape(h, w)).abs().max()) < 1e-4
    assert float((v - grid[1].reshape(h, w)).abs().max()) < 1e-4


def test_project3d_pushes_behind_camera_points_out_of_frame() -> None:
    """Points behind the camera must not alias onto valid pixels."""
    h, w = 8, 16
    K = torch.tensor([[100.0, 0, (w - 1) / 2.0], [0, 100.0, (h - 1) / 2.0], [0, 0, 1.0]]).unsqueeze(
        0
    )
    # a full grid of in-front points, with the middle column pushed behind
    pts = torch.zeros(1, h * w, 3)
    pts[..., 2] = 5.0
    pts[:, (h // 2) * w + (w // 2), 2] = -5.0
    coords = Project3D(h, w)(pts, K, torch.eye(4).unsqueeze(0))
    assert coords.shape == (1, h, w, 2)

    coords_out = coords.reshape(1, h * w, 2)
    front = torch.ones(h * w, dtype=torch.bool)
    front[(h // 2) * w + (w // 2)] = False
    assert float(coords_out[0][front].abs().max()) <= 1.0 + 1e-6
    assert float(coords_out[0][~front].abs().max()) > 1.0


def test_pose_vec_to_mat_matches_scipy_rodrigues() -> None:
    """pose_vec_to_mat must reproduce scipy's rotation vectors exactly."""
    from scipy.spatial.transform import Rotation

    for axis, angle in [
        ((0, 0, 1), 0.1),
        ((1, 0, 0), 0.5),
        ((0.3, 0.4, 0.5), 1.2),
        ((0, 0, 1), 1e-9),
    ]:
        a = np.array(axis, float)
        a = a / np.linalg.norm(a) * angle
        vec = torch.zeros(1, 6)
        vec[0, :3] = torch.tensor(a, dtype=torch.float32)
        T = pose_vec_to_mat(vec)
        R_ref = torch.tensor(Rotation.from_rotvec(a).as_matrix(), dtype=torch.float32)
        assert float((T[0, :3, :3] - R_ref).abs().max()) < 1e-5
        assert float(torch.det(T[0, :3, :3])) == pytest.approx(1.0, abs=1e-5)
    # translation goes through untouched
    vec = torch.zeros(1, 6)
    vec[0, 3:] = torch.tensor([1.0, -2.0, 3.0])
    T = pose_vec_to_mat(vec)
    assert torch.allclose(T[0, :3, :3], torch.eye(3), atol=1e-6)
    assert torch.allclose(T[0, :3, 3], torch.tensor([1.0, -2.0, 3.0]))


def test_skew_matches_cross_product() -> None:
    """[v]_x w must equal v x w."""
    v = torch.tensor([[0.3, -1.2, 2.0]])
    w = torch.tensor([[[1.0, 0.5, -0.25]]])
    assert torch.allclose(
        (skew(v) @ w.transpose(1, 2)).transpose(1, 2),
        torch.cross(v.unsqueeze(1), w, dim=-1),
        atol=1e-6,
    )


def test_pose_vec_to_mat_is_differentiable_at_identity() -> None:
    """Gradients must be finite at zero rotation (the training start point)."""
    vec = torch.zeros(2, 6, requires_grad=True)
    T = pose_vec_to_mat(vec)
    grad = torch.autograd.grad(T.sum(), vec)[0]
    assert torch.isfinite(grad).all()


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
def test_ssim_is_one_for_identical_images() -> None:
    """SSIM of an image with itself must be 1 and drop for a corrupted copy."""
    x = torch.rand(2, 3, 32, 64)
    assert float(ssim(x, x).mean()) == pytest.approx(1.0, abs=1e-4)
    y = torch.clamp(x + 0.5, 0, 1)
    assert float(ssim(x, y).mean()) < float(ssim(x, x).mean())
    assert ssim(x, x).shape == (2, 1, 32, 64)


def test_photometric_error_is_zero_for_identical_and_bounded() -> None:
    """The photometric error must vanish on identical images and stay in [0, 1]."""
    x = torch.rand(2, 3, 32, 64)
    e = photometric_error(x, x)
    assert float(e.max()) < 1e-4
    e2 = photometric_error(x, torch.zeros_like(x))
    assert float(e2.min()) >= 0.0
    assert float(e2.max()) <= 1.0


def test_photometric_loss_automask_excludes_identity_winners() -> None:
    """Auto-masking must discard pixels the unwarped source already explains."""
    target = torch.rand(1, 3, 16, 32)
    # (B, S, 3, H, W): two sources that already match the target perfectly
    sources = target.unsqueeze(1).repeat(1, 2, 1, 1, 1)
    warped = [torch.rand(1, 3, 16, 32) for _ in range(2)]  # bad warps
    out = photometric_loss(target, warped, unwarped_sources=sources, use_automask=True)
    assert float(out["mask_fraction"]) < 0.01, "static pixels should be masked out"
    # without the auto-mask the same inputs must produce a large loss
    out2 = photometric_loss(target, warped, unwarped_sources=sources, use_automask=False)
    assert float(out2["loss"]) > float(out["loss"])
    assert float(out2["mask_fraction"]) == pytest.approx(1.0)


def test_automask_prefers_the_warp_when_it_helps() -> None:
    """When warping is the better explanation, the mask must keep the pixels."""
    target = torch.rand(1, 3, 16, 32)
    # unwarped sources are noise, warped sources reproduce the target exactly
    unwarped = torch.rand(1, 2, 3, 16, 32)
    good = [target.clone(), target.clone()]
    gen = torch.Generator().manual_seed(0)
    loss, mask = automask_min_reprojection(target, good, unwarped, generator=gen)
    assert float(mask.mean()) == pytest.approx(1.0)
    assert float(loss.mean()) < 1e-4


def test_disparity_smoothness_is_edge_aware() -> None:
    """A gradient across an image edge must be penalised much less than a flat ramp."""
    img_edge = torch.zeros(1, 3, 16, 16)
    img_edge[:, :, :, 8:] = 1.0
    disp_edge = torch.zeros(1, 1, 16, 16)
    disp_edge[:, :, :, 8:] = 1.0

    img_flat = torch.full((1, 3, 16, 16), 0.5)
    loss_edge = disparity_smoothness(disp_edge, img_edge)
    loss_flat = disparity_smoothness(disp_edge, img_flat)
    assert float(loss_edge) < float(loss_flat)


def test_disp_to_depth_is_monotone_and_bounded() -> None:
    """disp_to_depth must be decreasing in disparity and respect the bounds."""
    disp = torch.linspace(0.01, 0.99, 32).view(1, 1, 4, 8)
    depth = disp_to_depth(disp, 0.5, 120.0)
    assert depth.min() >= 0.5 - 1e-3
    assert depth.max() <= 120.0 + 1e-3
    d = depth.flatten()
    assert torch.all(d[1:] < d[:-1])


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def test_depth_net_output_shapes_and_range() -> None:
    """DepthNet must emit the requested scales and a bounded depth map."""
    net = DepthNet(num_scales=4, min_depth=1.0, max_depth=100.0)
    disps, depth = net(torch.rand(2, 3, 64, 192))
    assert len(disps) == 4
    assert depth.shape == (2, 1, 64, 192)
    assert float(depth.detach().min()) >= 1.0 - 1e-4
    assert float(depth.detach().max()) <= 100.0 + 1e-4
    for s, d in zip((1, 2, 4, 8), disps, strict=False):
        assert d.shape[-2:] == (64 // s, 192 // s)
        assert float(d.detach().min()) >= 0.0 and float(d.detach().max()) <= 1.0


def test_pose_net_outputs_rigid_transforms() -> None:
    """PoseNet must produce proper rigid transforms."""
    net = PoseNet()
    T = net(torch.rand(2, 3, 64, 192), torch.rand(2, 3, 64, 192))
    assert T.shape == (2, 4, 4)
    R = T[:, :3, :3]
    assert torch.allclose(R @ R.transpose(1, 2), torch.eye(3).expand(2, 3, 3), atol=1e-4)
    assert torch.allclose(T[:, 3, 3], torch.ones(2), atol=1e-6)


# --------------------------------------------------------------------------- #
# warp direction: the ground-truth sanity check
# --------------------------------------------------------------------------- #
def test_warp_with_gt_depth_and_pose_matches_the_target(tmp_path) -> None:
    """Warping a source with GT depth + GT pose must beat a wrong pose.

    This pins down the *direction* of the transform: it is easy to build a warp
    that runs backwards, which still trains (to a poor optimum) but produces
    nonsense depth. Using the ground truth from the renderer catches it.
    """
    seq = load_sequence(
        "forward", size=(96, 320), num_frames=6, root=tmp_path, cache=False, verbose=False
    )
    ds = SequenceDataset(seq, frame_ids=(0, -1, 1))
    item = ds[2]
    S, _, h, w = item["sources"].shape
    target = item["target"].unsqueeze(0)
    sources = item["sources"].unsqueeze(0)
    depth = item["depth"].unsqueeze(0)
    K = item["K"].unsqueeze(0)
    inv_K = item["inv_K"].unsqueeze(0)
    rel = item["rel_poses"].unsqueeze(0)  # target camera -> source camera

    backproject = BackprojectDepth(h, w)
    project = Project3D(h, w)
    points = backproject(depth, inv_K)  # target pixels -> target-camera points

    for i in range(S):
        # points are in the *target* frame, so we project them into the
        # *source* frame: that transform is exactly rel_poses[i] (no inverse).
        coords = project(points, K, rel[:, i])
        warped = geometry.warp_image(sources[:, i], coords)
        err_gt = float(photometric_error(target, warped).mean())
        err_unwarped = float(photometric_error(target, sources[:, i]).mean())

        # deliberately wrong direction: source -> target
        coords_bad = project(points, K, torch.linalg.inv(rel[:, i]))
        err_bad = float(
            photometric_error(target, geometry.warp_image(sources[:, i], coords_bad)).mean()
        )

        assert (
            err_gt < err_unwarped
        ), f"GT warp ({err_gt:.4f}) must beat leaving the source unwarped ({err_unwarped:.4f})"
        assert (
            err_gt < err_bad
        ), f"GT warp ({err_gt:.4f}) must beat the reversed warp ({err_bad:.4f})"
        assert err_gt < 0.15, f"GT warp error {err_gt:.4f} is implausibly large"


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def test_overfit_two_triplets_reduces_photometric_loss(tmp_path) -> None:
    """30 optimisation steps on 2 triplets must cut the photometric loss by >30%.

    The two triplets are near-duplicates, so this is a pure optimiser/plumbing
    test: if gradients do not reach the depth and pose heads through the
    differentiable camera model, the loss cannot fall.

    Auto-masking is switched *off* here on purpose. The mask is recomputed every
    step from the current model, so as the warp improves it admits progressively
    harder pixels and the reported masked mean can rise even while the model is
    getting better. That moving target makes the raw number a poor unit test;
    without it the same quantity is measured on a fixed pixel set at every step.
    """
    cfg = load_config("configs/default.yaml")
    cfg["train_depth"]["use_automask"] = False
    torch.manual_seed(0)
    seq = load_sequence(
        "forward", size=(64, 192), num_frames=8, root=tmp_path, cache=False, verbose=False
    )
    ds = SequenceDataset(seq, frame_ids=(0, -1, 1), indices=[2, 3, 2, 3])
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    from monogeo.train_depth import DepthTrainer

    trainer = DepthTrainer(cfg, device="cpu", seed=0)
    params = list(trainer.depth_net.parameters()) + list(trainer.pose_net.parameters())
    # same optimiser and gradient clipping as the real training loop
    opt = torch.optim.Adam(params, lr=float(cfg["train_depth"]["lr"]))

    with torch.no_grad():
        first = float(trainer.compute_losses(batch)["photometric"])
    losses = []
    for _step in range(30):
        opt.zero_grad(set_to_none=True)
        out = trainer.compute_losses(batch)
        out["total"].backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
        opt.step()
        losses.append(float(out["photometric"].detach()))

    last = float(torch.tensor(losses[-3:]).mean())
    assert last < 0.7 * first, f"photometric loss {first:.4f} -> {last:.5f} (need >30% reduction)"


def test_compute_losses_is_finite_and_backpropagates(tmp_path) -> None:
    """One training step must produce finite gradients on both networks."""
    cfg = load_config("configs/default.yaml")
    seq = load_sequence(
        "forward", size=(64, 192), num_frames=6, root=tmp_path, cache=False, verbose=False
    )
    ds = SequenceDataset(seq, frame_ids=(0, -1, 1), indices=[2])
    batch = ds[0]
    for k in batch:
        if torch.is_tensor(batch[k]) and batch[k].is_floating_point():
            batch[k] = batch[k].unsqueeze(0)

    from monogeo.train_depth import DepthTrainer

    trainer = DepthTrainer(cfg, device="cpu", seed=0)
    out = trainer.compute_losses(batch)
    for key in ("total", "photometric", "smoothness", "mask_fraction"):
        assert torch.isfinite(out[key]), f"{key} is not finite"
    out["total"].backward()

    depth_grads = [p.grad for p in trainer.depth_net.parameters() if p.grad is not None]
    pose_grads = [p.grad for p in trainer.pose_net.parameters() if p.grad is not None]
    assert depth_grads and all(torch.isfinite(g).all() for g in depth_grads)
    assert pose_grads and all(torch.isfinite(g).all() for g in pose_grads)
    assert any(float(g.abs().sum()) > 0 for g in depth_grads)


def test_dataset_rel_poses_match_the_geometry_convention(tmp_path) -> None:
    """``rel_poses`` must satisfy ``X_source = R X_target + t``."""
    seq = load_sequence(
        "forward", size=(64, 192), num_frames=6, root=tmp_path, cache=False, verbose=False
    )
    item = SequenceDataset(seq, frame_ids=(0, -1, 1))[2]
    pose = item["pose"].numpy()
    for i, sp in enumerate(item["source_poses"].numpy()):
        rel = item["rel_poses"][i].numpy()
        np.testing.assert_allclose(rel, sp @ np.linalg.inv(pose), atol=1e-5)
        X = np.array([0.3, -0.2, 12.0])
        np.testing.assert_allclose(
            rel[:3, :3] @ X + rel[:3, 3],
            (sp @ np.linalg.inv(pose))[:3, :3] @ X + (sp @ np.linalg.inv(pose))[:3, 3],
            atol=1e-5,
        )
