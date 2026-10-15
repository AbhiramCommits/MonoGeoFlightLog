"""Three-way comparison of the geometry families, with explicit scale analysis.

All three methods are run on the *same* held-out sequence, at the same
resolution, on the same frame pairs, and scored with the same metrics:

* **classical_two_view** -- SIFT + essential matrix + triangulation. Its depth
  output is a sparse point cloud; the missing monocular scale is recovered from
  ground truth so the *rest* of the metrics are comparable.
* **self_supervised_monodepth** -- the trained depth and pose networks, which
  never saw ground-truth depth or pose.
* **pointmap** -- the trained two-view pointmap regressor, scored via its
  pointmap (depth = ``z``) and its PnP-recovered pose.

The ``scale_analysis`` block answers three questions that a single leaderboard
number hides:

1. how much scale does each method need (per-frame factor and its drift), and
2. what does alignment cost, comparing no alignment / median scaling / a full
   Umeyama similarity, and
3. what focal length does each method imply, against the calibrated one.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .cli import cfg_get, save_json
from .data import load_sequence, slice_sequence, splits
from .metrics import aligned_depth_metrics, depth_metrics
from .models import DepthNet, PoseNet
from .models.pointmap_net import PointmapNet
from .pointmap_ops import (
    depth_from_pointmap,
    focal_from_pointmaps,
    pose_from_pointmaps,
)
from .twoview import (
    depth_from_triangulation,
    estimate_pose,
    match_features,
    pose_error,
    relative_pose,
    scale_from_gt,
    triangulate,
)

__all__ = ["load_depth_model", "load_pointmap_model", "load_pose_model", "run_compare"]


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #
def load_depth_model(cfg: dict[str, Any], device: str = "cpu") -> DepthNet:
    """Load the trained depth network from ``runs/depth/depth.pt``.

    Args:
        cfg: Configuration dict.
        device: Torch device string.

    Returns:
        The depth network in eval mode.

    Raises:
        FileNotFoundError: If the checkpoint is missing (run ``monogeo train-depth``).
    """
    path = Path(cfg_get(cfg, "train_depth.out_dir", "runs/depth")) / "depth.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `monogeo train-depth` first")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = DepthNet(
        num_layers=int(ckpt.get("num_layers", 18)),
        num_scales=int(ckpt.get("num_scales", 4)),
        min_depth=float(ckpt.get("min_depth", 0.5)),
        max_depth=float(ckpt.get("max_depth", 120.0)),
    )
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    net.to(device)
    return net


def load_pose_model(cfg: dict[str, Any], device: str = "cpu") -> PoseNet:
    """Load the trained pose network from ``runs/depth/pose.pt``.

    Args:
        cfg: Configuration dict.
        device: Torch device string.

    Returns:
        The pose network in eval mode.

    Raises:
        FileNotFoundError: If the checkpoint is missing.
    """
    path = Path(cfg_get(cfg, "train_depth.out_dir", "runs/depth")) / "pose.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `monogeo train-depth` first")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = PoseNet(
        num_layers=int(ckpt.get("num_layers", 18)),
        translation_scale=float(ckpt.get("translation_scale", 1.0)),
    )
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    net.to(device)
    return net


def load_pointmap_model(cfg: dict[str, Any], device: str = "cpu") -> tuple[PointmapNet, str]:
    """Load the trained pointmap regressor from ``runs/pointmap/pointmap.pt``.

    Args:
        cfg: Configuration dict.
        device: Torch device string.

    Returns:
        ``(model, provenance)``.

    Raises:
        FileNotFoundError: If the checkpoint is missing.
    """
    path = Path(cfg_get(cfg, "train_pointmap.out_dir", "runs/pointmap")) / "pointmap.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `monogeo train-pointmap` first")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    size = tuple(int(v) for v in ckpt["img_size"])
    net = PointmapNet(
        img_size=size,  # type: ignore[arg-type]
        patch_size=int(ckpt.get("patch_size", 16)),
        embed_dim=int(ckpt.get("embed_dim", 128)),
        encoder_depth=int(cfg_get(cfg, "pointmap.encoder_depth", 4)),
        decoder_depth=int(cfg_get(cfg, "pointmap.decoder_depth", 2)),
        num_heads=int(cfg_get(cfg, "pointmap.num_heads", 4)),
        head_hidden=int(cfg_get(cfg, "pointmap.head_hidden", 128)),
    )
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    net.to(device)
    return net, str(ckpt.get("provenance", "compact (trained here)"))


# --------------------------------------------------------------------------- #
# per-method evaluation
# --------------------------------------------------------------------------- #
def _tensor_frame(frame: np.ndarray) -> torch.Tensor:
    """``(H, W, 3)`` uint8 RGB -> ``(1, 3, H, W)`` float tensor in ``[0, 1]``."""
    return torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Median-aggregate a list of per-pair records field by field."""
    if not records:
        return {}

    def collect(path: str) -> list[float]:
        """Gather one (possibly nested) field from every record, skipping non-finite ones."""
        out = []
        for r in records:
            cur: Any = r
            for part in path.split("."):
                if not isinstance(cur, dict) or part not in cur:
                    cur = None
                    break
                cur = cur[part]
            if isinstance(cur, (int, float)) and np.isfinite(cur):
                out.append(float(cur))
        return out

    keys = [
        "depth_raw.abs_rel",
        "depth_raw.sq_rel",
        "depth_raw.rmse",
        "depth_raw.rmse_log",
        "depth_raw.delta1",
        "depth_raw.delta2",
        "depth_raw.delta3",
        "depth_median_scaled.abs_rel",
        "depth_median_scaled.sq_rel",
        "depth_median_scaled.rmse",
        "depth_median_scaled.rmse_log",
        "depth_median_scaled.delta1",
        "depth_median_scaled.delta2",
        "depth_median_scaled.delta3",
        "depth_umeyama.abs_rel",
        "depth_umeyama.rmse",
        "depth_umeyama.delta1",
        "alignment.scale",
        "alignment.rotation_deg",
        "alignment.translation",
        "pose.rotation_deg",
        "pose.translation_deg",
        "runtime_ms",
        "scale_factor",
        "depth_coverage",
    ]
    agg: dict[str, Any] = {}
    for key in keys:
        vals = collect(key)
        agg[key] = float(np.median(vals)) if vals else float("nan")
    agg["num_pairs"] = len(records)
    scale_vals = collect("scale_factor")
    if scale_vals:
        arr = np.asarray(scale_vals)
        agg["scale_drift_std_over_mean"] = (
            float(arr.std() / abs(arr.mean())) if abs(arr.mean()) > 1e-9 else float("nan")
        )
        agg["scale_drift_std"] = float(arr.std())
    return agg


def evaluate_classical(
    cfg: dict[str, Any],
    seq: Any,
    pairs: list[tuple[int, int]],
    min_depth: float,
    max_depth: float,
    scale_from_gt_ok: bool = True,
) -> dict[str, Any]:
    """Run the classical two-view pipeline over a list of frame pairs.

    Args:
        cfg: Configuration dict.
        seq: The sequence to evaluate on.
        pairs: ``(i, j)`` frame index pairs.
        min_depth: Metric lower bound on ground-truth depth.
        max_depth: Metric upper bound.
        scale_from_gt_ok: Recover the missing scale from ground truth (the
            honest way to make a monocular triangulation metric-comparable).

    Returns:
        A dict with ``per_pair`` records and ``aggregate`` medians.
    """
    ratio = float(cfg_get(cfg, "twoview.ratio_thresh", 0.75))
    max_features = int(cfg_get(cfg, "twoview.max_features", 4000))
    thresh = float(cfg_get(cfg, "twoview.ransac_threshold_px", 1.0))
    size = seq.depths.shape[1:]

    records = []
    for i, j in pairs:
        t0 = time.perf_counter()
        rec: dict[str, Any] = {"pair": [int(i), int(j)]}
        matches = match_features(
            seq.frames[i], seq.frames[j], max_features=max_features, ratio=ratio
        )
        rec["num_matches"] = int(len(matches))
        if len(matches) < 8:
            rec["ok"] = False
            rec["runtime_ms"] = (time.perf_counter() - t0) * 1000.0
            records.append(rec)
            continue
        tv = estimate_pose(matches, seq.K, threshold_px=thresh)
        pts3d, valid = triangulate(matches, seq.K, tv.R, tv.t, mask=tv.inlier_mask)

        R_gt, t_gt = relative_pose(seq.poses, i, j)
        perr = pose_error(tv.R, tv.t, R_gt, t_gt)
        R_est_full, t_est_full = tv.R, tv.t

        gt = seq.depths[i]
        gt_mask = (gt > min_depth) & (gt < max_depth)
        if valid.sum() >= 3:
            uv = matches.points1[valid]
            sc = scale_from_gt(pts3d, gt, points1=uv) if scale_from_gt_ok else {"scale": 1.0}
            scale = float(sc["scale"])
            depth_raw = depth_from_triangulation(pts3d, uv, size)
            scale = scale if np.isfinite(scale) else 1.0
            depth_scaled = depth_raw * scale
        else:
            scale = float("nan")
            depth_raw = np.zeros(size, np.float32)
            depth_scaled = depth_raw

        raw = depth_metrics(depth_raw, gt, min_depth=min_depth, max_depth=max_depth)
        scaled = depth_metrics(depth_scaled, gt, min_depth=min_depth, max_depth=max_depth)
        um, uinfo = aligned_depth_metrics(
            depth_scaled,
            gt,
            seq.K,
            mask=gt_mask,
            alignment="umeyama",
            min_depth=min_depth,
            max_depth=max_depth,
        )
        rec.update(
            {
                "ok": True,
                "num_inliers": int(tv.num_inliers),
                "inlier_ratio": float(tv.inlier_ratio),
                "scale_factor": scale,
                "depth_coverage": float((depth_raw > 0).mean()),
                "depth_raw": raw,
                "depth_median_scaled": scaled,
                "depth_umeyama": um,
                "alignment": uinfo,
                "pose": perr,
                "T_est": np.block(
                    [[R_est_full, t_est_full.reshape(3, 1)], [np.zeros((1, 3)), np.ones((1, 1))]]
                ).tolist(),
                "runtime_ms": (time.perf_counter() - t0) * 1000.0,
            }
        )
        records.append(rec)

    ok = [r for r in records if r.get("ok")]
    return {"per_pair": records, "aggregate": _aggregate(ok), "num_ok": len(ok)}


def evaluate_monodepth(
    cfg: dict[str, Any],
    seq: Any,
    pairs: list[tuple[int, int]],
    depth_net: DepthNet,
    pose_net: PoseNet,
    min_depth: float,
    max_depth: float,
) -> dict[str, Any]:
    """Run the self-supervised depth + pose networks over a list of pairs.

    Args:
        cfg: Configuration dict.
        seq: The sequence to evaluate on.
        pairs: ``(i, j)`` frame index pairs.
        depth_net: Trained depth network.
        pose_net: Trained pose network.
        min_depth: Metric lower bound.
        max_depth: Metric upper bound.

    Returns:
        A dict with ``per_pair`` records and ``aggregate`` medians.
    """
    records = []
    for i, j in pairs:
        t0 = time.perf_counter()
        img_i = _tensor_frame(seq.frames[i])
        img_j = _tensor_frame(seq.frames[j])
        gt = seq.depths[i]
        gt_mask = (gt > min_depth) & (gt < max_depth)
        with torch.no_grad():
            _, depth = depth_net(img_i)
            T = pose_net(img_i, img_j)  # target -> source, per the package convention
        d = depth[0, 0].numpy()
        R_est_full = T[0, :3, :3].numpy()
        t_est_full = T[0, :3, 3].numpy()
        tn = np.linalg.norm(t_est_full)
        t_est = t_est_full / tn if tn > 0 else t_est_full
        R_est = R_est_full

        raw = depth_metrics(d, gt, min_depth=min_depth, max_depth=max_depth)
        sc_m, _ = aligned_depth_metrics(
            d, gt, seq.K, mask=gt_mask, alignment="median", min_depth=min_depth, max_depth=max_depth
        )
        scale = (
            float(np.median(gt[gt_mask] / np.clip(d[gt_mask], 1e-6, None)))
            if gt_mask.any()
            else float("nan")
        )
        um, uinfo = aligned_depth_metrics(
            d,
            gt,
            seq.K,
            mask=gt_mask,
            alignment="umeyama",
            min_depth=min_depth,
            max_depth=max_depth,
        )
        R_gt, t_gt = relative_pose(seq.poses, i, j)
        perr = pose_error(R_est, t_est, R_gt, t_gt)
        records.append(
            {
                "ok": True,
                "pair": [int(i), int(j)],
                "scale_factor": scale,
                "depth_coverage": 1.0,
                "depth_raw": raw,
                "depth_median_scaled": sc_m,
                "depth_umeyama": um,
                "alignment": uinfo,
                "pose": perr,
                # full target->source transform, so the per-pair estimates can be
                # composed into a trajectory for ATE/RPE
                "T_est": np.block(
                    [
                        [R_est_full, t_est_full.reshape(3, 1)],
                        [np.zeros((1, 3)), np.ones((1, 1))],
                    ]
                ).tolist(),
                "runtime_ms": (time.perf_counter() - t0) * 1000.0,
            }
        )
    return {"per_pair": records, "aggregate": _aggregate(records), "num_ok": len(records)}


def evaluate_pointmap(
    cfg: dict[str, Any],
    seq: Any,
    pairs: list[tuple[int, int]],
    net: PointmapNet,
    min_depth: float,
    max_depth: float,
) -> dict[str, Any]:
    """Run the pointmap regressor over a list of pairs.

    Depth is the ``z`` channel of view 1's predicted pointmap. Pose is recovered
    from the two predicted pointmaps with RANSAC PnP, and the focal length is
    estimated from the pointmap itself.

    Args:
        cfg: Configuration dict.
        seq: The sequence to evaluate on.
        pairs: ``(i, j)`` frame index pairs.
        net: Trained pointmap network.
        min_depth: Metric lower bound.
        max_depth: Metric upper bound.

    Returns:
        A dict with ``per_pair`` records, ``aggregate`` medians and the focal
        estimates.
    """
    records = []
    focals = []
    for i, j in pairs:
        t0 = time.perf_counter()
        img_i = _tensor_frame(seq.frames[i])
        img_j = _tensor_frame(seq.frames[j])
        with torch.no_grad():
            out = net(img_i, img_j)
        pm1 = out["pts1"][0].numpy()
        pm2 = out["pts2"][0].numpy()
        out["conf1"][0].numpy()
        conf2 = out["conf2"][0].numpy()

        gt = seq.depths[i]
        gt_mask = (gt > min_depth) & (gt < max_depth)
        d_raw = depth_from_pointmap(pm1)

        # the pointmap is only defined up to this pair's own scale
        raw = depth_metrics(d_raw, gt, min_depth=min_depth, max_depth=max_depth)
        sc_m, _ = aligned_depth_metrics(
            d_raw,
            gt,
            seq.K,
            mask=gt_mask,
            alignment="median",
            min_depth=min_depth,
            max_depth=max_depth,
        )
        scale = (
            float(np.median(gt[gt_mask] / np.clip(d_raw[gt_mask], 1e-6, None)))
            if gt_mask.any() and np.all(np.isfinite(d_raw[gt_mask]))
            else float("nan")
        )
        um, uinfo = aligned_depth_metrics(
            d_raw,
            gt,
            seq.K,
            mask=gt_mask,
            alignment="umeyama",
            min_depth=min_depth,
            max_depth=max_depth,
        )

        R_gt, t_gt = relative_pose(seq.poses, i, j)
        try:
            pose = pose_from_pointmaps(pm1, pm2, conf=conf2, K=seq.K, gt_R=R_gt, gt_t=t_gt)
            pose_rec = {
                "rotation_deg": pose.rotation_error_deg,
                "translation_deg": pose.translation_angle_error_deg,
                "inlier_ratio": pose.inlier_ratio,
                "reprojection_median_px": pose.reprojection_median_px,
                "inter_pointmap_scale": pose.scale,
            }
        except ValueError as exc:
            pose_rec = {
                "error": str(exc),
                "rotation_deg": float("nan"),
                "translation_deg": float("nan"),
            }

        f = focal_from_pointmaps(pm1, mask=gt_mask)
        focals.append({"pair": [int(i), int(j)], **{k: float(v) for k, v in f.items()}})

        records.append(
            {
                "ok": True,
                "pair": [int(i), int(j)],
                "scale_factor": scale,
                "depth_coverage": float((d_raw > 0).mean()) if np.isfinite(d_raw).all() else 0.0,
                "depth_raw": raw,
                "depth_median_scaled": sc_m,
                "depth_umeyama": um,
                "alignment": uinfo,
                "pose": pose_rec,
                "T_est": np.block(
                    [[pose.R, pose.t.reshape(3, 1)], [np.zeros((1, 3)), np.ones((1, 1))]]
                ).tolist(),
                "runtime_ms": (time.perf_counter() - t0) * 1000.0,
            }
        )

    agg = _aggregate(records)
    for key in (
        "rotation_deg",
        "translation_deg",
        "inlier_ratio",
        "reprojection_median_px",
        "inter_pointmap_scale",
    ):
        vals = [
            r["pose"][key]
            for r in records
            if isinstance(r.get("pose"), dict) and np.isfinite(r["pose"].get(key, np.nan))
        ]
        agg[f"pose.{key}"] = float(np.median(vals)) if vals else float("nan")
    return {
        "per_pair": records,
        "aggregate": agg,
        "num_ok": len(records),
        "focal_estimates": focals,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def run_compare(args: Any) -> int:
    """``monogeo compare``: run all methods on one held-out sequence.

    Args:
        args: Parsed argparse namespace.

    Returns:
        Process exit code (0 on success).
    """
    cfg = getattr(args, "cfg", {}) or {}
    seed = int(getattr(args, "seed", 0) or 0)
    torch.manual_seed(seed)
    np.random.seed(seed)

    size = (int(cfg_get(cfg, "data.height", 128)), int(cfg_get(cfg, "data.width", 416)))
    root = cfg_get(cfg, "data.data_root", "data")
    name = getattr(args, "sequence", None) or cfg_get(cfg, "data.sequence", "forward")
    num_frames = int(getattr(args, "frames", None) or cfg_get(cfg, "data.num_frames", 120))
    held_out_fraction = float(cfg_get(cfg, "eval.held_out_fraction", 0.7))
    min_depth = float(cfg_get(cfg, "eval.min_depth", 1.0))
    max_depth = float(cfg_get(cfg, "eval.max_depth", 120.0))
    gap = 1

    full = load_sequence(
        name,
        size=size,
        num_frames=num_frames,
        seed=seed,
        root=root,
        cache=bool(cfg_get(cfg, "data.cache", True)),
        fov_deg=float(cfg_get(cfg, "synth.fov_deg", 62.0)),
        distortion=cfg_get(cfg, "synth.distortion", None),
        verbose=bool(getattr(args, "verbose", False)),
    )
    _, held = splits(full, held_out_fraction)
    seq = slice_sequence(full, held[0], held[-1] + 1)
    pairs = [(i, i + gap) for i in range(0, len(seq) - gap)]
    print(
        f"[compare] held-out frames {held[0]}..{held[-1]} of {full.name} ({seq.source}) "
        f"{seq.width}x{seq.height}, {len(pairs)} pairs, gap={gap}"
    )
    print(f"[compare] evaluation slice {seq.name}, the training split is frames 1..{held[0]-1}")

    depth_net = load_depth_model(cfg)
    pose_net = load_pose_model(cfg)
    pm_net, provenance = load_pointmap_model(cfg)
    print(f"[compare] pointmap provenance: {provenance}")

    results: dict[str, Any] = {}
    results["classical_two_view"] = evaluate_classical(cfg, seq, pairs, min_depth, max_depth)
    results["self_supervised_monodepth"] = evaluate_monodepth(
        cfg, seq, pairs, depth_net, pose_net, min_depth, max_depth
    )
    results["pointmap"] = evaluate_pointmap(cfg, seq, pairs, pm_net, min_depth, max_depth)
    # an explicit conditioning ablation for the classical baseline: the same
    # pipeline with a doubled baseline, to separate "method is weak" from
    # "short-baseline two-view geometry is weak"
    pairs2 = [(i, i + 2) for i in range(0, len(seq) - 2)]
    results["classical_two_view_gap2"] = evaluate_classical(cfg, seq, pairs2, min_depth, max_depth)

    K_true = np.asarray(seq.K, np.float64)
    focal_error = {}
    for method, res in results.items():
        if method == "pointmap":
            fx = [f["fx"] for f in res["focal_estimates"] if np.isfinite(f["fx"])]
            est = float(np.median(fx)) if fx else float("nan")
            focal_error[method] = {
                "focal_estimated_from_pointmap": est,
                "focal_used_by_pipeline": float(K_true[0, 0]),
                "focal_true": float(K_true[0, 0]),
                "abs_error_px": abs(est - K_true[0, 0]) if np.isfinite(est) else float("nan"),
                "rel_error_pct": (
                    100.0 * abs(est - K_true[0, 0]) / K_true[0, 0]
                    if np.isfinite(est)
                    else float("nan")
                ),
                "source": "estimated from the predicted pointmap (focal_from_pointmaps)",
            }
        else:
            focal_error[method] = {
                "focal_estimated_from_pointmap": float("nan"),
                "focal_used_by_pipeline": float(K_true[0, 0]),
                "focal_true": float(K_true[0, 0]),
                "abs_error_px": float("nan"),
                "rel_error_pct": float("nan"),
                "source": "no focal estimate: the calibrated K is an input to this method",
            }

    scale_analysis = {
        "note": (
            "Monocular geometry is recovered only up to scale. `scale_factor` is the per-frame "
            "median ratio gt/pred; `drift` is the std/mean of that factor across frames. "
            "`with_umeyama` additionally allows a 7-DoF similarity, which a depth-only method "
            "cannot exploit but a pointmap method can."
        ),
        "per_method": {},
    }
    for method, res in results.items():
        agg = res["aggregate"]
        scale_analysis["per_method"][method] = {
            "scale_factor_median": agg.get("alignment.scale", float("nan")),
            "scale_factor_from_median_ratio": agg.get("scale_factor", float("nan")),
            "scale_drift_std_over_mean": agg.get("scale_drift_std_over_mean", float("nan")),
            "scale_drift_std": agg.get("scale_drift_std", float("nan")),
            "metrics_no_alignment": {
                "abs_rel": agg.get("depth_raw.abs_rel", float("nan")),
                "rmse": agg.get("depth_raw.rmse", float("nan")),
                "delta1": agg.get("depth_raw.delta1", float("nan")),
            },
            "metrics_with_median_scaling": {
                "abs_rel": agg.get("depth_median_scaled.abs_rel", float("nan")),
                "rmse": agg.get("depth_median_scaled.rmse", float("nan")),
                "delta1": agg.get("depth_median_scaled.delta1", float("nan")),
            },
            "metrics_with_umeyama_alignment": {
                "abs_rel": agg.get("depth_umeyama.abs_rel", float("nan")),
                "rmse": agg.get("depth_umeyama.rmse", float("nan")),
                "delta1": agg.get("depth_umeyama.delta1", float("nan")),
                "rotation_deg": agg.get("alignment.rotation_deg", float("nan")),
                "translation": agg.get("alignment.translation", float("nan")),
            },
            "runtime_ms_per_frame": agg.get("runtime_ms", float("nan")),
        }

    payload = {
        "step": "compare",
        "sequence": seq.name,
        "data_source": seq.source,
        "resolution_hw": [seq.height, seq.width],
        "num_frames": len(seq),
        "frame_gap": gap,
        "num_pairs": len(pairs),
        "seed": seed,
        "metric_range_m": [min_depth, max_depth],
        "ground_truth_used_for": {
            "classical_two_view": "a single median depth ratio, to fix the unobservable monocular scale",
            "self_supervised_monodepth": "nothing at inference; used for evaluation and for optional median scaling",
            "pointmap": "nothing at inference; used for evaluation and for optional median scaling",
        },
        "pointmap_provenance": provenance,
        "focal_analysis": focal_error,
        "scale_analysis": scale_analysis,
        "methods": {
            m: {"aggregate": r["aggregate"], "num_ok": r["num_ok"], "num_pairs": len(r["per_pair"])}
            for m, r in results.items()
        },
        "per_pair": {m: r["per_pair"] for m, r in results.items()},
    }
    out = save_json(getattr(args, "out", "results/method_comparison.json"), payload)

    print("\n[compare] per-method medians over held-out pairs")
    header = f"{'method':30s} {'AbsRel':>8s} {'AbsRel*':>8s} {'d1':>6s} {'d1*':>6s} {'rot':>7s} {'trans':>7s} {'ms/frame':>9s}"
    print("[compare] " + header)
    for m, r in results.items():
        a = r["aggregate"]
        print(
            f"[compare] {m:30s} {a.get('depth_raw.abs_rel', float('nan')):8.3f} "
            f"{a.get('depth_median_scaled.abs_rel', float('nan')):8.3f} "
            f"{a.get('depth_raw.delta1', float('nan')):6.3f} {a.get('depth_median_scaled.delta1', float('nan')):6.3f} "
            f"{a.get('pose.rotation_deg', float('nan')):7.3f} {a.get('pose.translation_deg', float('nan')):7.3f} "
            f"{a.get('runtime_ms', float('nan')):9.1f}"
        )
    print("[compare] AbsRel/d1 = no alignment; AbsRel*/d1* = after median scaling")
    for m, r in scale_analysis["per_method"].items():
        print(
            f"[compare] scale[{m}] factor {r['scale_factor_from_median_ratio']:.3f} "
            f"drift(std/mean) {r['scale_drift_std_over_mean']:.3f}"
        )
    print(f"[compare] wrote {out}")
    return 0
