"""Depth/pose evaluation and robustness stress tests.

``monogeo evaluate`` scores all three methods on the held-out tail of the flight
log: depth metrics before and after alignment, per-pair pose errors, and
absolute/relative trajectory error over the *composed* trajectory each method
produces.

``monogeo stress`` re-runs the same methods on corrupted frames (motion blur,
low texture, low light, JPEG) at three severities, and on two domain-shifted
sequences (a different viewing geometry and an altitude change), and reports the
degradation of each metric **relative to the clean evaluation**. That last part
is the point: a raw number for a corrupted frame tells you nothing without the
clean reference beside it.
"""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import torch

from .cli import cfg_get, save_json
from .compare import (
    evaluate_classical,
    evaluate_monodepth,
    evaluate_pointmap,
    load_depth_model,
    load_pointmap_model,
    load_pose_model,
)
from .corruptions import available_corruptions, corrupt_frames
from .data import Sequence, load_sequence, slice_sequence, splits
from .metrics import ate, compose_relative_poses, rpe

__all__ = ["evaluate_methods", "run_evaluate", "run_stress", "trajectory_from_records"]

#: The three geometry families compared throughout the repository.
METHOD_NAMES = ("classical_two_view", "self_supervised_monodepth", "pointmap")

#: Headline keys where a *higher* value is better, so that `_degradation` can
#: keep the sign convention "positive always means worse".
HIGHER_IS_BETTER = ("delta1_median_scaled", "delta1", "delta1_umeyama")


def _models(cfg: dict[str, Any]) -> dict[str, Any]:
    """Load the trained models once so every evaluation shares them."""
    return {
        "depth_net": load_depth_model(cfg),
        "pose_net": load_pose_model(cfg),
        "pointmap_net": load_pointmap_model(cfg)[0],
    }


def evaluate_methods(
    cfg: dict[str, Any],
    seq: Sequence,
    models: dict[str, Any],
    min_depth: float,
    max_depth: float,
    methods: tuple[str, ...] = METHOD_NAMES,
    gap: int = 1,
    max_pairs: int | None = None,
) -> dict[str, Any]:
    """Run the requested methods on a sequence and aggregate the metrics.

    Args:
        cfg: Configuration dict.
        seq: Sequence to evaluate on.
        models: Dict from :func:`_models`.
        min_depth: Metric lower bound.
        max_depth: Metric upper bound.
        methods: Which methods to run.
        gap: Frame gap between pairs.
        max_pairs: Optional cap on the number of pairs, sub-sampled uniformly.
            Used by the stress tests, which re-run every method dozens of times.

    Returns:
        ``{method: {"aggregate": ..., "per_pair": [...], "trajectory": ...}}``.
    """
    all_pairs = [(i, i + gap) for i in range(0, len(seq) - gap)]
    if max_pairs is not None and len(all_pairs) > int(max_pairs):
        idx = np.unique(np.linspace(0, len(all_pairs) - 1, int(max_pairs)).astype(int))
        pairs = [all_pairs[k] for k in idx]
    else:
        pairs = all_pairs
    out: dict[str, Any] = {}
    if "classical_two_view" in methods:
        res = evaluate_classical(cfg, seq, pairs, min_depth, max_depth)
        out["classical_two_view"] = res
    if "self_supervised_monodepth" in methods:
        out["self_supervised_monodepth"] = evaluate_monodepth(
            cfg, seq, pairs, models["depth_net"], models["pose_net"], min_depth, max_depth
        )
    if "pointmap" in methods:
        out["pointmap"] = evaluate_pointmap(
            cfg, seq, pairs, models["pointmap_net"], min_depth, max_depth
        )
    for _name, res in out.items():
        res["trajectory"] = trajectory_from_records(res["per_pair"])
    return out


def trajectory_from_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compose per-pair estimated transforms into a trajectory.

    Pairs are assumed to be ``(i, i+gap)`` in increasing ``i``, so chaining the
    per-pair transforms yields the camera path up to an arbitrary global scale
    and reference frame -- which is all a monocular method can observe.

    Args:
        records: Per-pair records containing a ``T_est`` entry.

    Returns:
        Dict with ``poses`` ``(T, 4, 4)`` (the composed path) or ``{}`` when no
        record carries a transform.
    """
    transforms = []
    for r in records:
        if r.get("ok") and isinstance(r.get("T_est"), list):
            transforms.append(np.asarray(r["T_est"], np.float64))
    if len(transforms) < 2:
        return {}
    return {"poses": compose_relative_poses(np.stack(transforms, 0)).tolist()}


def _trajectory_metrics(method_traj: dict[str, Any], gt_poses: np.ndarray) -> dict[str, Any]:
    """ATE and RPE of a composed trajectory against ground truth.

    Both methods and ground truth are rebased so their first pose is the
    identity: a monocular trajectory is only defined up to a global similarity,
    and Umeyama alignment inside :func:`monogeo.metrics.ate` removes the rest.
    """
    if not method_traj or "poses" not in method_traj:
        return {"ate": {"rmse": float("nan")}, "rpe": {"translation_rmse": float("nan")}}
    est = np.asarray(method_traj["poses"], np.float64)
    gt = np.asarray(gt_poses, np.float64)
    # compose_relative_poses returns T+1 poses for T transforms; align lengths
    n = min(len(est), len(gt))
    est = est[:n]
    gt_rebased = gt[:n] @ np.linalg.inv(gt[0])
    try:
        ate_res = ate(est, gt_rebased, align=True)
    except ValueError:
        ate_res = {
            "rmse": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
            "max": float("nan"),
            "scale": float("nan"),
            "n": 0,
        }
    try:
        rpe_res = rpe(est, gt_rebased, delta=1, align=True)
    except ValueError:
        rpe_res = {"translation_rmse": float("nan"), "rotation_rmse_deg": float("nan"), "n": 0}
    return {"ate": ate_res, "rpe": rpe_res}


def _degradation(clean: float, corrupted: float, higher_is_better: bool = False) -> float:
    """Percentage change of a metric, signed so that positive means *worse*."""
    if not np.isfinite(clean) or not np.isfinite(corrupted) or abs(clean) < 1e-12:
        return float("nan")
    delta = (corrupted - clean) if not higher_is_better else (clean - corrupted)
    return float(100.0 * delta / abs(clean))


def _headline(agg: dict[str, Any]) -> dict[str, float]:
    """Pick the metrics quoted in the results report."""
    return {
        "abs_rel": agg.get("depth_raw.abs_rel", float("nan")),
        "abs_rel_median_scaled": agg.get("depth_median_scaled.abs_rel", float("nan")),
        "abs_rel_umeyama": agg.get("depth_umeyama.abs_rel", float("nan")),
        "sq_rel_median_scaled": agg.get("depth_median_scaled.sq_rel", float("nan")),
        "rmse_median_scaled": agg.get("depth_median_scaled.rmse", float("nan")),
        "rmse_log_median_scaled": agg.get("depth_median_scaled.rmse_log", float("nan")),
        "delta1": agg.get("depth_raw.delta1", float("nan")),
        "delta1_median_scaled": agg.get("depth_median_scaled.delta1", float("nan")),
        "delta2_median_scaled": agg.get("depth_median_scaled.delta2", float("nan")),
        "delta3_median_scaled": agg.get("depth_median_scaled.delta3", float("nan")),
        "rotation_deg": agg.get("pose.rotation_deg", float("nan")),
        "translation_deg": agg.get("pose.translation_deg", float("nan")),
        "scale_factor": agg.get("scale_factor", float("nan")),
        "scale_drift_std_over_mean": agg.get("scale_drift_std_over_mean", float("nan")),
        "runtime_ms_per_frame": agg.get("runtime_ms", float("nan")),
        "depth_coverage": agg.get("depth_coverage", float("nan")),
    }


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #
def run_evaluate(args: Any) -> int:
    """``monogeo evaluate``: clean held-out depth and pose metrics.

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
    frac = float(cfg_get(cfg, "eval.held_out_fraction", 0.7))
    min_depth = float(cfg_get(cfg, "eval.min_depth", 1.0))
    max_depth = float(cfg_get(cfg, "eval.max_depth", 120.0))

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
    train_idx, held = splits(full, frac)
    seq = slice_sequence(full, held[0], held[-1] + 1)
    models = _models(cfg)
    print(
        f"[evaluate] held-out {seq.name} ({seq.source}) {seq.width}x{seq.height}, {len(seq)} frames"
    )

    t0 = time.perf_counter()
    results = evaluate_methods(
        cfg,
        seq,
        models,
        min_depth,
        max_depth,
        max_pairs=int(cfg_get(cfg, "eval.max_pairs", 24)),
    )

    gt_poses = np.asarray(seq.poses, np.float64)
    per_method: dict[str, Any] = {}
    for method, res in results.items():
        agg = res["aggregate"]
        head = _headline(agg)
        head.update(_trajectory_metrics(res["trajectory"], gt_poses))
        per_method[method] = {
            "depth_and_pair_pose": head,
            "aggregate_full": agg,
            "num_pairs": res["num_ok"],
        }

    K_true = np.asarray(seq.K, np.float64)
    payload = {
        "step": "evaluate",
        "sequence": seq.name,
        "parent_sequence": full.name,
        "data_source": seq.source,
        "resolution_hw": [seq.height, seq.width],
        "num_frames": len(seq),
        "held_out_frames": [int(held[0]), int(held[-1])],
        "train_frames": [int(train_idx[0]), int(train_idx[-1])],
        "seed": seed,
        "metric_range_m": [min_depth, max_depth],
        "frame_gap": 1,
        "intrinsics_true": {
            "fx": float(K_true[0, 0]),
            "fy": float(K_true[1, 1]),
            "cx": float(K_true[0, 2]),
            "cy": float(K_true[1, 2]),
        },
        "methods": per_method,
        "ground_truth_trajectory": [[float(v) for row in p for v in row] for p in gt_poses],
        "estimated_trajectories": {
            m: res["trajectory"].get("poses", []) for m, res in results.items()
        },
        "runtime_seconds": time.perf_counter() - t0,
    }
    out = save_json(getattr(args, "out", "results/metrics.json"), payload)

    print("\n[evaluate] depth (median over held-out pairs)")
    print(
        f"[evaluate] {'method':28s} {'AbsRel':>7s} {'AbsRel*':>8s} {'d1*':>6s} {'RMSE*':>7s} {'ATE(m)':>7s} {'ms':>7s}"
    )
    for method, m in per_method.items():
        h = m["depth_and_pair_pose"]
        print(
            f"[evaluate] {method:28s} {h['abs_rel']:7.3f} {h['abs_rel_median_scaled']:8.3f} "
            f"{h['delta1_median_scaled']:6.3f} {h['rmse_median_scaled']:7.2f} "
            f"{h['ate']['rmse']:7.2f} {h['runtime_ms_per_frame']:7.1f}"
        )
    print("[evaluate] * = after median scaling")
    print(f"[evaluate] wrote {out}")
    return 0


# --------------------------------------------------------------------------- #
# stress
# --------------------------------------------------------------------------- #
def run_stress(args: Any) -> int:
    """``monogeo stress``: corruptions plus domain shift, with degradations.

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
    frac = float(cfg_get(cfg, "eval.held_out_fraction", 0.7))
    min_depth = float(cfg_get(cfg, "eval.min_depth", 1.0))
    max_depth = float(cfg_get(cfg, "eval.max_depth", 120.0))
    corruptions = list(cfg_get(cfg, "stress.corruptions", available_corruptions()))
    severities = [int(s) for s in cfg_get(cfg, "stress.severities", [1, 2, 3])]

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
    _, held = splits(full, frac)
    seq = slice_sequence(full, held[0], held[-1] + 1)
    models = _models(cfg)
    print(f"[stress] base sequence {seq.name} ({seq.source}), {len(seq)} frames")
    print(f"[stress] corruptions {corruptions} at severities {severities}")

    t0 = time.perf_counter()
    max_pairs = int(cfg_get(cfg, "stress.max_pairs", 12))
    clean = evaluate_methods(cfg, seq, models, min_depth, max_depth, max_pairs=max_pairs)
    clean_head = {m: _headline(r["aggregate"]) for m, r in clean.items()}
    gt_poses = np.asarray(seq.poses, np.float64)
    for m, res in clean.items():
        clean_head[m].update(_trajectory_metrics(res["trajectory"], gt_poses))

    def _run_variant(variant: Sequence, tag: str, variant_gt: np.ndarray) -> dict[str, Any]:
        """Score one (possibly corrupted, possibly different) sequence.

        ``variant_gt`` must be the ground-truth poses *of that variant*: a
        domain-shifted sequence has its own trajectory, and scoring it against
        the base sequence's poses would silently produce nonsense.
        """
        variant = copy.copy(variant)
        variant.name = tag
        res = evaluate_methods(cfg, variant, models, min_depth, max_depth, max_pairs=max_pairs)
        out: dict[str, Any] = {}
        for method, r in res.items():
            n_ok = int(r["num_ok"])
            n_tried = len(r["per_pair"])
            head = _headline(r["aggregate"])
            head.update(_trajectory_metrics(r["trajectory"], variant_gt))
            deg = {
                key: _degradation(
                    clean_head[method][key], head[key], higher_is_better=key in HIGHER_IS_BETTER
                )
                for key in (
                    "abs_rel_median_scaled",
                    "delta1_median_scaled",
                    "rmse_median_scaled",
                    "rotation_deg",
                    "translation_deg",
                )
            }
            out[method] = {
                "metrics": head,
                "degradation_pct": deg,
                # how many pairs the method could actually be scored on: the
                # classical pipeline drops pairs where matching fails, so a
                # "missing" number is not the same as a "bad" one
                "num_pairs_ok": n_ok,
                "num_pairs_tried": n_tried,
                "all_pairs_missing": n_ok == 0,
            }
        return out

    corrupted: dict[str, Any] = {}
    for cname in corruptions:
        for sev in severities:
            tag = f"{cname}_sev{sev}"
            print(f"[stress] {tag} ...")
            frames = corrupt_frames(seq.frames, cname, sev, seed=seed)
            corrupted_seq = copy.copy(seq)
            corrupted_seq.frames = frames
            corrupted[tag] = {
                "corruption": cname,
                "severity": sev,
                "mean_std": float(np.mean([f.std() for f in frames])),
                # same trajectory and ground truth, corrupted pixels only
                "methods": _run_variant(corrupted_seq, tag, gt_poses),
            }

    domain_shift: dict[str, Any] = {}
    for label, seq_name in (
        ("altitude_descent", cfg_get(cfg, "data.shift_sequence", "descent")),
        ("viewpoint_orbit", cfg_get(cfg, "data.orbit_sequence", "orbit")),
    ):
        print(f"[stress] domain shift: {label} ({seq_name}) ...")
        other = load_sequence(
            seq_name,
            size=size,
            num_frames=min(num_frames, 60),
            seed=seed,
            root=root,
            cache=bool(cfg_get(cfg, "data.cache", True)),
            fov_deg=float(cfg_get(cfg, "synth.fov_deg", 62.0)),
            distortion=cfg_get(cfg, "synth.distortion", None),
            verbose=False,
        )
        domain_shift[label] = {
            "sequence": other.name,
            "num_frames": len(other),
            "methods": _run_variant(other, other.name, np.asarray(other.poses, np.float64)),
        }
        # the domain-shift variants are compared against the clean number, so
        # they show how much accuracy the shift costs even with no corruption

    all_methods = list(clean.keys())

    def _mean_deg(block: dict[str, Any], method: str, key: str) -> float:
        try:
            return float(block["methods"][method]["degradation_pct"][key])
        except KeyError:
            return float("nan")

    def _pairs_ok(block: dict[str, Any], method: str) -> int:
        return int(block["methods"].get(method, {}).get("num_pairs_ok", 0))

    summary = {}
    for method in all_methods:
        per_metric = {}
        for metric in ("abs_rel_median_scaled", "delta1_median_scaled", "rotation_deg"):
            vals = [_mean_deg(v, method, metric) for v in corrupted.values()]
            vals = [v for v in vals if np.isfinite(v)]
            per_metric[metric] = {
                "mean_degradation_pct_over_corruptions": (
                    float(np.mean(vals)) if vals else float("nan")
                ),
                "max_degradation_pct_over_corruptions": (
                    float(np.max(vals)) if vals else float("nan")
                ),
            }
        shift = {}
        for label, block in domain_shift.items():
            shift[label] = {
                "abs_rel_median_scaled_degradation_pct": _mean_deg(
                    block, method, "abs_rel_median_scaled"
                ),
                "delta1_median_scaled_degradation_pct": _mean_deg(
                    block, method, "delta1_median_scaled"
                ),
                "rotation_deg_degradation_pct": _mean_deg(block, method, "rotation_deg"),
                "abs_rel_median_scaled": block["methods"]
                .get(method, {})
                .get("metrics", {})
                .get("abs_rel_median_scaled", float("nan")),
            }
        summary[method] = {"corruptions": per_metric, "domain_shift": shift}

    payload = {
        "step": "stress",
        "sequence": seq.name,
        "parent_sequence": full.name,
        "data_source": seq.source,
        "resolution_hw": [seq.height, seq.width],
        "num_frames": len(seq),
        "held_out_frames": [int(held[0]), int(held[-1])],
        "seed": seed,
        "metric_range_m": [min_depth, max_depth],
        "note": (
            "degradation_pct is signed so that positive always means WORSE, for "
            "both error metrics and accuracy metrics (delta). The clean reference "
            "is the uncorrupted held-out evaluation in this same run."
        ),
        "clean": {m: clean_head[m] for m in all_methods},
        "corruptions_tested": corruptions,
        "severities_tested": severities,
        "corrupted": corrupted,
        "domain_shift": domain_shift,
        "summary": summary,
        "pairs_ok": {
            tag: {m: _pairs_ok(block, m) for m in block["methods"]}
            for tag, block in corrupted.items()
        },
        "domain_shift_pairs_ok": {
            label: {m: _pairs_ok(block, m) for m in block["methods"]}
            for label, block in domain_shift.items()
        },
        "runtime_seconds": time.perf_counter() - t0,
    }
    out = save_json(getattr(args, "out", "results/stress_tests.json"), payload)

    print("\n[stress] degradation vs clean (median-scaled AbsRel, %)")
    print(f"[stress] {'variant':22s} " + " ".join(f"{m[:14]:>15s}" for m in all_methods))
    for tag, block in corrupted.items():
        row = " ".join(f"{_mean_deg(block, m, 'abs_rel_median_scaled'):15.1f}" for m in all_methods)
        print(f"[stress] {tag:22s} {row}")
    for label, block in domain_shift.items():
        row = " ".join(f"{_mean_deg(block, m, 'abs_rel_median_scaled'):15.1f}" for m in all_methods)
        print(f"[stress] {label:22s} {row}")
    print(f"[stress] wrote {out}")
    return 0
