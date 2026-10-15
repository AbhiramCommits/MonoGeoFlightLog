"""Standard monocular depth and pose metrics.

The depth metrics are the ones used by the KITTI / NYU depth benchmarks
(Eigen et al.): absolute relative error, squared relative error, RMSE,
RMSE of the log, and the three accuracy thresholds ``delta < 1.25^k``.

Two things are reported that a naive implementation usually leaves implicit:

* **the mask**. A depth map predicts a value for every pixel, but the metric is
  only meaningful where ground truth exists and lies inside a plausible range.
  Every returned dict therefore carries ``n`` and the mask ratio, so a method
  cannot look better simply by predicting fewer pixels.
* **the alignment**. Monocular depth (and a pointmap) is only defined up to a
  scale, so metrics are reported for ``none``, ``median`` and ``umeyama``
  alignment, and the scale factor used is returned alongside them.

Absolute trajectory error (ATE) and relative pose error (RPE) are provided for
composed trajectories and are aligned with Umeyama before comparison, so a
method is never penalised for a global scale it cannot observe.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .pointmap_ops import umeyama_alignment

__all__ = [
    "align_trajectory",
    "aligned_depth_metrics",
    "ate",
    "compose_relative_poses",
    "depth_metrics",
    "rpe",
]

_EPS = 1e-9


def depth_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray | None = None,
    min_depth: float = 1e-3,
    max_depth: float = np.inf,
) -> dict[str, Any]:
    """Standard depth metrics over a masked, range-limited pixel set.

    Args:
        pred: Predicted depth (any shape), in the same units as ``gt``.
        gt: Ground-truth depth, broadcast against ``pred``.
        mask: Optional boolean mask of additional valid pixels.
        min_depth: Ground truth must exceed this to be counted.
        max_depth: Ground truth must be below this to be counted.

    Returns:
        Dict with ``abs_rel``, ``sq_rel``, ``rmse``, ``rmse_log``, ``delta1``,
        ``delta2``, ``delta3``, ``n`` and ``mask_ratio``. All metric values are
        ``nan`` when no pixel survives the mask.
    """
    p = np.asarray(pred, np.float64)
    g = np.asarray(gt, np.float64)
    p, g = np.broadcast_arrays(p, g)
    valid = (
        np.isfinite(p)
        & np.isfinite(g)
        & (p > _EPS)
        & (g > float(min_depth))
        & (g < float(max_depth))
    )
    if mask is not None:
        valid &= np.asarray(mask, bool).reshape(g.shape)

    n_total = int(g.size)
    n = int(valid.sum())
    out: dict[str, Any] = {
        "n": n,
        "mask_ratio": float(n) / float(n_total) if n_total else 0.0,
        "abs_rel": float("nan"),
        "sq_rel": float("nan"),
        "rmse": float("nan"),
        "rmse_log": float("nan"),
        "delta1": float("nan"),
        "delta2": float("nan"),
        "delta3": float("nan"),
    }
    if n == 0:
        return out

    pp = p[valid]
    gg = g[valid]
    diff = pp - gg
    ratio = np.maximum(pp / gg, gg / pp)
    out.update(
        {
            "abs_rel": float(np.mean(np.abs(diff) / gg)),
            "sq_rel": float(np.mean(diff**2 / gg)),
            "rmse": float(np.sqrt(np.mean(diff**2))),
            "rmse_log": float(np.sqrt(np.mean((np.log(pp) - np.log(gg)) ** 2))),
            "delta1": float(np.mean(ratio < 1.25)),
            "delta2": float(np.mean(ratio < 1.25**2)),
            "delta3": float(np.mean(ratio < 1.25**3)),
        }
    )
    return out


def aligned_depth_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray | None = None,
    alignment: str = "none",
    min_depth: float = 1e-3,
    max_depth: float = np.inf,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Depth metrics after none / median / full-similarity (Umeyama) alignment.

    ``median`` and ``umeyama`` both resolve the scale ambiguity, but ``umeyama``
    additionally allows a rotation and translation to absorb any residual pose
    error -- which is exactly what a pointmap method can and a depth-only method
    cannot exploit. Reporting both makes the "what does alignment cost?" question
    answerable.

    Args:
        pred: Predicted depth.
        gt: Ground-truth depth.
        K: ``(3, 3)`` intrinsics, needed to form point clouds for Umeyama.
        mask: Optional validity mask.
        alignment: ``"none"``, ``"median"`` or ``"umeyama"``.
        min_depth: Ground-truth lower bound.
        max_depth: Ground-truth upper bound.

    Returns:
        ``(metrics, info)`` where ``info`` holds the scale, rotation (deg) and
        translation magnitude applied by the alignment.

    Raises:
        ValueError: If ``alignment`` is unknown.
    """
    from .pointmap_ops import pointmap_from_depth

    mode = str(alignment).lower()
    if mode not in ("none", "median", "umeyama"):
        raise ValueError(f"unknown alignment {alignment!r}; expected none|median|umeyama")

    p = np.asarray(pred, np.float64)
    g = np.asarray(gt, np.float64)
    valid = (
        np.isfinite(p)
        & np.isfinite(g)
        & (p > _EPS)
        & (g > float(min_depth))
        & (g < float(max_depth))
    )
    if mask is not None:
        valid &= np.asarray(mask, bool)
    info = {"scale": 1.0, "rotation_deg": 0.0, "translation": 0.0, "n": int(valid.sum())}

    if mode == "none" or not valid.any():
        return depth_metrics(p, g, mask=valid, min_depth=min_depth, max_depth=max_depth), info

    if mode == "median":
        scale = float(np.median(g[valid] / p[valid]))
        info["scale"] = scale
        return (
            depth_metrics(p * scale, g, mask=valid, min_depth=min_depth, max_depth=max_depth),
            info,
        )

    # full similarity alignment on the point clouds
    pm_pred = pointmap_from_depth(p, np.asarray(K, np.float64), valid)
    pm_gt = pointmap_from_depth(g, np.asarray(K, np.float64), valid)
    X = pm_pred[valid]
    Y = pm_gt[valid]
    if X.shape[0] < 3:
        return depth_metrics(p, g, mask=valid, min_depth=min_depth, max_depth=max_depth), info
    s, R, t = umeyama_alignment(X, Y)
    aligned = s * (X @ R.T) + t
    pred_aligned = np.full_like(np.asarray(p, np.float64), np.nan)
    pred_aligned[valid] = aligned[:, 2]
    info.update(
        {
            "scale": float(s),
            "rotation_deg": float(
                np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
            ),
            "translation": float(np.linalg.norm(t)),
        }
    )
    return (
        depth_metrics(pred_aligned, g, mask=valid, min_depth=min_depth, max_depth=max_depth),
        info,
    )


# --------------------------------------------------------------------------- #
# trajectory metrics
# --------------------------------------------------------------------------- #
def compose_relative_poses(relatives: np.ndarray, origin: np.ndarray | None = None) -> np.ndarray:
    """Integrate a sequence of relative poses into absolute poses.

    Args:
        relatives: ``(T, 4, 4)`` transforms where entry ``i`` maps frame ``i`` to
            frame ``i+1``.
        origin: Optional ``(4, 4)`` world<-first-camera pose. Defaults to the
            identity.

    Returns:
        ``(T + 1, 4, 4)`` absolute world->camera poses.
    """
    rel = np.asarray(relatives, np.float64)
    if rel.ndim != 3 or rel.shape[1:] != (4, 4):
        raise ValueError(f"relatives must be (T, 4, 4), got {rel.shape}")
    poses = np.zeros((len(rel) + 1, 4, 4), np.float64)
    poses[0] = np.eye(4) if origin is None else np.asarray(origin, np.float64)
    for i in range(len(rel)):
        poses[i + 1] = rel[i] @ poses[i]
    return poses


def align_trajectory(
    est: np.ndarray, gt: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Umeyama-align estimated camera centres onto ground truth.

    Args:
        est: ``(T, 4, 4)`` estimated world->camera poses.
        gt: ``(T, 4, 4)`` ground-truth world->camera poses, same length.

    Returns:
        ``(est_aligned, scale, R, t)`` where ``est_aligned`` has the same shape
        as ``est`` with its camera centres moved by ``s R x + t``.

    Raises:
        ValueError: If the shapes differ or there are fewer than three poses.
    """
    est = np.asarray(est, np.float64)
    gt = np.asarray(gt, np.float64)
    if est.shape != gt.shape:
        raise ValueError(f"pose arrays must match, got {est.shape} and {gt.shape}")
    if len(est) < 3:
        raise ValueError(f"need at least 3 poses, got {len(est)}")
    c_est = np.stack([np.linalg.inv(p)[:3, 3] for p in est])
    c_gt = np.stack([np.linalg.inv(p)[:3, 3] for p in gt])
    s, R, t = umeyama_alignment(c_est, c_gt)
    out = est.copy()
    for i in range(len(out)):
        out[i, :3, 3] = s * (R @ est[i, :3, 3]) + t
        out[i, :3, :3] = R @ est[i, :3, :3]
    return out, float(s), R, t


def ate(est: np.ndarray, gt: np.ndarray, align: bool = True) -> dict[str, float]:
    """Absolute trajectory error over camera centres, in metres.

    Args:
        est: ``(T, 4, 4)`` estimated world->camera poses.
        gt: ``(T, 4, 4)`` ground-truth poses.
        align: Umeyama-align the estimate first (recommended: a monocular
            trajectory has no observable scale or global frame).

    Returns:
        Dict with ``rmse``, ``mean``, ``median``, ``max`` (metres), ``scale`` and
        ``n``.

    Raises:
        ValueError: If shapes differ.
    """
    est_a = align_trajectory(est, gt)[0] if align else np.asarray(est, np.float64)
    c_est = np.stack([np.linalg.inv(p)[:3, 3] for p in est_a])
    c_gt = np.stack([np.linalg.inv(p)[:3, 3] for p in gt])
    err = np.linalg.norm(c_est - c_gt, axis=1)
    scale = align_trajectory(est, gt)[1] if align else 1.0
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mean": float(err.mean()),
        "median": float(np.median(err)),
        "max": float(err.max()),
        "scale": float(scale),
        "n": int(len(err)),
    }


def rpe(est: np.ndarray, gt: np.ndarray, delta: int = 1, align: bool = True) -> dict[str, float]:
    """Relative pose error between frames ``i`` and ``i + delta``.

    Args:
        est: ``(T, 4, 4)`` estimated world->camera poses.
        gt: ``(T, 4, 4)`` ground-truth poses.
        delta: Frame gap for the relative measurement.
        align: Umeyama-align the estimate first.

    Returns:
        Dict with ``translation_rmse`` and ``translation_mean`` (metres per
        frame), ``rotation_rmse_deg`` and ``rotation_mean_deg`` (degrees per
        frame), ``delta`` and ``n``.

    Raises:
        ValueError: If shapes differ or the trajectories are shorter than
            ``delta + 1``.
    """
    est = np.asarray(est, np.float64)
    gt = np.asarray(gt, np.float64)
    if est.shape != gt.shape:
        raise ValueError(f"pose arrays must match, got {est.shape} and {gt.shape}")
    if len(est) <= delta:
        raise ValueError(f"need more than delta={delta} poses, got {len(est)}")
    if align:
        est = align_trajectory(est, gt)[0]

    t_err = []
    r_err = []
    for i in range(len(est) - delta):
        rel_e = np.linalg.inv(est[i + delta]) @ est[i]
        rel_g = np.linalg.inv(gt[i + delta]) @ gt[i]
        d = np.linalg.inv(rel_g) @ rel_e
        t_err.append(float(np.linalg.norm(d[:3, 3])))
        cos = (np.trace(d[:3, :3]) - 1.0) / 2.0
        r_err.append(float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))))
    t_arr = np.asarray(t_err)
    r_arr = np.asarray(r_err)
    return {
        "translation_rmse": float(np.sqrt(np.mean(t_arr**2))),
        "translation_mean": float(t_arr.mean()),
        "rotation_rmse_deg": float(np.sqrt(np.mean(r_arr**2))),
        "rotation_mean_deg": float(r_arr.mean()),
        "delta": int(delta),
        "n": int(len(t_arr)),
    }
