"""Two-view pointmap operations: pose, scale and focal recovery.

A *pointmap* is a per-pixel 3D coordinate: for every pixel of a frame, where the
surface it sees is, expressed in a reference camera's frame. It is strictly
richer than a depth map -- depth is the ``z`` component of a pointmap whose
reference frame is the frame itself.

The helpers here turn pointmaps into the quantities the pipeline compares
against the classical baseline:

* :func:`umeyama_alignment` / :func:`weighted_umeyama` -- the similarity
  (scale + rotation + translation) that best maps one point cloud onto another.
* :func:`median_scale` -- the robust per-scene scale factor, i.e. the minimum
  amount of information a monocular method needs to become metric.
* :func:`pose_from_pointmaps` -- relative camera pose from a pair of pointmaps
  that share a reference frame, estimated with RANSAC PnP. The pose itself is
  scale-free; the *inter-pointmap scale consistency* is measured separately with
  a weighted Umeyama fit over the overlapping scene content.
* :func:`focal_from_pointmaps` -- focal length straight out of a pointmap, which
  is the "intrinsics come for free" claim made about DUSt3R-style models.

Why PnP and not just Umeyama
----------------------------
Given pointmaps ``X1`` and ``X2`` *both expressed in camera 1's frame*, the two
clouds describe the same scene, so the rigid transform between them is the
identity -- Umeyama on ``X1`` vs ``X2`` measures *consistency*, not pose. The
pose has to come from a projection constraint: ``X2`` lives in camera 1's
coordinate frame while its pixels are observed by camera 2, so
``solvePnPRansac(X2, grid2, K)`` recovers exactly ``X_cam2 = R X_cam1 + t``.
That is the estimator used below, and it is why the pointmap pose is
scale-invariant (scaling ``X2`` scales ``t`` but leaves ``R`` untouched).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

__all__ = [
    "PointmapPose",
    "depth_from_pointmap",
    "focal_from_pointmaps",
    "median_scale",
    "pointmap_from_depth",
    "pointmap_scale_consistency",
    "pose_from_pointmaps",
    "rods_from_rvec",
    "two_view_pointmap_targets",
    "umeyama_alignment",
    "weighted_umeyama",
]


# --------------------------------------------------------------------------- #
# similarity alignment
# --------------------------------------------------------------------------- #
def weighted_umeyama(
    X: np.ndarray, Y: np.ndarray, weights: np.ndarray | None = None
) -> tuple[float, np.ndarray, np.ndarray]:
    """Weighted similarity transform aligning ``X`` onto ``Y`` (Umeyama 1991).

    Finds ``s, R, t`` minimising ``sum_i w_i || s R x_i + t - y_i ||^2``, with
    ``R`` a proper rotation (``det R = +1``) and ``s > 0``. With uniform
    weights this is :func:`umeyama_alignment`.

    Args:
        X: ``(N, 3)`` source points.
        Y: ``(N, 3)`` target points in correspondence with ``X``.
        weights: Optional ``(N,)`` non-negative weights (e.g. pointmap
            confidences). Negative weights are an error.

    Returns:
        ``(s, R, t)``: positive scale, ``(3, 3)`` rotation, ``(3,)`` translation.

    Raises:
        ValueError: On shape mismatch, fewer than three correspondences, a
            degenerate (zero-variance) source, or negative weights.
    """
    X = np.asarray(X, np.float64).reshape(-1, 3)
    Y = np.asarray(Y, np.float64).reshape(-1, 3)
    if X.shape != Y.shape:
        raise ValueError(f"point sets must match, got {X.shape} and {Y.shape}")
    if X.shape[0] < 3:
        raise ValueError(f"need at least 3 correspondences, got {X.shape[0]}")

    n = X.shape[0]
    if weights is None:
        w = np.full(n, 1.0 / n)
    else:
        w = np.asarray(weights, np.float64).reshape(-1)
        if w.shape[0] != n:
            raise ValueError(f"weights must have length {n}, got {w.shape[0]}")
        if not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("weights must be finite and non-negative")
        total = w.sum()
        if total <= 0:
            raise ValueError("weights sum to zero")
        w = w / total

    mu_x = (w[:, None] * X).sum(axis=0)
    mu_y = (w[:, None] * Y).sum(axis=0)
    Xc = X - mu_x
    Yc = Y - mu_y

    var_x = float((w * (Xc**2).sum(axis=1)).sum())
    if var_x < 1e-18:
        raise ValueError("source points are degenerate (zero weighted variance)")

    # cov = sum_i w_i (y_i - mu_y)(x_i - mu_x)^T
    cov = (w[:, None] * Yc).T @ Xc
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / var_x)
    t = mu_y - s * (R @ mu_x)
    return s, R, t


def umeyama_alignment(X: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Unweighted similarity transform aligning ``X`` onto ``Y``.

    Args:
        X: ``(N, 3)`` source points.
        Y: ``(N, 3)`` target points in correspondence with ``X``.

    Returns:
        ``(s, R, t)`` as in :func:`weighted_umeyama`.
    """
    return weighted_umeyama(X, Y, None)


def rods_from_rvec(rvec: np.ndarray) -> np.ndarray:
    """Convert an OpenCV rotation vector to a 3x3 matrix."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    return np.asarray(R, np.float64)


def median_scale(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Robust scale factor mapping ``pred`` depth onto ``gt`` depth.

    The *median* ratio is used rather than the mean so a handful of wildly wrong
    predictions cannot move the whole scene's scale.

    Args:
        pred: Predicted depth (any shape).
        gt: Ground-truth depth, broadcast against ``pred``.
        mask: Optional boolean mask selecting valid pixels.

    Returns:
        ``median(gt / pred)`` over valid, positive samples, or ``nan`` when there
        are none.
    """
    p = np.asarray(pred, np.float64)
    g = np.asarray(gt, np.float64)
    p, g = np.broadcast_arrays(p, g)
    valid = np.isfinite(p) & np.isfinite(g) & (p > 1e-6) & (g > 1e-6)
    if mask is not None:
        valid &= np.asarray(mask, bool)
    if not np.any(valid):
        return float("nan")
    return float(np.median(g[valid] / p[valid]))


# --------------------------------------------------------------------------- #
# pointmap <-> depth
# --------------------------------------------------------------------------- #
def pointmap_from_depth(
    depth: np.ndarray, K: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    """Lift a depth map into a camera-frame pointmap.

    Args:
        depth: ``(H, W)`` camera-frame depth in metres.
        K: ``(3, 3)`` intrinsics.
        mask: Optional ``(H, W)`` validity mask; invalid pixels become ``0``.

    Returns:
        ``(H, W, 3)`` pointmap in the same camera frame.
    """
    depth = np.asarray(depth, np.float64)
    h, w = depth.shape
    K = np.asarray(K, np.float64)
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    rays = np.stack([xs, ys, np.ones_like(xs)], -1).reshape(-1, 3) @ np.linalg.inv(K).T
    pts = rays * depth.reshape(-1, 1)
    if mask is not None:
        pts = np.where(np.asarray(mask).reshape(-1, 1), pts, 0.0)
    return pts.reshape(h, w, 3)


def depth_from_pointmap(pointmap: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Extract the camera-frame depth (``z``) from a pointmap.

    Args:
        pointmap: ``(H, W, 3)`` pointmap in the camera frame.
        mask: Optional ``(H, W)`` validity mask; invalid pixels become ``0``.

    Returns:
        ``(H, W)`` depth.

    Raises:
        ValueError: If the last axis is not 3.
    """
    pm = np.asarray(pointmap, np.float64)
    if pm.shape[-1] != 3:
        raise ValueError(f"pointmap must have 3 channels, got {pm.shape[-1]}")
    d = pm[..., 2].copy()
    if mask is not None:
        d = np.where(np.asarray(mask, bool), d, 0.0)
    return d


# --------------------------------------------------------------------------- #
# pose from pointmaps
# --------------------------------------------------------------------------- #
@dataclass
class PointmapPose:
    """Relative pose recovered from a pair of pointmaps.

    Attributes:
        R: ``(3, 3)`` rotation taking camera 1 coordinates to camera 2.
        t: ``(3,)`` unit-norm translation, same convention.
        scale: Inter-pointmap scale from a weighted Umeyama fit over the
            overlapping scene content (``1.0`` means the two pointmaps agree on
            scale).
        num_inliers: PnP inliers used for the final estimate.
        inlier_ratio: Inlier fraction over the confidence-filtered points.
        num_points: Points fed to RANSAC.
        reprojection_median_px: Median PnP reprojection residual over inliers.
        rotation_error_deg: Rotation error against ground truth, when supplied.
        translation_angle_error_deg: Translation-direction error, when supplied.
        scale_note: Human-readable reminder that the pose is scale-free.
    """

    R: np.ndarray
    t: np.ndarray
    scale: float
    num_inliers: int
    inlier_ratio: float
    num_points: int
    reprojection_median_px: float = float("nan")
    rotation_error_deg: float | None = None
    translation_angle_error_deg: float | None = None
    scale_note: str = (
        "The relative translation is unit-norm: a two-view pose from a single "
        "camera (learned or classical) never observes metric scale. The scale "
        "field measures inter-pointmap depth consistency, not metric scale."
    )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary."""
        return {
            "R": np.asarray(self.R, float).tolist(),
            "t_unit": np.asarray(self.t, float).tolist(),
            "scale": float(self.scale),
            "num_points": int(self.num_points),
            "num_inliers": int(self.num_inliers),
            "inlier_ratio": float(self.inlier_ratio),
            "reprojection_median_px": float(self.reprojection_median_px),
            "rotation_error_deg": self.rotation_error_deg,
            "translation_angle_error_deg": self.translation_angle_error_deg,
            "scale_note": self.scale_note,
        }


def pointmap_scale_consistency(
    pm1: np.ndarray,
    pm2: np.ndarray,
    conf: np.ndarray | None = None,
    K: np.ndarray | None = None,
    min_depth: float = 1e-9,
) -> dict[str, float]:
    """Measure how well two pointmaps agree on scale over their overlapping content.

    Both pointmaps live in camera 1's frame, so "the same surface" is decided
    *geometrically*: each valid point of ``pm2`` is projected into frame 1 with
    the identity pose and its ray is compared against the depth ``pm1`` reports
    along that same ray. Their norm ratio is exactly ``1`` when the two
    pointmaps are metric-consistent, and ``1/s`` when ``pm2`` is ``s`` times too
    large. The median over the overlap makes it robust to occluded points.

    This is deliberately *not* a 3D nearest-neighbour Umeyama between the two
    clouds: on a ground-plane-dominated scene every point has neighbours at
    every scale, so nearest-neighbour matching is ill-posed. Posing the question
    as "same ray" makes it exactly determined. (An earlier nearest-neighbour
    version of this function was measured to return 0.92 on perfectly consistent
    ground-truth pointmaps -- i.e. it was wrong -- which is why it was replaced.)

    Args:
        pm1: ``(H, W, 3)`` pointmap of frame 1, in camera 1's frame.
        pm2: ``(H, W, 3)`` pointmap of frame 2, **also in camera 1's frame**.
        conf: Optional ``(H, W)`` confidence for frame 2; the bottom half by
            confidence is discarded before the median.
        K: ``(3, 3)`` intrinsics, required to define the rays.
        min_depth: Points with ``z`` at or below this are ignored.

    Returns:
        Dict with ``scale`` (factor taking ``pm2`` onto ``pm1``),
        ``scale_iqr`` (inter-quartile spread of the ratios, a drift indicator),
        ``num_overlap`` and ``overlap_ratio``.

    Raises:
        ValueError: If ``K`` is missing, or the pointmaps are not ``(H, W, 3)``.
    """
    if K is None:
        raise ValueError("pointmap_scale_consistency needs the intrinsics K to compare rays")
    pm1 = np.asarray(pm1, np.float64)
    pm2 = np.asarray(pm2, np.float64)
    if pm1.ndim != 3 or pm1.shape[-1] != 3 or pm1.shape != pm2.shape:
        raise ValueError(f"expected matching (H, W, 3) pointmaps, got {pm1.shape} and {pm2.shape}")
    h, w = pm1.shape[:2]
    K = np.asarray(K, np.float64)

    X2 = pm2.reshape(-1, 3)
    valid2 = np.isfinite(X2).all(axis=1) & (X2[:, 2] > min_depth)
    if conf is not None:
        c = np.asarray(conf, np.float64).reshape(-1)
        if len(c) != len(X2):
            raise ValueError("conf must match the pointmap pixel grid")
        thr = float(np.percentile(c[valid2], 50)) if valid2.any() else 0.0
        valid2 &= np.isfinite(c) & (c >= thr)

    empty = {
        "scale": float("nan"),
        "scale_iqr": float("nan"),
        "num_overlap": 0,
        "overlap_ratio": 0.0,
    }
    if valid2.sum() < 10:
        return empty

    # project pm2's points into frame 1: they are already in camera 1's frame,
    # so the transform is the identity pinhole projection
    z = np.where(valid2, X2[:, 2], 1.0)
    proj = (K[:2, :2] @ X2[:, :2].T).T / z[:, None] + K[:2, 2]
    u = np.rint(proj[:, 0]).astype(np.int64)
    v = np.rint(proj[:, 1]).astype(np.int64)
    inside = valid2 & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if inside.sum() < 10:
        return empty

    flat = v[inside] * w + u[inside]
    X1 = pm1.reshape(-1, 3)[flat]
    X2i = X2[inside]
    ok = np.isfinite(X1).all(axis=1) & (X1[:, 2] > min_depth) & (np.linalg.norm(X1, axis=1) > 1e-9)
    if ok.sum() < 10:
        return empty

    ratios = np.linalg.norm(X1[ok], axis=1) / np.linalg.norm(X2i[ok], axis=1)
    lo, hi = np.percentile(ratios, [25, 75])
    return {
        "scale": float(np.median(ratios)),
        "scale_iqr": float(hi - lo),
        "num_overlap": int(ok.sum()),
        "overlap_ratio": float(ok.sum()) / float(max(1, valid2.sum())),
    }


def pose_from_pointmaps(
    pm1: np.ndarray,
    pm2: np.ndarray,
    conf: np.ndarray | None = None,
    K: np.ndarray | None = None,
    ransac_iters: int = 500,
    reproj_thresh_px: float = 2.0,
    confidence: float = 0.999,
    seed: int = 0,
    min_conf: float | None = None,
    gt_R: np.ndarray | None = None,
    gt_t: np.ndarray | None = None,
) -> PointmapPose:
    """Recover the relative camera pose from a pair of pointmaps.

    ``pm2`` is expressed in camera 1's frame but its pixels are observed by
    camera 2, so RANSAC PnP on ``(pm2, grid2)`` returns exactly
    ``X_cam2 = R X_cam1 + t``. The translation is returned unit-norm because a
    two-view pose has no metric scale.

    Args:
        pm1: ``(H, W, 3)`` pointmap of frame 1, in camera 1's frame.
        pm2: ``(H, W, 3)`` pointmap of frame 2, **also in camera 1's frame**.
        conf: Optional ``(H, W)`` confidence map for frame 2.
        K: ``(3, 3)`` intrinsics. Required.
        ransac_iters: RANSAC iteration cap.
        reproj_thresh_px: RANSAC reprojection threshold in pixels.
        confidence: RANSAC confidence.
        seed: Reserved for determinism of the (deterministic) OpenCV RANSAC.
        min_conf: Minimum confidence to keep a pixel; defaults to the median.
        gt_R: Optional ground-truth rotation, to fill the error fields.
        gt_t: Optional ground-truth translation direction.

    Returns:
        A :class:`PointmapPose`.

    Raises:
        ValueError: If ``K`` is missing or too few valid points remain.
    """
    if K is None:
        raise ValueError("pose_from_pointmaps needs the intrinsics K to run PnP")
    K = np.asarray(K, np.float64)
    pm2 = np.asarray(pm2, np.float64)
    if pm2.ndim != 3 or pm2.shape[-1] != 3:
        raise ValueError(f"pm2 must be (H, W, 3), got {pm2.shape}")
    h, w = pm2.shape[:2]

    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    pix = np.stack([xs.reshape(-1), ys.reshape(-1)], 1).astype(np.float64)
    obj = pm2.reshape(-1, 3)

    ok = np.isfinite(obj).all(1) & (obj[:, 2] > 1e-9)
    if conf is not None:
        c = np.asarray(conf, np.float64).reshape(-1)
        thr = float(np.percentile(c[ok], 50)) if min_conf is None else float(min_conf)
        ok &= np.isfinite(c) & (c >= thr)
    if ok.sum() < 10:
        raise ValueError(f"only {int(ok.sum())} valid pointmap points for pose estimation")

    obj_v, pix_v = obj[ok], pix[ok]
    np.random.default_rng(seed)
    cv2.setRNGSeed(int(seed) & 0x7FFFFFFF)
    _n_in, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_v.astype(np.float64),
        pix_v.astype(np.float64),
        K,
        None,
        iterationsCount=int(ransac_iters),
        reprojectionError=float(reproj_thresh_px),
        confidence=float(confidence),
        flags=cv2.SOLVEPNP_AP3P,
    )
    if inliers is None or len(inliers) < 4:
        raise ValueError("PnP failed to find a consistent relative pose from the pointmaps")
    inl = np.asarray(inliers).ravel().astype(int)

    # refine on the inliers with the iterative solver
    rvec, tvec = cv2.solvePnP(
        obj_v[inl].astype(np.float64),
        pix_v[inl].astype(np.float64),
        K,
        None,
        rvec=rvec,
        tvec=tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )[1:3]

    R = rods_from_rvec(rvec)
    t = np.asarray(tvec, np.float64).ravel()
    tn = float(np.linalg.norm(t))
    t_unit = t / tn if tn > 0 else t

    proj, _ = cv2.projectPoints(obj_v[inl], rvec, tvec, K, None)
    res = np.linalg.norm(proj.reshape(-1, 2) - pix_v[inl], axis=1)

    consist = pointmap_scale_consistency(pm1, pm2, conf, K=K)

    res_obj = PointmapPose(
        R=R,
        t=t_unit,
        scale=float(consist["scale"]),
        num_inliers=int(len(inl)),
        inlier_ratio=float(len(inl)) / float(ok.sum()),
        num_points=int(ok.sum()),
        reprojection_median_px=float(np.median(res)) if res.size else float("nan"),
    )
    if gt_R is not None:
        dR = R @ np.asarray(gt_R, np.float64).T
        res_obj.rotation_error_deg = float(
            np.degrees(np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)))
        )
    if gt_t is not None:
        g = np.asarray(gt_t, np.float64).ravel()
        gn = np.linalg.norm(g)
        if gn > 1e-12:
            res_obj.translation_angle_error_deg = float(
                np.degrees(np.arccos(np.clip(float(np.dot(t_unit, g / gn)), -1.0, 1.0)))
            )
    return res_obj


# --------------------------------------------------------------------------- #
# focal length
# --------------------------------------------------------------------------- #
def focal_from_pointmaps(
    pm1: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Estimate focal length and principal point from a pointmap alone.

    For a pointmap produced by a pinhole camera, ``u = cx + fx * X/Z`` and
    ``v = cy + fy * Y/Z`` hold exactly, so two independent linear least-squares
    fits recover ``fx, fy, cx, cy`` with no calibration target and no image
    content. This is the "intrinsics come for free" property claimed for
    DUSt3R-style models, measured here against the calibrated intrinsics.

    Args:
        pm1: ``(H, W, 3)`` pointmap in the camera frame.
        mask: Optional ``(H, W)`` validity mask.

    Returns:
        Dict with ``fx``, ``fy``, ``cx``, ``cy``, ``focal_mean`` and
        ``num_points``.
    """
    pm = np.asarray(pm1, np.float64)
    if pm.ndim != 3 or pm.shape[-1] != 3:
        raise ValueError(f"pm1 must be (H, W, 3), got {pm.shape}")
    h, w = pm.shape[:2]
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    u = xs.reshape(-1).astype(np.float64)
    v = ys.reshape(-1).astype(np.float64)
    X = pm.reshape(-1, 3)
    z = X[:, 2]
    ok = np.isfinite(X).all(1) & (z > 1e-9)
    if mask is not None:
        ok &= np.asarray(mask).reshape(-1).astype(bool)

    nan = float("nan")
    if ok.sum() < 8:
        return {
            "fx": nan,
            "fy": nan,
            "cx": nan,
            "cy": nan,
            "focal_mean": nan,
            "num_points": int(ok.sum()),
        }

    xr = X[ok, 0] / z[ok]
    yr = X[ok, 1] / z[ok]
    A_u = np.stack([np.ones_like(xr), xr], 1)
    A_v = np.stack([np.ones_like(yr), yr], 1)
    cx, fx = np.linalg.lstsq(A_u, u[ok], rcond=None)[0]
    cy, fy = np.linalg.lstsq(A_v, v[ok], rcond=None)[0]
    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "focal_mean": float(0.5 * (fx + fy)),
        "num_points": int(ok.sum()),
    }


def two_view_pointmap_targets(
    sequence: Any, i: int, j: int, mask_min_depth: float = 1e-3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Build ground-truth pointmaps for a frame pair, both in frame ``i``'s frame.

    This is the DUSt3R supervision target: view 1's pointmap is its own
    backprojection, and view 2's pointmap is its own backprojection rigidly
    moved into frame 1's coordinate system, so a perfect prediction is a
    consistent two-view geometry.

    Args:
        sequence: A :class:`monogeo.data.Sequence` (or anything with ``depths``,
            ``poses`` and ``K``).
        i: Index of the reference (first) frame.
        j: Index of the second frame.
        mask_min_depth: Depths at or below this are treated as invalid.

    Returns:
        ``(pm1, pm2, valid1, valid2, norm)`` where ``pm1``/``pm2`` are
        ``(H, W, 3)`` float32 pointmaps, ``valid1``/``valid2`` are ``(H, W)``
        boolean masks, and ``norm`` is the per-sample scale factor (mean
        distance of the reference view's valid points from the camera centre).
    """
    from .twoview import relative_pose

    K = np.asarray(sequence.K, np.float64)
    d1 = np.asarray(sequence.depths[i], np.float64)
    d2 = np.asarray(sequence.depths[j], np.float64)
    v1 = d1 > mask_min_depth
    v2 = d2 > mask_min_depth

    pm1 = pointmap_from_depth(d1, K, v1)
    pm2_camj = pointmap_from_depth(d2, K, v2)

    # R, t map camera i -> camera j; move view 2's points into camera i's frame
    R, t = relative_pose(sequence.poses, i, j)
    Rji = R.T
    tji = -R.T @ t
    pm2 = pm2_camj @ Rji.T + tji

    if v1.any():
        norm = float(np.mean(np.linalg.norm(pm1[v1], axis=1)))
    else:  # pragma: no cover - degenerate sequence
        norm = 1.0
    norm = max(norm, 1e-6)
    return (
        pm1.astype(np.float32),
        pm2.astype(np.float32),
        v1,
        v2,
        norm,
    )
