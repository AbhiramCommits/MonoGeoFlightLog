"""Classical projective two-view geometry baseline.

The pipeline is the textbook one, written out explicitly so that every stage is
inspectable and every intermediate number can be logged:

``match_features`` -> ``estimate_pose`` (essential matrix + ``recoverPose``) ->
``triangulate`` -> ``reprojection_error`` -> ``scale_from_gt`` / ``pose_error``.

Because a single calibrated camera cannot observe metric scale, triangulation
returns points in the first camera's frame with a *unit* baseline. Recovering
the true baseline requires one prior: here, a single ground-truth depth sample
(``scale_from_gt``). That is the honest statement of the "monocular scale"
problem, and the recovered factor is reported as a headline number.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from . import synth
from .cli import cfg_get, save_json

__all__ = [
    "FeatureMatches",
    "TwoViewResult",
    "match_features",
    "estimate_pose",
    "triangulate",
    "reprojection_error",
    "scale_from_gt",
    "pose_error",
    "rotation_error_deg",
    "translation_angle_deg",
    "relative_pose",
    "depth_from_triangulation",
    "run_two_view",
]


# --------------------------------------------------------------------------- #
# feature matching
# --------------------------------------------------------------------------- #
@dataclass
class FeatureMatches:
    """Correspondences between two frames plus matching statistics.

    Attributes:
        points1: ``(N, 2)`` float32 pixel coordinates in the first image.
        points2: ``(N, 2)`` float32 pixel coordinates in the second image.
        detector: ``"sift"`` or ``"orb"``.
        n_keypoints1: Keypoints found in image 1.
        n_keypoints2: Keypoints found in image 2.
        n_ratio: Matches surviving Lowe's ratio test.
        n_mutual: Matches surviving the mutual-nearest-neighbour test as well.
        ratio_threshold: Ratio test threshold used.
    """

    points1: np.ndarray
    points2: np.ndarray
    detector: str = "sift"
    n_keypoints1: int = 0
    n_keypoints2: int = 0
    n_ratio: int = 0
    n_mutual: int = 0
    ratio_threshold: float = 0.75

    def __len__(self) -> int:
        return int(self.points1.shape[0])

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable matching statistics."""
        return {
            "detector": self.detector,
            "n_matches": len(self),
            "n_keypoints1": int(self.n_keypoints1),
            "n_keypoints2": int(self.n_keypoints2),
            "n_after_ratio_test": int(self.n_ratio),
            "n_after_mutual_nn": int(self.n_mutual),
            "ratio_threshold": float(self.ratio_threshold),
        }


def _to_gray(img: np.ndarray) -> np.ndarray:
    """Convert an RGB/BGR frame to 8-bit greyscale."""
    if img.ndim == 2:
        return img.astype(np.uint8, copy=False)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def match_features(
    img1: np.ndarray,
    img2: np.ndarray,
    max_features: int = 4000,
    ratio: float = 0.75,
    detector: str = "sift",
    mutual_nn: bool = True,
    contrast_threshold: float = 0.02,
) -> FeatureMatches:
    """Detect and match features between two frames.

    Uses SIFT with a Lowe ratio test followed by a mutual nearest-neighbour
    cross-check, which removes most of the asymmetric outliers that a ratio
    test alone leaves behind. Falls back to ORB (Hamming matching) if SIFT is
    unavailable in the installed OpenCV build.

    Args:
        img1: First frame, ``(H, W)`` or ``(H, W, 3)``.
        img2: Second frame, same shape as ``img1``.
        max_features: Cap on the number of detected keypoints.
        ratio: Lowe ratio-test threshold on the best/second-best distance.
        detector: ``"sift"`` or ``"orb"``. Unknown values fall back to SIFT.
        mutual_nn: Also require the match to be mutual nearest neighbours.
        contrast_threshold: SIFT contrast threshold. The default is lowered from
            OpenCV's ``0.04`` because these are small (<=128 px tall) frames and
            a stricter threshold leaves too few keypoints to constrain the
            essential matrix.

    Returns:
        :class:`FeatureMatches` with the surviving correspondences (possibly
        empty, in which case ``len(matches) == 0``).

    Raises:
        ValueError: If the two frames have different shapes.
    """
    g1, g2 = _to_gray(img1), _to_gray(img2)
    if g1.shape != g2.shape:
        raise ValueError(f"frame shapes differ: {g1.shape} vs {g2.shape}")

    det = (detector or "sift").lower()
    if det == "orb" or not hasattr(cv2, "SIFT_create"):
        create = getattr(cv2, "ORB_create", None)
        if create is None:  # pragma: no cover - OpenCV always has one of the two
            raise RuntimeError("neither SIFT nor ORB is available in this OpenCV build")
        engine = create(nfeatures=int(max_features))
        norm = cv2.NORM_HAMMING
        det = "orb"
    else:
        engine = cv2.SIFT_create(
            nfeatures=int(max_features), contrastThreshold=float(contrast_threshold)
        )
        norm = cv2.NORM_L2
        det = "sift"

    kp1, d1 = engine.detectAndCompute(g1, None)
    kp2, d2 = engine.detectAndCompute(g2, None)
    nkp1, nkp2 = len(kp1 or []), len(kp2 or [])
    empty = FeatureMatches(
        np.zeros((0, 2), np.float32),
        np.zeros((0, 2), np.float32),
        detector=det,
        n_keypoints1=nkp1,
        n_keypoints2=nkp2,
        ratio_threshold=float(ratio),
    )
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return empty

    bf = cv2.BFMatcher(norm)
    knn12 = bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in (pair for pair in knn12 if len(pair) == 2) if m.distance < ratio * n.distance]
    if not good:
        return empty

    n_ratio = len(good)
    if mutual_nn:
        knn21 = bf.knnMatch(d2, d1, k=2)
        back: dict[int, int] = {}
        for pair in knn21:
            if len(pair) == 2:
                back[pair[0].queryIdx] = pair[0].trainIdx
        good = [m for m in good if back.get(m.trainIdx, -1) == m.queryIdx]

    pts1 = np.array([kp1[m.queryIdx].pt for m in good], np.float32).reshape(-1, 2)
    pts2 = np.array([kp2[m.trainIdx].pt for m in good], np.float32).reshape(-1, 2)
    return FeatureMatches(
        pts1,
        pts2,
        detector=det,
        n_keypoints1=nkp1,
        n_keypoints2=nkp2,
        n_ratio=n_ratio,
        n_mutual=len(good),
        ratio_threshold=float(ratio),
    )


# --------------------------------------------------------------------------- #
# pose estimation
# --------------------------------------------------------------------------- #
@dataclass
class TwoViewResult:
    """Relative pose and triangulation for one frame pair.

    Attributes:
        R: ``(3, 3)`` rotation taking the first camera frame to the second.
        t: ``(3,)`` unit-norm translation, same convention.
        essential: ``(3, 3)`` essential matrix from RANSAC.
        inlier_mask: ``(N,)`` boolean RANSAC mask over the input matches.
        inlier_ratio: Fraction of matches that were RANSAC inliers.
        num_matches: Number of input matches.
        num_inliers: Number of RANSAC inliers.
    """

    R: np.ndarray
    t: np.ndarray
    essential: np.ndarray
    inlier_mask: np.ndarray
    inlier_ratio: float
    num_matches: int
    num_inliers: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable pose summary."""
        return {
            "num_matches": int(self.num_matches),
            "num_inliers": int(self.num_inliers),
            "inlier_ratio": float(self.inlier_ratio),
            "R": np.asarray(self.R, float).tolist(),
            "t_unit": np.asarray(self.t, float).tolist(),
        }


def estimate_pose(
    matches: FeatureMatches,
    K: np.ndarray,
    ransac_prob: float = 0.999,
    threshold_px: float = 1.0,
    refine: bool = True,
) -> TwoViewResult:
    """Recover relative rotation and unit translation from correspondences.

    Runs ``cv2.findEssentialMat`` with RANSAC followed by ``cv2.recoverPose``,
    which disambiguates the four-fold decomposition with a cheirality check.

    Args:
        matches: Correspondences from :func:`match_features`.
        K: ``(3, 3)`` intrinsics.
        ransac_prob: RANSAC confidence.
        threshold_px: RANSAC inlier threshold in pixels.
        refine: Re-estimate the essential matrix from the inliers only.

    Returns:
        :class:`TwoViewResult`.

    Raises:
        ValueError: If fewer than 5 correspondences are available, or if
            ``findEssentialMat`` fails to return a solution.
    """
    pts1 = np.asarray(matches.points1, np.float64).reshape(-1, 2)
    pts2 = np.asarray(matches.points2, np.float64).reshape(-1, 2)
    if len(pts1) < 5:
        raise ValueError(f"need at least 5 correspondences, got {len(pts1)}")

    K = np.asarray(K, np.float64)
    emat, mask = cv2.findEssentialMat(
        pts1, pts2, K, cv2.RANSAC, float(ransac_prob), float(threshold_px)
    )
    if emat is None or emat.size == 0:
        raise ValueError("findEssentialMat failed to find a solution")
    emat = np.asarray(emat, np.float64).reshape(3, 3)
    mask = np.asarray(mask).ravel().astype(bool)

    if refine and mask.sum() >= 8:
        emat2, mask2 = cv2.findEssentialMat(
            pts1[mask], pts2[mask], K, cv2.RANSAC, float(ransac_prob), float(threshold_px)
        )
        if emat2 is not None and emat2.size == 9:
            emat = np.asarray(emat2, np.float64).reshape(3, 3)
            full = np.zeros(len(pts1), bool)
            full[np.nonzero(mask)[0][np.asarray(mask2).ravel().astype(bool)]] = True
            mask = full

    inl = np.nonzero(mask)[0]
    if len(inl) >= 5:
        _, R, t, pose_mask = cv2.recoverPose(emat, pts1[inl], pts2[inl], K)
        cheir = np.zeros(len(pts1), bool)
        cheir[inl[np.asarray(pose_mask).ravel().astype(bool)]] = True
    else:  # pragma: no cover - extremely rare fallback
        _, R, t, pose_mask = cv2.recoverPose(emat, pts1, pts2, K)
        cheir = np.asarray(pose_mask).ravel().astype(bool)
    R = np.asarray(R, np.float64)
    t = np.asarray(t, np.float64).ravel()
    tn = np.linalg.norm(t)
    if tn > 0:
        t = t / tn

    combined = mask & cheir if cheir.shape == mask.shape else mask
    if combined.sum() < 3:
        combined = mask

    return TwoViewResult(
        R=R,
        t=t,
        essential=emat,
        inlier_mask=combined,
        inlier_ratio=float(combined.sum()) / float(len(pts1)),
        num_matches=int(len(pts1)),
        num_inliers=int(combined.sum()),
    )


# --------------------------------------------------------------------------- #
# triangulation
# --------------------------------------------------------------------------- #
def triangulate(
    matches: FeatureMatches,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    mask: np.ndarray | None = None,
    min_parallax_deg: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear triangulation with cheirality and parallax filtering.

    Args:
        matches: Correspondences.
        K: ``(3, 3)`` intrinsics.
        R: ``(3, 3)`` rotation from camera 1 to camera 2.
        t: ``(3,)`` translation from camera 1 to camera 2 (unit norm is fine;
            the triangulated points inherit that scale).
        mask: Optional ``(N,)`` boolean selection (e.g. RANSAC inliers).
        min_parallax_deg: Minimum angle between the two viewing rays. Points
            below it are numerically meaningless for a monocular pair and are
            dropped.

    Returns:
        ``(points3d, valid)``: ``(M, 3)`` float64 points in camera 1's frame and
        an ``(N,)`` boolean mask selecting the input matches that produced them.
    """
    pts1 = np.asarray(matches.points1, np.float64).reshape(-1, 2)
    pts2 = np.asarray(matches.points2, np.float64).reshape(-1, 2)
    K = np.asarray(K, np.float64)

    if mask is None:
        sel = np.ones(len(pts1), bool)
    else:
        sel = np.asarray(mask).ravel().astype(bool)
    if sel.sum() < 2:
        return np.zeros((0, 3), np.float64), sel

    R = np.asarray(R, np.float64)
    t = np.asarray(t, np.float64).ravel()

    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    X = cv2.triangulatePoints(P1, P2, pts1[sel].T, pts2[sel].T)
    X = (X[:3] / X[3]).T

    # cheirality: the point must be in front of both cameras
    z1 = X[:, 2]
    X2 = X @ R.T + t
    z2 = X2[:, 2]
    front = (z1 > 1e-9) & (z2 > 1e-9)

    # parallax: angle between the two rays through the point
    v1 = X - np.zeros(3)
    v2 = X - (-R.T @ t)
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    ok = n1 > 1e-12
    cosang = np.ones(len(X))
    cosang[ok] = np.clip(
        np.sum(v1[ok] * v2[ok], axis=1) / (n1[ok] * n2[ok]), -1.0, 1.0
    )
    parallax = np.degrees(np.arccos(cosang))
    good = front & (parallax >= float(min_parallax_deg))

    full = np.zeros(len(pts1), bool)
    full[np.nonzero(sel)[0][good]] = True
    return X[good], full


def reprojection_error(
    points3d: np.ndarray,
    matches: FeatureMatches,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    mask: np.ndarray | None = None,
    view: str = "second",
) -> dict[str, float]:
    """Reprojection error of triangulated points, in pixels.

    Args:
        points3d: ``(M, 3)`` points in camera 1's frame.
        matches: Correspondences (used for the reference pixel positions).
        K: ``(3, 3)`` intrinsics.
        R: ``(3, 3)`` rotation from camera 1 to camera 2.
        t: ``(3,)`` translation from camera 1 to camera 2, in the same scale as
            ``points3d``.
        mask: Optional ``(N,)`` boolean mask selecting which matches produced
            ``points3d``.
        view: ``"second"`` (default) or ``"first"``. The second view is the
            meaningful one: points triangulated *from* the first view
            reproject onto it by construction.

    Returns:
        Dict with ``mean``, ``median``, ``rms`` and ``max`` error in pixels.
    """
    K = np.asarray(K, np.float64)
    R = np.asarray(R, np.float64)
    t = np.asarray(t, np.float64).ravel()
    pts2 = np.asarray(matches.points2, np.float64).reshape(-1, 2)
    pts1 = np.asarray(matches.points1, np.float64).reshape(-1, 2)
    if mask is not None:
        sel = np.asarray(mask).ravel().astype(bool)
        ref = (pts2 if view == "second" else pts1)[sel]
    else:
        ref = pts2 if view == "second" else pts1

    if len(points3d) == 0 or len(ref) == 0:
        return {"mean": float("nan"), "median": float("nan"), "rms": float("nan"), "max": float("nan"), "n": 0}

    X = np.asarray(points3d, np.float64)
    if view == "second":
        Xc = X @ R.T + t
    else:
        Xc = X
    z = Xc[:, 2]
    ok = z > 1e-9
    proj = np.full_like(Xc[:, :2], np.nan)
    proj[ok] = (K[:2, :2] @ Xc[ok, :2].T).T / z[ok, None] + K[:2, 2]
    err = np.linalg.norm(proj - ref[: len(X)], axis=1)
    err = err[np.isfinite(err)]
    if err.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "rms": float("nan"), "max": float("nan"), "n": 0}
    return {
        "mean": float(err.mean()),
        "median": float(np.median(err)),
        "rms": float(np.sqrt(np.mean(err**2))),
        "max": float(err.max()),
        "n": int(err.size),
    }


# --------------------------------------------------------------------------- #
# scale and pose comparison
# --------------------------------------------------------------------------- #
def scale_from_gt(
    points3d: np.ndarray,
    gt_depth: np.ndarray,
    points1: np.ndarray | None = None,
    K: np.ndarray | None = None,
    min_depth: float = 0.1,
) -> dict[str, float]:
    """Recover the missing monocular scale factor from ground-truth depth.

    Triangulation from a single calibrated camera yields points whose distance
    from the camera is only known up to a global factor (the unknown baseline).
    Comparing the triangulated ``Z`` against a ground-truth depth map at the
    same pixels recovers that factor as a robust median ratio.

    Args:
        points3d: ``(M, 3)`` triangulated points in camera 1's frame.
        gt_depth: ``(H, W)`` ground-truth camera-frame depth of frame 1.
        points1: ``(M, 2)`` pixel coordinates of the points (the matched
            keypoints in frame 1). If ``None``, ``K`` must be given so the
            points can be projected.
        K: ``(3, 3)`` intrinsics; only used when ``points1`` is ``None``.
        min_depth: Ignore samples with truth shallower than this many metres.

    Returns:
        Dict with ``scale`` (multiply triangulated points by this to get
        metric units), ``n`` valid samples, and the interquartile spread of the
        per-point ratios (``iqr``, a scale-drift indicator).
    """
    X = np.asarray(points3d, np.float64)
    gt = np.asarray(gt_depth, np.float64)
    if X.size == 0:
        return {"scale": float("nan"), "n": 0, "iqr": float("nan"), "median_abs_ratio": float("nan")}

    if points1 is not None:
        uv = np.asarray(points1, np.float64)
    else:
        if K is None:
            raise ValueError("scale_from_gt needs either points1 or K")
        K = np.asarray(K, np.float64)
        z = X[:, 2]
        uv = (K[:2, :2] @ X[:, :2].T).T / z[:, None] + K[:2, 2]

    h, w = gt.shape
    px = np.rint(uv[:, 0]).astype(np.int64)
    py = np.rint(uv[:, 1]).astype(np.int64)
    inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    ratios = np.full(len(X), np.nan)
    zt = X[:, 2]
    good = inside & (zt > 1e-9)
    truth = gt[py[good], px[good]]
    valid = truth > float(min_depth)
    ratios[np.nonzero(good)[0][valid]] = truth[valid] / zt[good][valid]

    r = ratios[np.isfinite(ratios)]
    if r.size == 0:
        return {"scale": float("nan"), "n": 0, "iqr": float("nan"), "median_abs_ratio": float("nan")}
    q1, q3 = np.percentile(r, [25, 75])
    return {
        "scale": float(np.median(r)),
        "n": int(r.size),
        "iqr": float(q3 - q1),
        "median_abs_ratio": float(np.median(np.abs(r))),
    }


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """Geodesic rotation error between two rotations, in degrees."""
    R = np.asarray(R_est, np.float64) @ np.asarray(R_gt, np.float64).T
    # guard against tiny numerical drift outside acos domain
    cos = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def translation_angle_deg(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    """Angular error between two translation *directions*, in degrees.

    Monocular two-view geometry cannot observe translation magnitude, so only
    the direction is comparable.
    """
    a = np.asarray(t_est, np.float64).ravel()
    b = np.asarray(t_gt, np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    cos = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def pose_error(
    R_est: np.ndarray,
    t_est: np.ndarray,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
) -> dict[str, float]:
    """Rotation error (deg) and translation angular error (deg).

    Args:
        R_est: Estimated rotation, camera 1 -> camera 2.
        t_est: Estimated translation direction, camera 1 -> camera 2.
        R_gt: Ground-truth rotation, camera 1 -> camera 2.
        t_gt: Ground-truth translation, camera 1 -> camera 2 (any scale).

    Returns:
        Dict with ``rotation_deg`` and ``translation_deg``.
    """
    return {
        "rotation_deg": rotation_error_deg(R_est, R_gt),
        "translation_deg": translation_angle_deg(t_est, t_gt),
    }


def relative_pose(poses: np.ndarray, i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
    """Relative rotation and translation from frame ``i`` to frame ``j``.

    Args:
        poses: ``(T, 4, 4)`` world->camera matrices.
        i: Target frame index.
        j: Source frame index.

    Returns:
        ``(R, t)`` such that ``X_j = R @ X_i + t``.
    """
    poses = np.asarray(poses, np.float64)
    T = poses[j] @ np.linalg.inv(poses[i])
    return T[:3, :3].copy(), T[:3, 3].copy()


def depth_from_triangulation(
    points3d: np.ndarray,
    points1: np.ndarray,
    size: tuple[int, int],
    K: np.ndarray | None = None,
    splat: int = 3,
) -> np.ndarray:
    """Splat triangulated points into a sparse depth map for metric scoring.

    Args:
        points3d: ``(M, 3)`` points in camera 1's frame.
        points1: ``(M, 2)`` pixel coordinates of those points in frame 1.
        size: ``(height, width)`` of the output map.
        K: Unused placeholder kept for signature symmetry with the renderer.
        splat: Side length in pixels of each point's square footprint.

    Returns:
        ``(height, width)`` float32 depth map, zero where no point landed.
    """
    height, width = int(size[0]), int(size[1])
    depth = np.zeros((height, width), np.float32)
    X = np.asarray(points3d, np.float64)
    if X.size == 0:
        return depth
    uv = np.asarray(points1, np.float64)
    z = X[:, 2]
    order = np.argsort(-z, kind="stable")  # far -> near
    offs = np.arange(splat) - (splat - 1) // 2
    for k in order:
        px, py = int(round(uv[k, 0])), int(round(uv[k, 1]))
        for dy in offs:
            yy = py + dy
            if yy < 0 or yy >= height:
                continue
            for dx in offs:
                xx = px + dx
                if 0 <= xx < width:
                    depth[yy, xx] = z[k]
    return depth


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def evaluate_pair(
    frames: np.ndarray,
    depths: np.ndarray,
    poses: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    i: int,
    j: int,
    max_features: int = 4000,
    ratio: float = 0.75,
    ransac_prob: float = 0.999,
    threshold_px: float = 1.0,
) -> dict[str, Any]:
    """Run the full classical two-view pipeline on one frame pair.

    Args:
        frames: ``(T, H, W, 3)`` uint8 frames.
        depths: ``(T, H, W)`` ground-truth depth.
        poses: ``(T, 4, 4)`` world->camera poses.
        K: ``(3, 3)`` intrinsics.
        dist: Distortion coefficients (unused here; frames are already the
            images the camera produced).
        i: Index of the first frame of the pair.
        j: Index of the second frame of the pair.
        max_features: Feature budget.
        ratio: Lowe ratio threshold.
        ransac_prob: RANSAC confidence for the essential matrix.
        threshold_px: RANSAC inlier threshold in pixels.

    Returns:
        A JSON-serialisable dict with matching statistics, pose errors,
        reprojection errors and the recovered scale factor.
    """
    import time

    t0 = time.perf_counter()
    matches = match_features(
        frames[i], frames[j], max_features=max_features, ratio=ratio
    )
    out: dict[str, Any] = {
        "frame_pair": [int(i), int(j)],
        "matches": matches.to_dict(),
    }
    if len(matches) < 8:
        out["ok"] = False
        out["error"] = f"only {len(matches)} matches"
        out["runtime_ms"] = (time.perf_counter() - t0) * 1000.0
        return out

    tv = estimate_pose(
        matches, K, ransac_prob=ransac_prob, threshold_px=threshold_px
    )
    pts3d, valid = triangulate(matches, K, tv.R, tv.t, mask=tv.inlier_mask)

    R_gt, t_gt = relative_pose(poses, i, j)
    perr = pose_error(tv.R, tv.t, R_gt, t_gt)

    rep = reprojection_error(pts3d, matches, K, tv.R, tv.t, mask=valid, view="second")
    rep1 = reprojection_error(pts3d, matches, K, tv.R, tv.t, mask=valid, view="first")

    # Scale is recovered from ground truth, which is exactly the prior a
    # monocular pipeline does not have: report how much scale is missing.
    gt1 = depths[i]
    uv = matches.points1[valid] if valid.sum() == len(pts3d) else None
    sc = (
        scale_from_gt(pts3d, gt1, points1=uv)
        if uv is not None
        else scale_from_gt(pts3d, gt1, K=K)
    )
    gt_baseline = float(np.linalg.norm(t_gt))
    scale_corrected_rot = rotation_error_deg(tv.R, R_gt)

    # A depth map in metric units, for the depth metrics in step 4/5.
    depth_pred = depth_from_triangulation(
        pts3d, matches.points1[valid] if valid.sum() == len(pts3d) else matches.points1[: len(pts3d)],
        depths.shape[1:],
    )

    runtime_ms = (time.perf_counter() - t0) * 1000.0

    out.update(
        {
            "ok": True,
            "detector": matches.detector,
            "num_matches": int(len(matches)),
            "num_inliers": int(tv.num_inliers),
            "inlier_ratio": float(tv.inlier_ratio),
            "num_triangulated": int(len(pts3d)),
            "triangulation_keep_ratio": float(len(pts3d)) / float(max(1, len(matches))),
            "reprojection_error_px": {
                "view": "second",
                "mean": rep["mean"],
                "median": rep["median"],
                "rms": rep["rms"],
                "max": rep["max"],
                "n": rep["n"],
            },
            "reprojection_error_first_view_px": {
                "mean": rep1["mean"],
                "median": rep1["median"],
                "rms": rep1["rms"],
                "n": rep1["n"],
            },
            "pose_error": perr,
            "rotation_error_deg": perr["rotation_deg"],
            "translation_angle_error_deg": perr["translation_deg"],
            "scale_from_gt": sc,
            "true_baseline_m": gt_baseline,
            "depth_map_coverage": float((depth_pred > 0).mean()),
            "runtime_ms": runtime_ms,
        }
    )
    return out


def run_two_view(args: Any) -> int:
    """``monogeo two-view``: classical baseline on real synthetic pairs.

    Args:
        args: Parsed argparse namespace.

    Returns:
        Process exit code (0 on success).
    """
    import time

    from .data import load_sequence

    cfg = getattr(args, "cfg", {}) or {}
    name = getattr(args, "sequence", None) or cfg_get(cfg, "data.sequence", "forward")
    size = (int(cfg_get(cfg, "data.height", 96)), int(cfg_get(cfg, "data.width", 320)))
    num_frames = int(cfg_get(cfg, "data.num_frames", 48))
    seed = int(getattr(args, "seed", 0) or 0)
    gap = int(getattr(args, "gap", None) or cfg_get(cfg, "twoview.frame_gap", 1))
    num_pairs = int(getattr(args, "pairs", None) or cfg_get(cfg, "twoview.num_pairs", 12))
    ratio = float(cfg_get(cfg, "twoview.ratio_thresh", 0.75))
    prob = float(cfg_get(cfg, "twoview.ransac_prob", 0.999))
    thresh = float(cfg_get(cfg, "twoview.ransac_threshold_px", 1.0))
    max_features = int(cfg_get(cfg, "twoview.max_features", 4000))

    t0 = time.perf_counter()
    seq = load_sequence(
        name,
        size=size,
        num_frames=num_frames,
        seed=seed,
        root=cfg_get(cfg, "data.data_root", "data"),
        cache=bool(cfg_get(cfg, "data.cache", True)),
        fov_deg=float(cfg_get(cfg, "synth.fov_deg", 62.0)),
        distortion=cfg_get(cfg, "synth.distortion", None),
        verbose=bool(getattr(args, "verbose", False)),
    )
    load_ms = (time.perf_counter() - t0) * 1000.0

    # Evenly spaced non-overlapping pairs, all within the rendered sequence.
    max_i = len(seq) - gap
    starts = np.unique(np.linspace(0, max(0, max_i - 1), num_pairs).astype(int))
    pairs = [(int(i), int(i + gap)) for i in starts if i + gap < len(seq)]

    per_pair = []
    for i, j in pairs:
        r = evaluate_pair(
            seq.frames, seq.depths, seq.poses, seq.K, seq.dist, i, j,
            max_features=max_features, ratio=ratio, ransac_prob=prob, threshold_px=thresh,
        )
        per_pair.append(r)

    ok_pairs = [r for r in per_pair if r.get("ok")]
    if not ok_pairs:
        print("[two-view] no pair produced a valid pose estimate")
        return 1

    def med(key: str) -> float:
        vals = [r[key] for r in ok_pairs if r.get(key) is not None and np.isfinite(r[key])]
        return float(np.median(vals)) if vals else float("nan")

    def med_nested(outer: str, key: str) -> float:
        vals = []
        for r in ok_pairs:
            v = r.get(outer, {}).get(key)
            if v is not None and np.isfinite(v):
                vals.append(v)
        return float(np.median(vals)) if vals else float("nan")

    headline = max(ok_pairs, key=lambda r: r["num_inliers"])
    payload = {
        "step": "two-view",
        "method": "classical (SIFT + essential matrix RANSAC + triangulation)",
        "sequence": seq.name,
        "data_source": seq.source,
        "resolution_hw": [seq.height, seq.width],
        "num_frames": len(seq),
        "frame_gap": gap,
        "seed": seed,
        "settings": {
            "ratio_threshold": ratio,
            "ransac_prob": prob,
            "ransac_threshold_px": thresh,
            "max_features": max_features,
            "min_parallax_deg": 0.5,
        },
        "sequence_load_ms": load_ms,
        "headline_pair": headline,
        "aggregate_over_pairs": {
            "num_pairs": len(ok_pairs),
            "num_pairs_attempted": len(per_pair),
            "n_matches_median": med("num_matches"),
            "n_inliers_median": med("num_inliers"),
            "inlier_ratio_median": med("inlier_ratio"),
            "reprojection_median_px": med_nested("reprojection_error_px", "median"),
            "reprojection_mean_px": med_nested("reprojection_error_px", "mean"),
            "reprojection_rms_px": med_nested("reprojection_error_px", "rms"),
            "rotation_error_deg_median": med("rotation_error_deg"),
            "translation_angle_error_deg_median": med("translation_angle_error_deg"),
            "scale_from_gt_median": med_nested("scale_from_gt", "scale"),
            "true_baseline_m_median": med("true_baseline_m"),
            "runtime_ms_median": med("runtime_ms"),
        },
        "pairs": per_pair,
    }
    out = save_json(getattr(args, "out", "results/two_view_baseline.json"), payload)

    agg = payload["aggregate_over_pairs"]
    print(
        f"[two-view] {seq.name} ({seq.source}) {seq.height}x{seq.width}, "
        f"{agg['num_pairs']}/{agg['num_pairs_attempted']} usable pairs, gap={gap}"
    )
    print(
        f"[two-view]   matches {agg['n_matches_median']:.0f}, inliers {agg['n_inliers_median']:.0f}, "
        f"inlier ratio {agg['inlier_ratio_median']:.3f}"
    )
    print(
        f"[two-view]   reprojection median {agg['reprojection_median_px']:.3f} px, "
        f"RMS {agg['reprojection_rms_px']:.3f} px"
    )
    print(
        f"[two-view]   rotation error {agg['rotation_error_deg_median']:.3f} deg, "
        f"translation angle error {agg['translation_angle_error_deg_median']:.3f} deg"
    )
    print(
        f"[two-view]   recovered scale {agg['scale_from_gt_median']:.3f} "
        f"(true baseline {agg['true_baseline_m_median']:.3f} m => scale IS the baseline)"
    )
    print(f"[two-view]   runtime {agg['runtime_ms_median']:.1f} ms/pair")
    print(f"[two-view] wrote {out}")
    return 0
