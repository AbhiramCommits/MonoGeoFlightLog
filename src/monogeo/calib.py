"""Camera calibration from checkerboard views.

Classical, explicitly-written pipeline: detect inner corners, refine them to
sub-pixel accuracy, run :func:`cv2.calibrateCamera`, then report how far the
recovered intrinsics and distortion coefficients are from the known ground
truth of the synthetic rig.

The report deliberately includes *two* distortion metrics. Individual radial
coefficients ``k1``/``k2`` are strongly correlated when the calibration target
is small relative to the field of view, so a large per-coefficient error can
coexist with an essentially perfect distortion field. Reporting the maximum
pixel displacement of the distortion field separates "the calibration is wrong"
from "the parameters are not identifiable".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from . import synth
from .cli import cfg_get, save_json

__all__ = [
    "CalibrationResult",
    "calibrate_from_checkerboards",
    "calibration_report",
    "distortion_displacement_px",
    "run_calibrate",
]


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
@dataclass
class CalibrationResult:
    """Outcome of a checkerboard calibration.

    Attributes:
        K: ``(3, 3)`` estimated intrinsics.
        dist: ``(5,)`` estimated distortion coefficients (OpenCV order).
        rms: Overall RMS reprojection error in pixels returned by OpenCV.
        per_view_rms: ``(n_views,)`` RMS reprojection error of each accepted view.
        image_size: ``(width, height)`` of the calibration images.
        pattern: ``(rows, cols)`` of inner corners.
        square_size: Physical square size in metres.
        num_views: Number of views that produced a detectable board.
        num_detected: Number of input views offered to the detector.
        used_view_indices: Indices of the accepted views within the input list.
    """

    K: np.ndarray
    dist: np.ndarray
    rms: float
    per_view_rms: np.ndarray
    image_size: tuple[int, int]
    pattern: tuple[int, int]
    square_size: float
    num_views: int
    num_detected: int
    used_view_indices: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary."""
        return {
            "K": np.asarray(self.K, float).tolist(),
            "dist": np.asarray(self.dist, float).tolist(),
            "rms_reprojection_px": float(self.rms),
            "per_view_rms_px": np.asarray(self.per_view_rms, float).tolist(),
            "image_size_wh": [int(self.image_size[0]), int(self.image_size[1])],
            "pattern_rows_cols": [int(self.pattern[0]), int(self.pattern[1])],
            "square_size_m": float(self.square_size),
            "num_views_used": int(self.num_views),
            "num_views_offered": int(self.num_detected),
        }


def _object_grid(rows: int, cols: int, square_size: float) -> np.ndarray:
    """Canonical object points, OpenCV calibration convention (``Z = 0``)."""
    grid = np.zeros((rows * cols, 3), np.float32)
    gi, gj = np.meshgrid(np.arange(cols), np.arange(rows))
    grid[:, 0] = gi.ravel() * square_size
    grid[:, 1] = gj.ravel() * square_size
    return grid


def calibrate_from_checkerboards(
    images: Sequence[np.ndarray],
    pattern: tuple[int, int] = (6, 7),
    square_size: float = 0.04,
    refine_subpix: bool = True,
    flags: int = 0,
) -> CalibrationResult:
    """Calibrate intrinsics and distortion from checkerboard views.

    Args:
        images: Colour (BGR or RGB) or greyscale images of a checkerboard.
        pattern: ``(rows, cols)`` number of **inner** corners.
        square_size: Physical square size in metres.
        refine_subpix: Run ``cv2.cornerSubPix`` on detected corners.
        flags: Extra flags forwarded to :func:`cv2.calibrateCamera`.

    Returns:
        A :class:`CalibrationResult`.

    Raises:
        ValueError: If fewer than three views are usable, which makes the
            calibration under-constrained.
    """
    rows, cols = int(pattern[0]), int(pattern[1])
    objp = _object_grid(rows, cols, square_size)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    detect_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    used: list[int] = []
    size: tuple[int, int] | None = None

    for i, img in enumerate(images):
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if size is None:
            size = (int(gray.shape[1]), int(gray.shape[0]))
        elif (int(gray.shape[1]), int(gray.shape[0])) != size:
            raise ValueError(f"view {i} has size {gray.shape} but earlier views are {size}")

        found, corners = cv2.findChessboardCorners(gray, (cols, rows), detect_flags)
        if not found:
            continue
        corners = np.asarray(corners, np.float32).reshape(-1, 1, 2)
        if refine_subpix:
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
        used.append(i)

    if len(objpoints) < 3:
        raise ValueError(
            f"only {len(objpoints)} of {len(images)} views produced a detectable "
            f"{rows}x{cols} checkerboard; need at least 3"
        )
    assert size is not None

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, size, None, None, flags=flags
    )

    # Per-view RMS reprojection error, computed explicitly rather than trusting
    # the aggregate OpenCV returns.
    per_view = np.zeros(len(objpoints), np.float64)
    for i, (op, ip, rvec, tvec) in enumerate(zip(objpoints, imgpoints, rvecs, tvecs, strict=False)):
        proj, _ = cv2.projectPoints(op, rvec, tvec, K, dist)
        err = np.linalg.norm(proj.reshape(-1, 2) - ip.reshape(-1, 2), axis=1)
        per_view[i] = float(np.sqrt(np.mean(err**2)))

    dist = np.asarray(dist, np.float64).ravel()
    if dist.size < 5:
        dist = np.concatenate([dist, np.zeros(5 - dist.size)])

    return CalibrationResult(
        K=np.asarray(K, np.float64),
        dist=dist[:5],
        rms=float(rms),
        per_view_rms=per_view,
        image_size=size,
        pattern=(rows, cols),
        square_size=float(square_size),
        num_views=len(objpoints),
        num_detected=len(images),
        used_view_indices=used,
    )


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def distortion_displacement_px(
    K: np.ndarray,
    dist_a: Sequence[float],
    dist_b: Sequence[float],
    image_size: tuple[int, int],
    n: int = 33,
) -> tuple[float, float]:
    """Mean/max pixel displacement between two distortion fields.

    Both models are evaluated over the image's normalized coordinate rectangle
    and the resulting pixel positions are compared. This measures whether the
    *distortion field* was recovered, independently of whether the individual
    coefficients are identifiable.

    Args:
        K: ``(3, 3)`` intrinsics used to convert to pixels.
        dist_a: First distortion vector.
        dist_b: Second distortion vector.
        image_size: ``(width, height)`` in pixels.
        n: Grid resolution per axis.

    Returns:
        ``(mean_px, max_px)`` displacement.
    """
    width, height = int(image_size[0]), int(image_size[1])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xn = np.linspace((0.0 - cx) / fx, (width - 1 - cx) / fx, n)
    yn = np.linspace((0.0 - cy) / fy, (height - 1 - cy) / fy, n)
    XX, YY = np.meshgrid(xn, yn)
    xy = np.stack([XX.ravel(), YY.ravel()], 1)

    da = synth.distort_normalized(xy, dist_a)
    db = synth.distort_normalized(xy, dist_b)
    pa = np.stack([fx * da[:, 0] + cx, fy * da[:, 1] + cy], 1)
    pb = np.stack([fx * db[:, 0] + cx, fy * db[:, 1] + cy], 1)
    d = np.linalg.norm(pa - pb, axis=1)
    return float(d.mean()), float(d.max())


def calibration_report(
    K_est: np.ndarray,
    dist_est: Sequence[float],
    K_true: np.ndarray,
    dist_true: Sequence[float],
    rms: float | None = None,
    image_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Compare estimated calibration parameters against ground truth.

    Args:
        K_est: ``(3, 3)`` estimated intrinsics.
        dist_est: Estimated distortion coefficients.
        K_true: ``(3, 3)`` true intrinsics.
        dist_true: True distortion coefficients.
        rms: Optional RMS reprojection error to include in the report.
        image_size: Optional ``(width, height)``; enables the distortion-field
            displacement metric.

    Returns:
        A JSON-serialisable report with focal/principal-point errors in pixels,
        per-coefficient distortion errors, and (when ``image_size`` is given)
        the mean/max pixel displacement of the distortion field.
    """
    K_est = np.asarray(K_est, np.float64)
    K_true = np.asarray(K_true, np.float64)
    d_est = np.asarray(dist_est, np.float64).ravel()[:5]
    d_true = np.asarray(dist_true, np.float64).ravel()[:5]
    if d_est.size < 5:
        d_est = np.concatenate([d_est, np.zeros(5 - d_est.size)])
    if d_true.size < 5:
        d_true = np.concatenate([d_true, np.zeros(5 - d_true.size)])

    names = ["fx", "fy", "cx", "cy"]
    est = np.array([K_est[0, 0], K_est[1, 1], K_est[0, 2], K_est[1, 2]])
    true = np.array([K_true[0, 0], K_true[1, 1], K_true[0, 2], K_true[1, 2]])

    intrinsics = {
        "estimated": {n: float(v) for n, v in zip(names, est, strict=False)},
        "true": {n: float(v) for n, v in zip(names, true, strict=False)},
        "abs_error_px": {n: float(v) for n, v in zip(names, np.abs(est - true), strict=False)},
        "rel_error_pct": {
            n: float(100.0 * abs(e - t) / t) if t else 0.0
            for n, e, t in zip(names, est, true, strict=False)
        },
    }

    dnames = ["k1", "k2", "p1", "p2", "k3"]
    report: dict[str, Any] = {
        "intrinsics": intrinsics,
        "focal_mean_abs_error_px": float(np.mean(np.abs(est[:2] - true[:2]))),
        "principal_point_mean_abs_error_px": float(np.mean(np.abs(est[2:] - true[2:]))),
        "distortion": {
            "estimated": {n: float(v) for n, v in zip(dnames, d_est, strict=False)},
            "true": {n: float(v) for n, v in zip(dnames, d_true, strict=False)},
            "abs_error": {
                n: float(v) for n, v in zip(dnames, np.abs(d_est - d_true), strict=False)
            },
        },
        "distortion_max_abs_coeff_error": float(np.max(np.abs(d_est - d_true))),
    }
    if rms is not None:
        report["rms_reprojection_px"] = float(rms)
    if image_size is not None:
        mean_px, max_px = distortion_displacement_px(K_est, d_est, d_true, image_size)
        report["distortion_field_displacement_px"] = {
            "mean": mean_px,
            "max": max_px,
        }
        report["distortion_field_note"] = (
            "k1/k2 are strongly correlated for a small planar target near the image "
            "centre; the residual *field* displacement is the honest measure of "
            "whether distortion was recovered."
        )
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def run_calibrate(args: Any) -> int:
    """``monogeo calibrate``: render/target the calibration rig and report.

    Args:
        args: Parsed argparse namespace (``out``, ``views``, ``image_size``,
            ``seed``, ``cfg``, ``verbose``).

    Returns:
        Process exit code (0 on success).
    """
    cfg = getattr(args, "cfg", {}) or {}
    image_size = tuple(
        getattr(args, "image_size", None) or cfg_get(cfg, "calib.image_size", [480, 640])
    )
    height, width = int(image_size[0]), int(image_size[1])
    pattern = tuple(cfg_get(cfg, "calib.checkerboard", [6, 7]))
    rows, cols = int(pattern[0]), int(pattern[1])
    square_size = float(cfg_get(cfg, "calib.square_size", 0.04))
    num_views = int(getattr(args, "views", None) or cfg_get(cfg, "calib.num_views", 20))
    seed = int(getattr(args, "seed", 0) or 0)
    fov = float(cfg_get(cfg, "synth.fov_deg", 62.0))
    dist_true = np.asarray(
        cfg_get(cfg, "synth.distortion", synth.DEFAULT_DIST.tolist()), np.float64
    )

    K_true = synth.default_intrinsics(width, height, fov)
    images, _objp, _ = synth.render_checkerboard_views(
        K_true,
        dist_true,
        n=num_views,
        size=(height, width),
        rows=rows,
        cols=cols,
        square_size=square_size,
        seed=seed,
    )

    result = calibrate_from_checkerboards(images, pattern=(rows, cols), square_size=square_size)
    report = calibration_report(
        result.K, result.dist, K_true, dist_true, rms=result.rms, image_size=(width, height)
    )

    payload = {
        "step": "calibrate",
        "data_source": "synthetic",
        "sequence": None,
        "image_size_wh": [width, height],
        "fov_deg": fov,
        "seed": seed,
        "checkerboard": {
            "inner_corners_rows_cols": [rows, cols],
            "square_size_m": square_size,
            "views_offered": num_views,
            "views_used": result.num_views,
            "used_view_indices": result.used_view_indices,
            "per_view_rms_px": result.per_view_rms.tolist(),
        },
        "report": report,
        "K_estimated": result.K.tolist(),
        "K_true": K_true.tolist(),
        "dist_estimated": result.dist.tolist(),
        "dist_true": dist_true.tolist(),
        "rms_reprojection_px": float(result.rms),
    }
    out = save_json(getattr(args, "out", "results/calibration.json"), payload)

    intr = report["intrinsics"]
    print(f"[calibrate] views used {result.num_views}/{num_views}, RMS {result.rms:.4f} px")
    for name in ("fx", "fy", "cx", "cy"):
        print(
            f"[calibrate]   {name}: est {intr['estimated'][name]:9.3f} "
            f"true {intr['true'][name]:9.3f} "
            f"err {intr['abs_error_px'][name]:7.3f} px "
            f"({intr['rel_error_pct'][name]:5.2f}%)"
        )
    print(f"[calibrate] distortion coeff error {report['distortion']['estimated']}")
    disp = report.get("distortion_field_displacement_px")
    if disp:
        print(
            f"[calibrate] distortion field displacement mean {disp['mean']:.4f} px, max {disp['max']:.4f} px"
        )
    print(f"[calibrate] wrote {out}")
    return 0
