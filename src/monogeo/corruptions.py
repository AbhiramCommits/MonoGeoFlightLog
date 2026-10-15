"""Image corruptions and domain shifts for stress-testing the geometry methods.

Each corruption takes a uint8 RGB frame and a **severity** in ``{1, 2, 3}`` and
returns a new uint8 RGB frame. The severities are chosen to bracket the point
where classical feature matching breaks, not to be photorealistic:

===============  ==========================================================
corruption       what it does
===============  ==========================================================
``motion_blur``  linear blur along a direction, 5/11/19 px -- destroys the
                 high-frequency structure SIFT relies on
``low_texture``  blend towards a blurred copy -- removes the mid-frequency
                 contrast that creates repeatable keypoints
``low_light``    gain + gamma + read noise -- throws away the low-contrast
                 features first, as a real sensor does
``jpeg_artifacts`` extreme JPEG quantisation (q=30/15/7) -- adds 8x8 blocking
                 that competes with the real texture
===============  ==========================================================

A *domain shift* is handled separately: it means evaluating the same model on a
different flight regime (:func:`monogeo.synth.make_trajectory` ``"descent"`` for
an altitude change, ``"orbit"`` for a different viewing geometry) or, when real
data is present, on KITTI/EuRoC.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

__all__ = [
    "CORRUPTIONS",
    "SEVERITIES",
    "apply_corruption",
    "available_corruptions",
    "jpeg_artifacts",
    "low_light",
    "low_texture",
    "motion_blur",
    "motion_blur_kernel",
]

#: Supported severities.
SEVERITIES = (1, 2, 3)


def _as_uint8(img: np.ndarray) -> np.ndarray:
    """Return ``img`` as a contiguous uint8 RGB array."""
    if img.dtype != np.uint8:
        img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def motion_blur_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Build a normalised linear motion-blur kernel.

    Args:
        length: Kernel length in pixels (forced odd and >= 1).
        angle_deg: Blur direction in degrees, measured from the +x axis.

    Returns:
        ``(length, length)`` float32 kernel summing to 1.
    """
    n = max(1, int(length))
    if n % 2 == 0:
        n += 1
    kernel = np.zeros((n, n), np.float32)
    if n == 1:
        kernel[0, 0] = 1.0
        return kernel
    c = (n - 1) / 2.0
    a = np.deg2rad(float(angle_deg))
    dx, dy = np.cos(a), np.sin(a)
    # walk the whole line so the kernel is gap-free for any angle
    for t in np.linspace(-c, c, 4 * n):
        x = int(round(c + t * dx))
        y = int(round(c + t * dy))
        if 0 <= x < n and 0 <= y < n:
            kernel[y, x] = 1.0
    total = kernel.sum()
    return kernel / total if total > 0 else kernel


def motion_blur(img: np.ndarray, severity: int = 1, angle_deg: float = -18.0) -> np.ndarray:
    """Apply linear motion blur.

    Args:
        img: ``(H, W, 3)`` uint8 RGB frame.
        severity: ``1``, ``2`` or ``3`` -> 5, 11 or 19 pixel kernel.
        angle_deg: Blur direction in degrees.

    Returns:
        Blurred frame, same shape and dtype.

    Raises:
        ValueError: If ``severity`` is not 1..3.
    """
    lengths = {1: 5, 2: 11, 3: 19}
    if int(severity) not in lengths:
        raise ValueError(f"severity must be one of {sorted(lengths)}, got {severity}")
    kernel = motion_blur_kernel(lengths[int(severity)], angle_deg)
    out = cv2.filter2D(_as_uint8(img), -1, kernel, borderType=cv2.BORDER_REPLICATE)
    return _as_uint8(out)


def low_texture(img: np.ndarray, severity: int = 1) -> np.ndarray:
    """Suppress texture by blending towards a blurred copy.

    Args:
        img: ``(H, W, 3)`` uint8 RGB frame.
        severity: ``1``, ``2`` or ``3`` -> blend 0.35, 0.65, 0.9 towards the
            blurred version.

    Returns:
        Flattened frame, same shape and dtype.

    Raises:
        ValueError: If ``severity`` is not 1..3.
    """
    blends = {1: 0.35, 2: 0.65, 3: 0.9}
    if int(severity) not in blends:
        raise ValueError(f"severity must be one of {sorted(blends)}, got {severity}")
    src = _as_uint8(img).astype(np.float32)
    blurred = cv2.GaussianBlur(src, (0, 0), sigmaX=2.0 + 2.0 * int(severity))
    w = blends[int(severity)]
    out = (1.0 - w) * src + w * blurred
    # also compress the remaining contrast towards the frame mean
    out = np.clip(out, 0, 255)
    mean = out.mean()
    out = mean + (out - mean) * (1.0 - 0.3 * w)
    return _as_uint8(out)


def low_light(img: np.ndarray, severity: int = 1, seed: int = 0) -> np.ndarray:
    """Simulate a dark, noisy exposure.

    Args:
        img: ``(H, W, 3)`` uint8 RGB frame.
        severity: ``1``, ``2`` or ``3`` -> gain 0.45, 0.25, 0.12 with increasing
            gamma and read noise.
        seed: Seed for the noise so the stress test is deterministic.

    Returns:
        Darkened frame, same shape and dtype.

    Raises:
        ValueError: If ``severity`` is not 1..3.
    """
    gains = {1: 0.45, 2: 0.25, 3: 0.12}
    gammas = {1: 1.2, 2: 1.6, 3: 2.2}
    noise_sigma = {1: 3.0, 2: 8.0, 3: 16.0}
    if int(severity) not in gains:
        raise ValueError(f"severity must be one of {sorted(gains)}, got {severity}")
    s = int(severity)
    src = _as_uint8(img).astype(np.float32) / 255.0
    out = np.power(np.clip(src * gains[s], 0.0, 1.0), gammas[s])
    rng = np.random.default_rng(seed)
    out = out * 255.0 + rng.normal(0.0, noise_sigma[s], out.shape)
    return _as_uint8(out)


def jpeg_artifacts(img: np.ndarray, severity: int = 1) -> np.ndarray:
    """Re-encode the frame as a very low quality JPEG.

    Args:
        img: ``(H, W, 3)`` uint8 RGB frame.
        severity: ``1``, ``2`` or ``3`` -> JPEG quality 30, 15, 7.

    Returns:
        Re-encoded frame, same shape and dtype.

    Raises:
        ValueError: If ``severity`` is not 1..3.
    """
    qualities = {1: 30, 2: 15, 3: 7}
    if int(severity) not in qualities:
        raise ValueError(f"severity must be one of {sorted(qualities)}, got {severity}")
    src = _as_uint8(img)
    ok, buf = cv2.imencode(
        ".jpg",
        cv2.cvtColor(src, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), qualities[int(severity)]],
    )
    if not ok:  # pragma: no cover - encoder failure
        return src
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return _as_uint8(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))


#: Registry of corruption name -> callable ``(img, severity) -> img``.
CORRUPTIONS: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "motion_blur": motion_blur,
    "low_texture": low_texture,
    "low_light": low_light,
    "jpeg_artifacts": jpeg_artifacts,
}


def available_corruptions() -> list[str]:
    """Return the names of all registered corruptions."""
    return sorted(CORRUPTIONS)


def apply_corruption(img: np.ndarray, name: str, severity: int = 1, seed: int = 0) -> np.ndarray:
    """Apply a named corruption at a given severity.

    Args:
        img: ``(H, W, 3)`` uint8 RGB frame.
        name: A key of :data:`CORRUPTIONS`, or ``"clean"``/``"none"`` which are
            no-ops.
        severity: ``1``, ``2`` or ``3``.
        seed: Seed forwarded to corruptions that use randomness.

    Returns:
        The corrupted frame.

    Raises:
        ValueError: If ``name`` is unknown, or ``severity`` is not 1..3.
    """
    key = str(name).lower()
    if key in ("clean", "none", ""):
        return _as_uint8(img)
    if key not in CORRUPTIONS:
        raise ValueError(f"unknown corruption {name!r}; expected one of {available_corruptions()}")
    if int(severity) not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}, got {severity}")
    fn = CORRUPTIONS[key]
    if key == "low_light":
        return fn(img, int(severity), seed=seed)
    return fn(img, int(severity))


def corrupt_frames(frames: np.ndarray, name: str, severity: int = 1, seed: int = 0) -> np.ndarray:
    """Apply a corruption to every frame of a sequence.

    Args:
        frames: ``(T, H, W, 3)`` uint8 RGB frames.
        name: Corruption name.
        severity: ``1``, ``2`` or ``3``.
        seed: Base seed; frame ``i`` uses ``seed + i``.

    Returns:
        ``(T, H, W, 3)`` uint8 RGB frames.
    """
    frames = np.asarray(frames)
    return np.stack(
        [apply_corruption(frames[i], name, severity, seed=seed + i) for i in range(len(frames))],
        0,
    )
