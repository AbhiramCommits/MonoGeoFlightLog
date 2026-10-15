"""Deterministic synthetic flight-log renderer.

The renderer exists so that the whole repository runs offline on CPU while still
producing *exactly* known ground truth: a procedural texture is baked onto a
dense 3D point cloud (a noisy ground plane plus a handful of boxes with known
heights), the cloud is projected with the pinhole model ``x = K [R|t] X`` and
splatted with a z-buffer, and radial-tangential distortion is applied with the
same forward model that :func:`cv2.undistort` inverts.

Conventions
-----------
* World frame: ``+X`` right, ``+Y`` **up**, ``+Z`` forward. The ground plane is
  ``Y = 0`` and the aircraft flies above it.
* Camera frame follows OpenCV: ``+X`` right, ``+Y`` **down**, ``+Z`` forward.
  Poses are 4x4 ``world -> camera`` matrices, so ``X_cam = R @ X_world + t``.
* Distortion coefficients are ordered ``[k1, k2, p1, p2, k3]`` (OpenCV order).
* Ground-truth depth maps store camera-frame ``Z`` in metres (Monodepth2
  convention), with ``0`` marking pixels that received no point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence as TypingSequence

import numpy as np

__all__ = [
    "Scene",
    "make_scene",
    "make_trajectory",
    "distort_normalized",
    "project_points",
    "render_frame",
    "render_sequence",
    "render_checkerboard_views",
    "default_intrinsics",
    "default_distortion",
    "DEFAULT_DIST",
]

#: A realistic-but-recoverable radial-tangential distortion vector, OpenCV order.
DEFAULT_DIST = np.array([-0.12, 0.03, 0.0012, -0.0008, 0.0], dtype=np.float64)


# --------------------------------------------------------------------------- #
# noise helpers
# --------------------------------------------------------------------------- #
def _value_noise(shape: tuple[int, int], rng: np.random.Generator, res: int) -> np.ndarray:
    """Bilinearly-upsampled uniform noise on a ``res x res`` lattice.

    Args:
        shape: ``(height, width)`` of the output field.
        rng: Seeded generator; consumption order defines determinism.
        res: Lattice resolution in samples per axis.

    Returns:
        Float32 array of ``shape`` with values in ``[0, 1]``.
    """
    from scipy.ndimage import zoom

    h, w = shape
    res = max(2, int(res))
    grid = rng.random((res, res)).astype(np.float32)
    up = zoom(grid, (h / res, w / res), order=1)
    out = np.zeros((h, w), np.float32)
    hh = min(h, up.shape[0])
    ww = min(w, up.shape[1])
    out[:hh, :ww] = up[:hh, :ww]
    if hh < h:
        out[hh:, :] = out[hh - 1 : hh, :]
    if ww < w:
        out[:, ww:] = out[:, ww - 1 : ww]
    return np.clip(out, 0.0, 1.0)


def _fractal_noise(
    shape: tuple[int, int],
    rng: np.random.Generator,
    octaves: int = 5,
    base: int = 4,
    persistence: float = 0.55,
) -> np.ndarray:
    """Sum of ``octaves`` value-noise octaves, normalised to ``[0, 1]``."""
    acc = np.zeros(shape, np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        acc += amp * _value_noise(shape, rng, base * 2**o)
        total += amp
        amp *= persistence
    return np.clip(acc / total, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# scene construction
# --------------------------------------------------------------------------- #
@dataclass
class Scene:
    """A textured 3D point cloud with exactly known camera-frame geometry.

    Attributes:
        points: ``(N, 3)`` float32 world coordinates in metres.
        colors: ``(N, 3)`` float32 linear RGB in ``[0, 1]``.
        seed: Seed the scene was generated from.
        meta: Free-form provenance (box heights, ground extent, ...).
    """

    points: np.ndarray
    colors: np.ndarray
    seed: int
    meta: dict

    def __len__(self) -> int:
        return int(self.points.shape[0])


def _palette(t: np.ndarray, tint: np.ndarray) -> np.ndarray:
    """Map a scalar field ``t`` in ``[0, 1]`` to RGB multiplied by ``tint``."""
    t = np.clip(t, 0.0, 1.0)
    base = np.stack(
        [
            0.25 + 0.75 * t,
            0.30 + 0.65 * (1.0 - np.abs(t - 0.5) * 2.0),
            0.20 + 0.70 * (1.0 - t),
        ],
        axis=-1,
    )
    return np.clip(base * tint[None, :], 0.0, 1.0).astype(np.float32)


def _sample_box(
    center: np.ndarray,
    size: np.ndarray,
    samples_per_meter: float = 3.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the surfaces of an axis-aligned box on a regular grid.

    Args:
        center: ``(3,)`` box centre in world coordinates.
        size: ``(3,)`` full side lengths.
        samples_per_meter: Grid density along each axis.
        rng: Optional generator used for per-face texture variation.

    Returns:
        ``(points, gray)`` where ``points`` is ``(M, 3)`` and ``gray`` is
        ``(M,)`` a texture value in ``[0, 1]``.
    """
    rng = rng or np.random.default_rng(0)
    hx, hy, hz = size / 2.0
    pts: list[np.ndarray] = []
    tex: list[np.ndarray] = []

    def _grid(u: int, v: int) -> tuple[np.ndarray, np.ndarray]:
        a = np.linspace(-1.0, 1.0, max(2, u))
        b = np.linspace(-1.0, 1.0, max(2, v))
        return np.meshgrid(a, b, indexing="ij")

    nu = [max(2, int(s * samples_per_meter)) for s in size]

    # top face (y = +hy)
    u, v = _grid(nu[0], nu[2])
    pts.append(np.stack([center[0] + u.ravel() * hx, np.full(u.size, center[1] + hy), center[2] + v.ravel() * hz], 1))
    tex.append(_value_noise((nu[0], nu[2]), rng, 6).ravel())

    for axis, sign in ((0, +1), (0, -1), (2, +1), (2, -1)):
        u, v = _grid(nu[0] if axis == 2 else nu[2], nu[1])
        if axis == 0:
            x = np.full(u.size, center[0] + sign * hx)
            y = center[1] + u.ravel() * hy
            z = center[2] + v.ravel() * hz
        else:
            x = center[0] + u.ravel() * hx
            y = center[1] + u.ravel() * hy
            z = np.full(u.size, center[2] + sign * hz)
        pts.append(np.stack([x, y, z], 1))
        tex.append(_value_noise((u.shape[0], u.shape[1]), rng, 6).ravel())

    points = np.concatenate(pts, 0).astype(np.float32)
    gray = np.concatenate(tex, 0).astype(np.float32)
    return points, gray


def make_scene(seed: int = 0, n_boxes: int = 6) -> Scene:
    """Build a deterministic textured height-field scene.

    The scene is a fractal-noise ground plane (amplitude ~2 m, so it is a real
    height field rather than a flat card) plus ``n_boxes`` axis-aligned
    buildings with known heights. Texture frequency is deliberately high so
    SIFT/ORB find plenty of stable corners for the classical two-view baseline.

    Args:
        seed: Seed for the procedural generator. Identical seeds give identical
            point clouds bit-for-bit.
        n_boxes: Number of boxes to place ahead of the camera.

    Returns:
        A :class:`Scene` whose ``points``/``colors`` fully define the geometry.
    """
    rng = np.random.default_rng(seed)

    # --- textured ground ---------------------------------------------------
    # The ground is sampled on a *camera-aware fan*: depth is spaced
    # quadratically (dense near the aircraft) and lateral extent grows with
    # depth so that angular sampling density stays roughly constant. This keeps
    # ground-truth depth maps nearly hole-free at small resolutions without
    # needing millions of points.
    nu, nz = 620, 720
    u = np.linspace(-1.0, 1.0, nu, dtype=np.float32)
    zt = np.linspace(0.0, 1.0, nz, dtype=np.float32) ** 2
    z = (1.5 + (180.0 - 1.5) * zt).astype(np.float32)
    UU, ZZ = np.meshgrid(u, z, indexing="ij")
    half_width = np.maximum(0.85 * ZZ, 32.0).astype(np.float32)
    XX = (UU * half_width).astype(np.float32)
    shape = XX.shape

    # Height field in *world* space: fractal noise sampled on the ground grid.
    height = 2.2 * (_fractal_noise(shape, rng, octaves=5, base=3) - 0.5)
    # Fade the relief out far away so the horizon stays tidy.
    height *= np.exp(-((ZZ - 40.0) / 90.0) ** 2).astype(np.float32)
    ground = np.stack([XX.ravel(), height.ravel(), ZZ.ravel()], 1).astype(np.float32)

    tex = _fractal_noise(shape, rng, octaves=5, base=2, persistence=0.6)
    # Directional ripples add mid-frequency structure.
    ripple = 0.15 * np.sin(0.9 * XX + 0.35 * ZZ) * np.sin(0.31 * ZZ - 0.17 * XX)
    tex = np.clip(tex + ripple.astype(np.float32), 0.0, 1.0)

    # --- painted ground-control markers -----------------------------------
    # The pure fractal texture is not repeatable enough across viewpoints for
    # SIFT: descriptors change with scale and the resulting matches are weak.
    # Painting high-contrast concentric targets at deterministic world
    # positions (the aerial-imagery analogue of painted GCPs) gives stable,
    # repeatable corners, which is what a classical baseline actually needs.
    from scipy.spatial import cKDTree

    marker_rng = np.random.default_rng(seed + 991)
    n_markers = int(marker_rng.integers(90, 140))
    marker_xz = np.column_stack(
        [
            marker_rng.uniform(-58.0, 58.0, n_markers),
            marker_rng.uniform(4.0, 150.0, n_markers),
        ]
    )
    tree = cKDTree(marker_xz)
    dist, _ = tree.query(np.column_stack([XX.ravel(), ZZ.ravel()]))
    tex_flat = tex.ravel()
    core = dist < 0.55
    ring = (dist >= 0.55) & (dist < 1.25)
    tex_flat[core] = 0.99
    tex_flat[ring] = 0.03
    tex = tex_flat.reshape(shape)

    tint = np.array([1.0, 0.98, 0.92], np.float32)
    colors = _palette(tex.ravel(), tint)

    all_pts = [ground]
    all_cols = [colors]

    # --- boxes with exactly known heights ---------------------------------
    box_meta = []
    h_mid = height[nu // 2, :]
    for i in range(n_boxes):
        lateral = float(rng.uniform(-26.0, 26.0))
        forward = float(rng.uniform(14.0, 110.0))
        wx = float(rng.uniform(4.0, 11.0))
        wz = float(rng.uniform(4.0, 11.0))
        hgt = float(rng.uniform(3.0, 12.0))
        # Sink the box slightly into the terrain so it never floats.
        base = float(np.interp(forward, z, h_mid)) - 0.4
        center = np.array([lateral, base + hgt / 2.0, forward], np.float32)
        size = np.array([wx, hgt, wz], np.float32)
        pts, gray = _sample_box(center, size, samples_per_meter=6.0, rng=rng)
        tint = np.array(
            [
                float(rng.uniform(0.55, 1.0)),
                float(rng.uniform(0.45, 0.95)),
                float(rng.uniform(0.40, 0.90)),
            ],
            np.float32,
        )
        all_pts.append(pts)
        all_cols.append(_palette(gray, tint))
        box_meta.append(
            {
                "center": center.tolist(),
                "size": size.tolist(),
                "height": hgt,
                "base": base,
            }
        )

    points = np.concatenate(all_pts, 0)
    colors = np.concatenate(all_cols, 0)

    # --- simple lambertian shading from a fixed sun -------------------------
    sun = np.array([0.35, 0.86, -0.37], np.float32)
    sun /= np.linalg.norm(sun)
    # Cheap normal proxy: how "up" the local neighbourhood is, from an
    # approximate gradient of the point cloud is expensive, so shade the
    # ground with the height-field gradient and the boxes with their face
    # direction (encoded in the palette tint already). Good enough visually.
    gy, gx_ = np.gradient(height.astype(np.float32))
    n0 = np.stack([-gx_.ravel(), np.ones(gx_.size, np.float32), -gy.ravel()], 1)
    n0 /= np.linalg.norm(n0, axis=1, keepdims=True)
    shade = np.clip(n0 @ sun, 0.0, 1.0)
    # normalize to mean ~1 so the ground is neither blown out nor black
    shade = 0.45 + 0.55 * shade
    colors[: ground.shape[0]] *= shade[:, None]

    meta = {
        "boxes": box_meta,
        "ground_extent": {"x": [-60.0, 60.0], "z": [1.5, 180.0]},
        "height_amplitude": 2.2,
    }
    return Scene(points=points, colors=colors, seed=seed, meta=meta)


# --------------------------------------------------------------------------- #
# camera model
# --------------------------------------------------------------------------- #
def default_intrinsics(width: int, height: int, fov_deg: float = 62.0) -> np.ndarray:
    """Pinhole intrinsics with a centred principal point.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        fov_deg: Horizontal field of view in degrees.

    Returns:
        ``(3, 3)`` float64 intrinsic matrix.
    """
    f = (width / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    return np.array(
        [[f, 0.0, (width - 1) / 2.0], [0.0, f, (height - 1) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def distort_normalized(xy: np.ndarray, dist: TypingSequence[float]) -> np.ndarray:
    """Apply the OpenCV radial-tangential distortion model to normalised coords.

    Implements exactly the forward model that :func:`cv2.undistort` inverts:

    ``x_d = x(1 + k1 r^2 + k2 r^4 + k3 r^6) + 2 p1 x y + p2 (r^2 + 2 x^2)``

    Args:
        xy: ``(N, 2)`` normalised (``X/Z``, ``Y/Z``) coordinates.
        dist: Distortion vector in OpenCV order ``[k1, k2, p1, p2, k3]``.

    Returns:
        ``(N, 2)`` distorted normalised coordinates.
    """
    d = np.zeros(5, np.float64)
    d[: min(5, len(dist))] = np.asarray(dist, np.float64)[:5]
    k1, k2, p1, p2, k3 = d
    x = xy[:, 0].astype(np.float64)
    y = xy[:, 1].astype(np.float64)
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return np.stack([xd, yd], axis=1)


def _rotation_from_lookat(position: np.ndarray, target: np.ndarray, world_up: np.ndarray | None = None) -> np.ndarray:
    """World->camera rotation for a camera at ``position`` looking at ``target``.

    Builds a *proper* (det ``+1``) rotation for the OpenCV camera convention
    (``+X`` right, ``+Y`` down, ``+Z`` forward). Note that looking along a world
    ``+Z`` axis in a world where ``+Y`` is up puts world ``+X`` on the *left* of
    the image; getting this sign wrong yields a mirrored image and a
    determinant of ``-1``, which silently breaks essential-matrix pose recovery.
    """
    up = np.array([0.0, 1.0, 0.0]) if world_up is None else np.asarray(world_up, np.float64)
    z_c = target - position
    n = np.linalg.norm(z_c)
    if n < 1e-9:
        raise ValueError("camera position coincides with its look-at target")
    z_c = z_c / n
    # image-right axis, then complete the right-handed frame with x cross y = z
    x_c = np.cross(z_c, up)
    nx = np.linalg.norm(x_c)
    if nx < 1e-9:  # looking straight up/down: pick any consistent right vector
        x_c = np.array([1.0, 0.0, 0.0])
    else:
        x_c = x_c / nx
    y_c = np.cross(z_c, x_c)
    return np.stack([x_c, y_c, z_c], axis=0)


def lookat_pose(position: np.ndarray, target: np.ndarray, roll_deg: float = 0.0) -> np.ndarray:
    """Build a 4x4 world->camera pose looking from ``position`` to ``target``."""
    R = _rotation_from_lookat(np.asarray(position, np.float64), np.asarray(target, np.float64))
    if abs(roll_deg) > 1e-12:
        a = np.deg2rad(roll_deg)
        Rz = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
        R = Rz @ R
    t = -R @ np.asarray(position, np.float64)
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose


def project_points(
    points: np.ndarray,
    pose: np.ndarray,
    K: np.ndarray,
    dist: TypingSequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points into an image with distortion.

    Args:
        points: ``(N, 3)`` world points.
        pose: ``(4, 4)`` world->camera transform.
        K: ``(3, 3)`` intrinsics.
        dist: Distortion coefficients, OpenCV order.

    Returns:
        ``(pix, z, valid)`` where ``pix`` is ``(N, 2)`` float64 pixel
        coordinates (continuous, *before* rounding), ``z`` is the camera-frame
        depth of every point, and ``valid`` is a boolean mask for points in
        front of the camera.
    """
    P = np.asarray(points, np.float64)
    R = np.asarray(pose, np.float64)[:3, :3]
    t = np.asarray(pose, np.float64)[:3, 3]
    cam = P @ R.T + t
    z = cam[:, 2]
    valid = z > 1e-3
    zz = np.where(valid, z, 1.0)
    xy = cam[:, :2] / zz[:, None]
    xyd = distort_normalized(xy, dist)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    pix = np.stack([fx * xyd[:, 0] + cx, fy * xyd[:, 1] + cy], axis=1)
    return pix, z, valid


def render_frame(
    scene: Scene,
    pose: np.ndarray,
    K: np.ndarray,
    dist: TypingSequence[float],
    size: tuple[int, int],
    splat: int = 2,
    z_max: float = 400.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splat a scene into one frame with a lexicographic z-buffer.

    Every point is expanded to an ``splat x splat`` footprint and *all*
    resulting samples are sorted by ``(pixel, depth)`` with :func:`numpy.lexsort`;
    the first entry of each pixel group is therefore the nearest sample. This
    gives a correct z-buffer, which matters: a naive per-offset painter's
    algorithm lets a far sample overwrite a nearer one and silently corrupts
    the ground-truth depth used to train the learned models.

    Args:
        scene: Textured point cloud.
        pose: ``(4, 4)`` world->camera transform.
        K: ``(3, 3)`` intrinsics.
        dist: Distortion coefficients.
        size: ``(height, width)`` output resolution.
        splat: Side length in pixels of the square footprint of each point.
        z_max: Points further than this are dropped.

    Returns:
        ``(image, depth, pix, visible)``: uint8 ``(H, W, 3)`` frame, float32
        ``(H, W)`` depth (0 where no point landed), the ``(M, 2)`` continuous
        pixel coordinates of every kept point, and the ``(M,)`` boolean mask
        indexing back into ``scene.points``.
    """
    height, width = int(size[0]), int(size[1])
    pix, z, valid = project_points(scene.points, pose, K, dist)
    keep = valid & (z < z_max)
    keep &= pix[:, 0] >= 0
    keep &= pix[:, 1] >= 0
    keep &= pix[:, 0] <= width - 1
    keep &= pix[:, 1] <= height - 1

    p = pix[keep]
    zz = z[keep]
    cols = scene.colors[keep]

    image = np.zeros((height, width, 3), np.uint8)
    depth = np.zeros((height, width), np.float32)
    if p.shape[0] == 0:
        return image, depth, p, keep

    base_x = np.rint(p[:, 0]).astype(np.int64)
    base_y = np.rint(p[:, 1]).astype(np.int64)

    offs = np.arange(splat) - (splat - 1) // 2
    flats, zs, src = [], [], []
    for dy in offs:
        for dx in offs:
            yy = base_y + dy
            xx = base_x + dx
            ok = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
            if not np.any(ok):
                continue
            idx = np.nonzero(ok)[0]
            flats.append(yy[idx] * width + xx[idx])
            zs.append(zz[idx].astype(np.float32))
            src.append(idx)

    flat = np.concatenate(flats)
    zall = np.concatenate(zs)
    sall = np.concatenate(src)

    # primary key: pixel index; secondary key: depth ascending -> nearest wins
    order = np.lexsort((zall, flat))
    fs = flat[order]
    first = np.empty(fs.shape[0], bool)
    first[0] = True
    np.not_equal(fs[1:], fs[:-1], out=first[1:])
    sel = order[first]

    image.reshape(-1, 3)[flat[sel]] = (cols[sall[sel]] * 255.0 + 0.5).astype(np.uint8)
    depth.reshape(-1)[flat[sel]] = zall[sel]

    return image, depth, p, keep


def render_sequence(
    scene: Scene,
    poses: np.ndarray,
    K: np.ndarray,
    dist: TypingSequence[float],
    size: tuple[int, int],
    splat: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render a whole trajectory.

    Args:
        scene: Textured point cloud.
        poses: ``(T, 4, 4)`` world->camera transforms.
        K: ``(3, 3)`` intrinsics.
        dist: Distortion coefficients.
        size: ``(height, width)`` output resolution.
        splat: Point footprint side length in pixels.

    Returns:
        ``(frames, depths, poses)`` with shapes ``(T, H, W, 3)`` uint8,
        ``(T, H, W)`` float32 and ``(T, 4, 4)`` float32.
    """
    poses = np.asarray(poses, np.float64)
    frames, depths = [], []
    for pose in poses:
        img, dep, _, _ = render_frame(scene, pose, K, dist, size, splat=splat)
        frames.append(img)
        depths.append(dep)
    return (
        np.stack(frames, 0),
        np.stack(depths, 0).astype(np.float32),
        poses.astype(np.float32),
    )


# --------------------------------------------------------------------------- #
# trajectories
# --------------------------------------------------------------------------- #
def make_trajectory(
    kind: str = "forward",
    n: int = 40,
    altitude: float = 9.0,
    speed: float = 1.6,
) -> np.ndarray:
    """Generate a smooth 4x4 world->camera trajectory.

    The camera is pitched down by 15-25 degrees (a typical oblique aerial /
    dashcam view) so that the ground fills most of the frame and lands at a
    depth where consecutive frames have useful parallax.

    Args:
        kind: One of ``"forward"`` (straight flight with a gentle sway),
            ``"orbit"`` (circular flight around a ground point) or
            ``"descent"`` (forward flight while losing altitude, used later as
            the altitude / domain-shift sequence).
        n: Number of frames.
        altitude: Starting height above the ground plane, in metres.
        speed: Forward speed in metres per frame.

    Returns:
        ``(n, 4, 4)`` float64 array of world->camera matrices.

    Raises:
        ValueError: If ``kind`` is unknown.
    """
    n = int(n)
    poses = np.zeros((n, 4, 4), np.float64)
    ts = np.arange(n, dtype=np.float64)
    alt0 = float(altitude)

    if kind == "forward":
        for i, t in enumerate(ts):
            pos = np.array(
                [
                    1.8 * np.sin(0.28 * t),
                    alt0 + 0.6 * np.sin(0.21 * t + 0.6),
                    4.0 + speed * t,
                ]
            )
            target = pos + np.array([0.0, -3.1, 11.0])
            poses[i] = lookat_pose(pos, target)
    elif kind == "orbit":
        center = np.array([0.0, 3.0, 34.0])
        radius = 20.0
        for i, t in enumerate(ts):
            ang = 0.16 * t
            pos = center + np.array(
                [radius * np.sin(ang), alt0 + 3.0 + 2.0 * np.cos(0.3 * t), radius * np.cos(ang)]
            )
            poses[i] = lookat_pose(pos, center)
    elif kind == "descent":
        for i, t in enumerate(ts):
            pos = np.array(
                [
                    0.5 * np.sin(0.22 * t),
                    alt0 + 7.0 - 0.28 * t,
                    4.0 + 0.9 * speed * t,
                ]
            )
            # pitch the nose further down as the aircraft descends
            target = pos + np.array([0.0, -2.6 - 0.05 * t, 11.0])
            poses[i] = lookat_pose(pos, target)
    else:
        raise ValueError(f"unknown trajectory kind {kind!r}; expected forward|orbit|descent")

    return poses


# --------------------------------------------------------------------------- #
# calibration targets
# --------------------------------------------------------------------------- #
def make_checkerboard_points(
    rows: int = 6,
    cols: int = 7,
    square_size: float = 0.04,
    samples: int = 14,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a planar checkerboard as a dense point cloud.

    The board is built in *corner coordinates*: inner corner ``(i, j)`` sits at
    ``(i * square_size, j * square_size, 0)``, the pattern occupies
    ``x in [0, cols * square_size]`` and ``y in [0, rows * square_size]``, and a
    one-square **white margin** surrounds the pattern so that
    :func:`cv2.findChessboardCorners` has dark/light quads to lock onto at the
    border.

    Args:
        rows: Number of *inner* corners along the board's vertical axis.
        cols: Number of *inner* corners along the board's horizontal axis.
        square_size: Physical square edge length in metres.
        samples: Sub-samples per square edge (drives texture density).

    Returns:
        ``(points, colors)`` where ``points`` is ``(M, 3)`` float32 on the
        ``Z = 0`` plane and ``colors`` is ``(M, 3)`` float32 greyscale RGB.
    """
    # An ``cols x rows`` inner-corner pattern needs ``(cols+1) x (rows+1)``
    # squares, plus a one-square white margin ring on every side.
    xs = (np.arange(-samples, (cols + 2) * samples + 1) / samples) * square_size
    ys = (np.arange(-samples, (rows + 2) * samples + 1) / samples) * square_size
    XX, YY = np.meshgrid(xs, ys, indexing="xy")

    si = np.floor(XX / square_size + 1e-9).astype(np.int64)
    sj = np.floor(YY / square_size + 1e-9).astype(np.int64)
    in_pattern = (si >= 0) & (si <= cols) & (sj >= 0) & (sj <= rows)
    white = np.where(in_pattern, ((si + sj) % 2) == 0, True)
    val = np.where(white, 0.93, 0.05).astype(np.float32)
    # Slight per-square exposure jitter (deterministic, no aliasing moire).
    jitter = 0.015 * (((np.abs(si) * 7 + np.abs(sj) * 13) % 5) / 4.0 - 0.5)
    val = np.clip(val + jitter, 0.0, 1.0).astype(np.float32)

    pts = np.stack([XX.ravel(), YY.ravel(), np.zeros(XX.size)], 1).astype(np.float32)
    rgb = np.stack([val, val, val], -1).reshape(-1, 3).astype(np.float32)
    return pts, rgb


def render_checkerboard_views(
    K: np.ndarray,
    dist: TypingSequence[float],
    n: int = 15,
    size: tuple[int, int] = (480, 640),
    rows: int = 6,
    cols: int = 7,
    square_size: float = 0.04,
    seed: int = 0,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Render ``n`` views of a known-geometry checkerboard for calibration.

    The board is tilted and shifted with a seeded random walk so that
    ``cv2.calibrateCamera`` sees variation in plane orientation.

    Args:
        K: ``(3, 3)`` ground-truth intrinsics.
        dist: Ground-truth distortion coefficients.
        n: Number of views.
        size: ``(height, width)`` of each view.
        rows: Inner-corner rows of the board.
        cols: Inner-corner columns of the board.
        square_size: Physical square size in metres.
        seed: Seed for the view sampling.

    Returns:
        ``(images, objp, board_centers)``: a list of ``n`` uint8 images, the
        ``(rows*cols, 3)`` canonical object points (OpenCV convention, ``Z=0``),
        and the ``(n, 3)`` board centre used for each view.
    """
    rng = np.random.default_rng(seed)
    pts, cols_rgb = make_checkerboard_points(rows, cols, square_size, samples=16)
    # Re-centre the board on the pattern centre so tilting rotates about it.
    mid = np.array([(cols + 1) * square_size / 2.0, (rows + 1) * square_size / 2.0, 0.0])
    pts_c = (pts - mid).astype(np.float64)

    images: list[np.ndarray] = []
    centers = np.zeros((n, 3), np.float64)
    for i in range(n):
        board_dist = float(rng.uniform(0.42, 0.85))
        # real tilt in all three axes, drifting across views
        ang_x = float(rng.uniform(-0.45, 0.45)) + 0.055 * i
        ang_y = float(rng.uniform(-0.5, 0.5)) - 0.045 * i
        ang_z = float(rng.uniform(-0.3, 0.3))
        # camera sits on the -Z side looking down +Z at the board at Z = 0
        pos = np.array(
            [
                float(rng.uniform(-0.10, 0.10)),
                float(rng.uniform(-0.08, 0.08)),
                -board_dist,
            ]
        )
        pose = lookat_pose(pos, pos + np.array([0.0, 0.0, 1.0]))

        # roll the camera a little
        Rz = np.array(
            [
                [np.cos(ang_z), -np.sin(ang_z), 0.0],
                [np.sin(ang_z), np.cos(ang_z), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        pose[:3, :3] = Rz @ pose[:3, :3]
        pose[:3, 3] = -pose[:3, :3] @ pos

        # tilt the *board* (pitch then yaw) about its centre
        Rx = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(ang_x), -np.sin(ang_x)],
                [0.0, np.sin(ang_x), np.cos(ang_x)],
            ]
        )
        Ry = np.array(
            [
                [np.cos(ang_y), 0.0, np.sin(ang_y)],
                [0.0, 1.0, 0.0],
                [-np.sin(ang_y), 0.0, np.cos(ang_y)],
            ]
        )
        tilt_scene = Scene(
            points=(pts_c @ (Ry @ Rx).T).astype(np.float32),
            colors=cols_rgb.astype(np.float32),
            seed=seed,
            meta={},
        )
        img, _, _, _ = render_frame(tilt_scene, pose, K, dist, size, splat=3)
        images.append(img)
        centers[i] = pos

    # canonical object points in the OpenCV calibration convention
    grid = np.zeros((rows * cols, 3), np.float32)
    gi, gj = np.meshgrid(np.arange(cols), np.arange(rows))
    grid[:, 0] = gi.ravel() * square_size
    grid[:, 1] = gj.ravel() * square_size
    return images, grid, centers


def frame_shape(size: tuple[int, int]) -> TypingSequence[int]:
    """Return ``(height, width, 3)`` for a given ``(height, width)`` size."""
    return (int(size[0]), int(size[1]), 3)


def iter_pairs(n: int) -> Iterable[tuple[int, int, int]]:
    """Yield ``(target, source_prev, source_next)`` index triplets for ``n`` frames."""
    for i in range(1, n - 1):
        yield i, i - 1, i + 1
