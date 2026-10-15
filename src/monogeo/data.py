"""Data layer: sequences, synthetic fallback, and real-dataset loaders.

The pipeline only ever consumes :class:`Sequence` objects. A sequence carries
frames, ground-truth depth, ground-truth poses, intrinsics and distortion, so
downstream code neither knows nor cares whether the numbers came from the
deterministic synthetic renderer in :mod:`monogeo.synth` or from a real
KITTI odometry / EuRoC recording dropped into ``data/``.

Synthetic sequences are cached to ``data/cache/`` as ``.npz`` files because
rendering is the only slow part of dataset construction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence as TypingSequence

import numpy as np
import torch
from torch.utils.data import Dataset

from . import synth

__all__ = [
    "Sequence",
    "SequenceDataset",
    "load_sequence",
    "SYNTHETIC_KINDS",
    "to_tensor",
]

#: Trajectory kinds produced by the synthetic renderer.
SYNTHETIC_KINDS = ("forward", "orbit", "descent")

#: Default render resolution (height, width) -- kept small so CPU training is
#: minutes rather than hours.
DEFAULT_SIZE = (128, 416)


# --------------------------------------------------------------------------- #
# sequence container
# --------------------------------------------------------------------------- #
@dataclass
class Sequence:
    """A monocular flight log with ground truth.

    Attributes:
        name: Identifier, e.g. ``"forward"`` or ``"kitti:00"``.
        source: Where the data came from: ``"synthetic"``, ``"kitti"`` or
            ``"euroc"``.
        frames: ``(T, H, W, 3)`` uint8 RGB frames.
        depths: ``(T, H, W)`` float32 ground-truth camera-frame depth in metres
            (``0`` where unknown).
        poses: ``(T, 4, 4)`` float32 world->camera transforms.
        K: ``(3, 3)`` float64 intrinsics.
        dist: ``(5,)`` float64 OpenCV-order distortion coefficients.
        meta: Provenance and loader details recorded for the results files.
    """

    name: str
    source: str
    frames: np.ndarray
    depths: np.ndarray
    poses: np.ndarray
    K: np.ndarray
    dist: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    def __getitem__(self, index: int) -> np.ndarray:
        """Return frame ``index`` as ``(H, W, 3)`` uint8."""
        return self.frames[index]

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return int(self.frames.shape[1])

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return int(self.frames.shape[2])

    def dataset(self, **kwargs: Any) -> "SequenceDataset":
        """Wrap this sequence in a :class:`SequenceDataset`."""
        return SequenceDataset(self, **kwargs)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable description used in results files."""
        valid = self.depths > 0
        return {
            "name": self.name,
            "source": self.source,
            "num_frames": len(self),
            "height": self.height,
            "width": self.width,
            "K": np.asarray(self.K, float).tolist(),
            "dist": np.asarray(self.dist, float).tolist(),
            "depth_valid_ratio": float(valid.mean()) if valid.size else 0.0,
            "depth_median_m": float(np.median(self.depths[valid])) if valid.any() else 0.0,
            **({"meta": self.meta} if self.meta else {}),
        }


# --------------------------------------------------------------------------- #
# synthetic sequences
# --------------------------------------------------------------------------- #
def _cache_path(root: Path, name: str, size: tuple[int, int], n: int, seed: int, fov: float) -> Path:
    return root / "cache" / f"synth_{name}_{size[0]}x{size[1]}_n{n}_s{seed}_f{fov:g}.npz"


def _render_synthetic(
    kind: str,
    size: tuple[int, int] = DEFAULT_SIZE,
    n: int = 48,
    seed: int = 0,
    fov_deg: float = 62.0,
    dist: TypingSequence[float] | None = None,
    root: str | os.PathLike[str] = "data",
    cache: bool = True,
    scene_seed: int | None = None,
) -> Sequence:
    """Render a synthetic flight log, optionally caching the result on disk."""
    height, width = int(size[0]), int(size[1])
    dist_vec = np.asarray(synth.DEFAULT_DIST if dist is None else dist, np.float64)
    scene_seed = seed if scene_seed is None else scene_seed

    root_path = Path(root)
    cache_file = _cache_path(root_path, kind, (height, width), n, seed, fov_deg)

    if cache and cache_file.exists():
        with np.load(cache_file) as z:
            return Sequence(
                name=kind,
                source="synthetic",
                frames=z["frames"],
                depths=z["depths"].astype(np.float32),
                poses=z["poses"].astype(np.float32),
                K=z["K"].astype(np.float64),
                dist=z["dist"].astype(np.float64),
                meta={"cached": True, "cache_file": str(cache_file)},
            )

    K = synth.default_intrinsics(width, height, fov_deg)
    scene = synth.make_scene(scene_seed)
    poses = synth.make_trajectory(kind, n=n)
    frames, depths, poses = synth.render_sequence(scene, poses, K, dist_vec, (height, width))

    if cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_file, frames=frames, depths=depths, poses=poses, K=K, dist=dist_vec
        )

    return Sequence(
        name=kind,
        source="synthetic",
        frames=frames,
        depths=depths,
        poses=poses,
        K=K,
        dist=dist_vec,
        meta={"cached": False, "scene_seed": scene_seed, "fov_deg": fov_deg},
    )


# --------------------------------------------------------------------------- #
# real dataset loaders
# --------------------------------------------------------------------------- #
def _kitti_depth_from_velodyne(
    velo_path: Path, K: np.ndarray, height: int, width: int
) -> np.ndarray:
    """Project a KITTI Velodyne scan into a sparse camera-frame depth map.

    Args:
        velo_path: Path to a ``.bin`` Velodyne scan (float32 x,y,z,i records).
        K: ``(3, 3)`` rectified-camera intrinsics.
        height: Output height in pixels.
        width: Output width in pixels.

    Returns:
        ``(height, width)`` float32 depth in metres, zero where no return landed.
    """
    scan = np.fromfile(str(velo_path), dtype=np.float32).reshape(-1, 4)[:, :3]
    # KITTI velodyne -> camera (rectified) extrinsic.
    T = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    cam = scan @ T[:3, :3].T + T[:3, 3]
    z = cam[:, 2]
    ok = z > 0.5
    cam = cam[ok]
    z = z[ok]
    uv = (K[:3, :3] @ cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    depth = np.zeros((height, width), np.float32)
    px = np.rint(uv[:, 0]).astype(int)
    py = np.rint(uv[:, 1]).astype(int)
    keep = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    # painter's algorithm: write points far -> near so the nearest survives
    idx = np.argsort(-z[keep])
    depth[py[keep][idx], px[keep][idx]] = z[keep][idx]
    return depth


def load_kitti_odometry(
    root: str | os.PathLike[str],
    sequence: str = "00",
    size: tuple[int, int] = DEFAULT_SIZE,
    num_frames: int = 48,
    start: int = 0,
) -> Sequence:
    """Load a KITTI odometry sequence from the standard on-disk layout.

    Expects ``<root>/sequences/<seq>/image_0/*.png``, ``calib.txt`` and either
    ``poses.txt`` or a KITTI ground-truth pose file. Ground-truth depth is taken
    from ``velodyne_points/data/*.bin`` when present and left at zero otherwise
    (KITTI odometry does not ship dense depth).

    Args:
        root: Dataset root, e.g. ``data/kitti_odom``.
        sequence: Two-digit sequence id, e.g. ``"00"``.
        size: ``(height, width)`` to resize frames to.
        num_frames: Number of frames to return.
        start: First frame index to return.

    Returns:
        A :class:`Sequence`.

    Raises:
        FileNotFoundError: If the expected KITTI files are missing.
    """
    import cv2

    root = Path(root)
    seq_dir = root / "sequences" / str(sequence)
    img_dir = seq_dir / "image_0"
    if not img_dir.is_dir():
        raise FileNotFoundError(
            f"KITTI odometry image_0 directory not found at {img_dir}. "
            "Run `python scripts/fetch_data.py --dataset kitti_odom` for instructions."
        )

    calib_file = seq_dir / "calib.txt"
    if not calib_file.exists():
        raise FileNotFoundError(f"missing KITTI calibration file {calib_file}")
    K = None
    for line in calib_file.read_text().splitlines():
        if line.startswith("P0:"):
            P = np.array([float(v) for v in line.split()[1:]]).reshape(3, 4)
            K = P[:3, :3].copy()
            break
    if K is None:
        raise ValueError(f"no P0: entry in {calib_file}")

    images = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"no images in {img_dir}")
    images = images[start : start + num_frames]

    height, width = int(size[0]), int(size[1])
    frames = []
    for p in images:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA))
    frames = np.stack(frames, 0)
    sx = width / 1242.0  # KITTI image_0 is 1242x375
    sy = height / 375.0
    Ks = K.copy()
    Ks[0, :] *= sx
    Ks[1, :] *= sy

    poses_file = seq_dir / "poses.txt"
    poses = np.tile(np.eye(4), (len(images), 1, 1))
    if poses_file.exists():
        lines = [ln for ln in poses_file.read_text().splitlines() if ln.strip()]
        for i in range(min(len(lines), len(images))):
            T = np.array([float(v) for v in lines[i].split()]).reshape(3, 4)
            poses[i, :3, :] = T  # world->cam already, KITTI convention

    depths = np.zeros((len(images), height, width), np.float32)
    velo_dir = seq_dir / "velodyne_points" / "data"
    if velo_dir.is_dir():
        bins = sorted(velo_dir.glob("*.bin"))[start : start + num_frames]
        for i, b in enumerate(bins[: len(images)]):
            depths[i] = _kitti_depth_from_velodyne(b, Ks, height, width)

    return Sequence(
        name=f"kitti:{sequence}",
        source="kitti",
        frames=frames,
        depths=depths,
        poses=poses.astype(np.float32),
        K=Ks,
        dist=np.zeros(5, np.float64),
        meta={
            "root": str(root),
            "sequence": str(sequence),
            "image_size_native": [375, 1242],
            "has_lidar_depth": bool(velo_dir.is_dir()),
            "resize_scale": [sx, sy],
        },
    )


def load_euroc(
    root: str | os.PathLike[str],
    sequence: str = "MH_01_easy",
    size: tuple[int, int] = DEFAULT_SIZE,
    num_frames: int = 48,
    start: int = 0,
) -> Sequence:
    """Load an EuRoC MAV sequence (cam0) from the standard on-disk layout.

    Expects ``<root>/<sequence>/mav0/cam0/data/*.png``, ``cam0/sensor.yaml``
    (intrinsics + distortion) and ``state_groundtruth_estimate0/data.csv`` for
    poses. EuRoC ships stereo, not depth, so ground-truth depth is left at zero.

    Args:
        root: Dataset root, e.g. ``data/euroc``.
        sequence: EuRoC sequence name, e.g. ``"MH_01_easy"``.
        size: ``(height, width)`` to resize frames to.
        num_frames: Number of frames to return.
        start: First frame index to return.

    Returns:
        A :class:`Sequence`.

    Raises:
        FileNotFoundError: If the expected EuRoC files are missing.
    """
    import cv2
    import csv

    root = Path(root)
    seq_dir = root / str(sequence) / "mav0"
    cam_dir = seq_dir / "cam0"
    img_dir = cam_dir / "data"
    if not img_dir.is_dir():
        raise FileNotFoundError(
            f"EuRoC cam0 image directory not found at {img_dir}. "
            "Run `python scripts/fetch_data.py --dataset euroc` for instructions."
        )

    images = sorted(img_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"no images in {img_dir}")
    images = images[start : start + num_frames]

    height, width = int(size[0]), int(size[1])
    frames = []
    native = None
    for p in images:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if native is None:
            native = img.shape[:2]
        frames.append(cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA))
    # EuRoC cam0 is greyscale; replicate to 3 channels so the whole pipeline
    # (which assumes RGB) works unchanged.
    frames = np.stack([np.stack([f] * 3, -1) for f in frames], 0)

    # --- intrinsics + distortion from sensor.yaml -------------------------
    K = np.array([[458.654, 0.0, 367.215], [0.0, 457.296, 248.375], [0.0, 0.0, 1.0]])
    dist = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0])
    yaml_file = cam_dir / "sensor.yaml"
    if yaml_file.exists():
        try:
            import yaml

            cfg = yaml.safe_load(yaml_file.read_text())
            intr = cfg.get("intrinsics", K.ravel().tolist())
            K = np.array(intr, float).reshape(3, 3)
            d = cfg.get("distortion_coefficients", dist.tolist())
            dist = np.array(list(d)[:5] + [0.0] * max(0, 5 - len(d)), float)
            dist = dist[:5]
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[monogeo.data] could not parse {yaml_file}: {exc}; using defaults")

    if native is not None:
        sx = width / native[1]
        sy = height / native[0]
        Ks = K.copy()
        Ks[0, :] *= sx
        Ks[1, :] *= sy
    else:
        Ks = K.copy()

    # --- ground-truth poses (TUM-style: t[ns], px,py,pz, qw,qx,qy,qz) -------
    gt_file = seq_dir / "state_groundtruth_estimate0" / "data.csv"
    poses = np.tile(np.eye(4), (len(images), 1, 1))
    if gt_file.exists():
        rows = []
        with gt_file.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 8:
                    rows.append([float(v) for v in row[:8]])
        rows = rows[start : start + num_frames]
        for i, r in enumerate(rows[: len(images)]):
            qw, qx, qy, qz = r[4], r[5], r[6], r[7]
            R = _quat_to_matrix(qw, qx, qy, qz)
            # TUM ground truth is body(world)->world; camera pose is the inverse
            # of that, which is what the rest of the pipeline expects.
            T_world_body = np.eye(4)
            T_world_body[:3, :3] = R
            T_world_body[:3, 3] = r[1:4]
            poses[i] = np.linalg.inv(T_world_body)

    return Sequence(
        name=f"euroc:{sequence}",
        source="euroc",
        frames=frames,
        depths=np.zeros((len(images), height, width), np.float32),
        poses=poses.astype(np.float32),
        K=Ks,
        dist=dist.astype(np.float64),
        meta={
            "root": str(root),
            "sequence": str(sequence),
            "image_size_native": list(native) if native else None,
            "has_depth_gt": False,
        },
    )


def _quat_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Convert a (w, x, y, z) quaternion to a 3x3 rotation matrix."""
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n < 1e-12:
        return np.eye(3)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


# --------------------------------------------------------------------------- #
# public loader
# --------------------------------------------------------------------------- #
def load_sequence(
    name: str = "forward",
    size: tuple[int, int] = DEFAULT_SIZE,
    num_frames: int = 48,
    seed: int = 0,
    root: str | os.PathLike[str] = "data",
    cache: bool = True,
    fov_deg: float = 62.0,
    distortion: TypingSequence[float] | None = None,
    speed: float | None = None,
    altitude: float | None = None,
    verbose: bool = True,
) -> Sequence:
    """Load a flight log, falling back to the synthetic renderer.

    ``name`` selects the source:

    * ``"forward"`` / ``"orbit"`` / ``"descent"`` -> synthetic renderer
      (deterministic in ``seed``).
    * ``"kitti:<seq>"`` -> KITTI odometry under ``<root>/kitti_odom``.
    * ``"euroc:<seq>"`` -> EuRoC MAV under ``<root>/euroc``.

    If a real dataset is requested but its files are absent, the synthetic
    renderer is used instead and the fallback is reported.

    Args:
        name: Sequence specifier (see above).
        size: ``(height, width)`` for every frame.
        num_frames: Number of frames to load or render.
        seed: Seed for the synthetic renderer.
        root: Directory searched for real datasets and used for the render cache.
        cache: Whether to cache synthetic sequences as ``.npz``.
        fov_deg: Horizontal field of view of the synthetic camera.
        distortion: Synthetic distortion coefficients (defaults to
            :data:`monogeo.synth.DEFAULT_DIST`).
        speed: Optional override of the synthetic forward speed.
        altitude: Optional override of the synthetic starting altitude.
        verbose: Print which source was used.

    Returns:
        A :class:`Sequence`.

    Raises:
        ValueError: If ``name`` is an unknown synthetic kind and is not a valid
            ``kitti:``/``euroc:`` specifier.
    """
    if name.startswith("kitti"):
        seq = name.split(":", 1)[1] if ":" in name else "00"
        try:
            loaded = load_kitti_odometry(root, seq, size=size, num_frames=num_frames)
            if verbose:
                print(f"[monogeo.data] loaded KITTI odometry sequence {seq} from {root}")
            return loaded
        except FileNotFoundError as exc:
            if verbose:
                print(f"[monogeo.data] KITTI unavailable ({exc}); falling back to synthetic")
            name = "forward"

    if name.startswith("euroc"):
        seq = name.split(":", 1)[1] if ":" in name else "MH_01_easy"
        try:
            loaded = load_euroc(root, seq, size=size, num_frames=num_frames)
            if verbose:
                print(f"[monogeo.data] loaded EuRoC sequence {seq} from {root}")
            return loaded
        except FileNotFoundError as exc:
            if verbose:
                print(f"[monogeo.data] EuRoC unavailable ({exc}); falling back to synthetic")
            name = "forward"

    if name not in SYNTHETIC_KINDS:
        raise ValueError(
            f"unknown sequence {name!r}; expected one of {SYNTHETIC_KINDS} or 'kitti:<seq>' / 'euroc:<seq>'"
        )

    # Honour speed/altitude overrides by bypassing the cache when they are set.
    if speed is None and altitude is None and cache:
        seq = _render_synthetic(
            name,
            size=size,
            n=int(num_frames),
            seed=seed,
            fov_deg=fov_deg,
            dist=distortion,
            root=root,
            cache=True,
        )
        if verbose:
            cached = "cache" if seq.meta.get("cached") else "fresh render"
            print(f"[monogeo.data] using synthetic renderer ({name}, {cached}, seed={seed})")
        return seq

    height, width = int(size[0]), int(size[1])
    dist_vec = np.asarray(synth.DEFAULT_DIST if distortion is None else distortion, np.float64)
    K = synth.default_intrinsics(width, height, fov_deg)
    scene = synth.make_scene(seed)
    poses = synth.make_trajectory(
        name,
        n=num_frames,
        **({} if speed is None and altitude is None else {
            **({"speed": speed} if speed is not None else {}),
            **({"altitude": altitude} if altitude is not None else {}),
        }),
    )
    frames, depths, poses = synth.render_sequence(scene, poses, K, dist_vec, (height, width))
    if verbose:
        print(f"[monogeo.data] using synthetic renderer ({name}, uncached, seed={seed})")
    return Sequence(
        name=name,
        source="synthetic",
        frames=frames,
        depths=depths,
        poses=poses,
        K=K,
        dist=dist_vec,
        meta={"cached": False, "scene_seed": seed, "fov_deg": fov_deg},
    )


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
def to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert a numpy array to a contiguous float32 tensor."""
    return torch.from_numpy(np.ascontiguousarray(array)).float()


class SequenceDataset(Dataset):
    """Training/eval triplets ``(target, sources, K, GT depth, GT pose)``.

    Each item is a dictionary:

    ==================  ==========================  =========================
    key                 shape                       meaning
    ==================  ==========================  =========================
    ``target``          ``(3, H, W)``               target frame, RGB in [0, 1]
    ``sources``         ``(2, 3, H, W)``            +/-1 frames, RGB in [0, 1]
    ``K``               ``(3, 3)``                  intrinsics
    ``inv_K``           ``(3, 3)``                  ``inv(K)``
    ``depth``           ``(1, H, W)``               GT camera-frame depth (m)
    ``pose``            ``(4, 4)``                  target world->camera
    ``source_poses``    ``(2, 4, 4)``               source world->camera
    ``rel_poses``       ``(2, 4, 4)``               target->source transforms
    ``index``           scalar                      target frame index
    ==================  ==========================  =========================

    Invalid target frames (index 0 and ``T-1``, which have no +/-1 neighbours)
    are dropped, so ``len(dataset) == T - 2`` before ``max_samples`` clamping.
    """

    def __init__(
        self,
        sequence: Sequence,
        frame_ids: TypingSequence[int] = (0, -1, 1),
        max_samples: int | None = None,
        indices: TypingSequence[int] | None = None,
        augment: bool = False,
    ) -> None:
        """Build the triplet index list.

        Args:
            sequence: Sequence to sample from.
            frame_ids: Target-relative neighbour offsets. Only ``-1``/``+1`` can
                be used as sources; ``0`` marks the target.
            max_samples: Optional cap on the number of triplets (for CPU budget).
            indices: Explicit list of target frame indices to use.
            augment: Apply horizontal flip augmentation (training only).
                The intrinsics are unchanged by a horizontal flip only if it is
                paired with a principal-point shift, so this is left off by
                default and documented rather than silently wrong.
        """
        self.sequence = sequence
        self.frame_ids = tuple(frame_ids)
        self.augment = augment
        self.sources_ids = [f for f in self.frame_ids if f != 0]
        n = len(sequence)

        if indices is not None:
            valid = [int(i) for i in indices if 0 <= int(i) < n]
        else:
            valid = [
                i
                for i in range(n)
                if all(0 <= i + f < n for f in self.frame_ids)
            ]
        if max_samples is not None:
            valid = valid[: int(max_samples)]
        self.indices = valid
        if len(self.indices) == 0:
            raise ValueError(
                f"sequence {sequence.name!r} has {n} frames, too few for frame_ids={self.frame_ids}"
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, item: int) -> dict[str, Any]:
        """Return the triplet whose target frame is ``self.indices[item]``."""
        i = self.indices[item]
        seq = self.sequence

        def frame(idx: int) -> np.ndarray:
            return seq.frames[idx].astype(np.float32) / 255.0

        target = frame(i)
        sources = np.stack([frame(i + f) for f in self.sources_ids], 0)
        if self.augment:
            target = target[:, ::-1].copy()
            sources = sources[:, :, ::-1].copy()

        K = np.asarray(seq.K, np.float64)
        inv_K = np.linalg.inv(K)

        pose = np.asarray(seq.poses[i], np.float64)
        source_poses = np.stack([np.asarray(seq.poses[i + f], np.float64) for f in self.sources_ids], 0)
        # target->source rigid transform: X_src = T_rel @ X_target
        rel = np.stack([sp @ np.linalg.inv(pose) for sp in source_poses], 0)

        depth = np.asarray(seq.depths[i], np.float32)

        return {
            "target": torch.from_numpy(target.transpose(2, 0, 1).copy()),
            "sources": torch.from_numpy(sources.transpose(0, 3, 1, 2).copy()),
            "K": to_tensor(K),
            "inv_K": to_tensor(inv_K),
            "depth": torch.from_numpy(depth[None].copy()),
            "pose": to_tensor(pose),
            "source_poses": to_tensor(source_poses),
            "rel_poses": to_tensor(rel),
            "index": int(i),
        }

    def frame_indices(self) -> list[int]:
        """Return the target frame indices covered by this dataset."""
        return list(self.indices)
