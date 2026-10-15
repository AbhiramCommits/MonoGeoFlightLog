"""Differentiable camera geometry for the self-supervised depth model.

This module is the *camera physics* the neural network is trained through. The
pair

* :class:`BackprojectDepth` -- pixel + depth -> camera-frame 3D point, using
  ``K^-1``; and
* :class:`Project3D` -- 3D point -> pixel in another camera, using ``K`` and a
  relative pose

is fully differentiable in both directions, so gradients from a photometric
image loss flow back into the predicted disparity *and* the predicted ego-motion.
Nothing here uses a finite-difference or a lookup approximation: the pinhole
model is written out explicitly.

Coordinate conventions
----------------------
* Camera points are ``(x, y, z)`` with ``z`` the optical axis, in metres.
* Poses are ``4x4`` transforms ``T`` such that ``X_b = R @ X_a + t``, i.e. ``T``
  maps frame ``a`` into frame ``b``.
* The **warp** used by the photometric loss backprojects *target* pixels with
  the target depth and then looks the result up in the *source* image. The
  transform it needs is therefore ``target -> source``, which for world->camera
  poses ``P`` is ``P_source @ inv(P_target)`` (exactly
  ``SequenceDataset['rel_poses']``).
* :class:`Project3D` returns ``grid_sample``-ready normalised coordinates in
  ``[-1, 1]``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "BackprojectDepth",
    "Project3D",
    "pixel_grid",
    "resize_like",
    "scale_intrinsics",
    "transform_points",
    "warp_image",
]


def pixel_grid(
    batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Homogeneous pixel coordinate grid.

    Args:
        batch: Batch size ``B``.
        height: Image height ``H``.
        width: Image width ``W``.
        device: Target device.
        dtype: Target floating dtype.

    Returns:
        ``(B, 3, H*W)`` tensor whose column ``n`` is ``[u, v, 1]`` for the
        ``n``-th pixel in row-major order.
    """
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    grid = torch.stack([xs, ys, ones], dim=0).reshape(3, -1)
    return grid.unsqueeze(0).expand(batch, 3, height * width).contiguous()


class BackprojectDepth(nn.Module):
    """Pixels + depth -> camera-frame 3D points (``K^-1`` applied to the grid).

    The pixel grid is cached per ``(H, W)`` so repeated calls in a training loop
    cost nothing.
    """

    def __init__(self, height: int, width: int) -> None:
        """Cache the homogeneous grid for images of size ``(height, width)``."""
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.register_buffer(
            "_grid",
            pixel_grid(1, self.height, self.width, torch.device("cpu")),
            persistent=False,
        )

    def forward(self, depth: torch.Tensor, inv_K: torch.Tensor) -> torch.Tensor:
        """Backproject a depth map into camera coordinates.

        Args:
            depth: ``(B, 1, H, W)`` camera-frame ``z`` depth in metres.
            inv_K: ``(B, 3, 3)`` inverse intrinsics.

        Returns:
            ``(B, H*W, 3)`` camera-frame points, row-major pixel order.
        """
        if depth.dim() != 4:
            raise ValueError(f"depth must be (B,1,H,W), got {tuple(depth.shape)}")
        b, _, h, w = depth.shape
        if (h, w) != (self.height, self.width):
            grid = pixel_grid(b, h, w, depth.device, depth.dtype)
        else:
            grid = self._grid.to(device=depth.device, dtype=depth.dtype).expand(b, 3, h * w)
        # (B, 3, N) -> unproject with inv_K, then scale by depth along the ray
        cam = inv_K @ grid
        cam = cam / cam[:, 2:3, :].clamp(min=1e-9)  # guard against a degenerate K
        points = cam.permute(0, 2, 1).contiguous()  # (B, N, 3)
        return points * depth.reshape(b, 1, h * w).permute(0, 2, 1)


class Project3D(nn.Module):
    """Camera-frame 3D points -> ``grid_sample`` coordinates in another camera.

    Implements ``p = pi(K (R X + t))`` with the perspective divide, then maps
    the result to ``grid_sample``'s ``[-1, 1]`` convention.
    """

    def __init__(self, height: int, width: int, eps: float = 1e-7) -> None:
        """Record the destination image size and the perspective-divide epsilon."""
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.eps = float(eps)

    def forward(
        self,
        points: torch.Tensor,
        K: torch.Tensor,
        T: torch.Tensor,
        return_depth: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Project points through a relative pose and the pinhole model.

        Args:
            points: ``(B, N, 3)`` points in the *source* camera frame.
            K: ``(B, 3, 3)`` destination intrinsics.
            T: ``(B, 4, 4)`` source->destination transform.
            return_depth: Also return the destination-frame ``z``.

        Returns:
            ``(B, H, W, 2)`` normalised coordinates, or a tuple with the
            ``(B, H, W)`` destination depth when ``return_depth`` is set.
            ``z <= 0`` samples are pushed far outside ``[-1, 1]`` so that
            ``grid_sample`` returns zeros for them instead of a mirrored point.
        """
        b = points.shape[0]
        cam = transform_points(points, T)
        proj = cam @ K.permute(0, 2, 1)  # (B, N, 3)
        z = proj[:, :, 2]
        safe_z = torch.where(z.abs() < self.eps, torch.full_like(z, self.eps), z)
        u = proj[:, :, 0] / safe_z
        v = proj[:, :, 1] / safe_z

        # normalise to grid_sample's [-1, 1] range
        u = 2.0 * u / max(self.width - 1, 1) - 1.0
        v = 2.0 * v / max(self.height - 1, 1) - 1.0
        invalid = (z <= self.eps).unsqueeze(-1)
        coords = torch.stack([u, v], dim=-1)
        coords = torch.where(invalid, torch.full_like(coords, -2.0), coords)
        coords = coords.reshape(b, self.height, self.width, 2)
        if return_depth:
            return coords, z.reshape(b, self.height, self.width)
        return coords


def transform_points(points: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Apply a rigid transform to a batch of 3D points.

    Args:
        points: ``(B, N, 3)`` points in the source frame.
        T: ``(B, 4, 4)`` source->destination transforms.

    Returns:
        ``(B, N, 3)`` points in the destination frame.
    """
    R = T[:, :3, :3]
    t = T[:, :3, 3]
    return torch.einsum("bij,bnj->bni", R, points) + t.unsqueeze(1)


def scale_intrinsics(K: torch.Tensor, scale: float) -> torch.Tensor:
    """Rescale intrinsics for an image resized by ``scale``.

    Uses the half-pixel-correct convention ``cx' = (cx + 0.5) * s - 0.5`` so that
    area-resized images stay geometrically consistent with the resized camera.

    Args:
        K: ``(B, 3, 3)`` intrinsics at the original resolution.
        scale: Target/original size ratio (``0.5`` halves the image).

    Returns:
        ``(B, 3, 3)`` rescaled intrinsics (a new tensor; the input is unchanged).
    """
    s = float(scale)
    Ks = K.clone()
    Ks[:, 0, 0] = K[:, 0, 0] * s
    Ks[:, 1, 1] = K[:, 1, 1] * s
    Ks[:, 0, 2] = (K[:, 0, 2] + 0.5) * s - 0.5
    Ks[:, 1, 2] = (K[:, 1, 2] + 0.5) * s - 0.5
    return Ks


def resize_like(img: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Area-resize ``(B, C, H, W)`` images to ``(height, width)``."""
    if img.shape[-2:] == (height, width):
        return img
    return F.interpolate(img, size=(height, width), mode="area")


def warp_image(source: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Sample ``source`` at ``coords`` (source image -> target frame).

    Args:
        source: ``(B, 3, H, W)`` source image.
        coords: ``(B, H, W, 2)`` normalised coordinates from :class:`Project3D`.

    Returns:
        ``(B, 3, H, W)`` warped image, zero outside the source frame.
    """
    return F.grid_sample(source, coords, mode="bilinear", padding_mode="zeros", align_corners=True)
