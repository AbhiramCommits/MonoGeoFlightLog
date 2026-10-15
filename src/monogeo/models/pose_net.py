"""Ego-motion network: 6-DoF relative pose from a concatenated frame pair.

The shared ResNet-18 trunk sees the target and source frame stacked along the
channel axis (a single 6-channel input) and the head regresses an axis-angle
rotation plus a translation. :func:`pose_vec_to_mat` lifts that to a 4x4 matrix
with Rodrigues' formula in a numerically safe form.

Convention: the returned matrix ``T`` maps *target* camera coordinates into the
*source* camera frame, i.e. ``X_source = R @ X_target + t``. That is exactly the
transform the photometric warp needs: pixels are backprojected with the target
depth and then looked up in the source image, so ``T = P_source @ inv(P_target)``.
Predicting the inverse instead still trains, but to a mirrored depth field --
which is why ``tests/test_train_depth.py`` pins the direction down with the
renderer's ground truth.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .depth_net import ResNetEncoder

__all__ = ["PoseNet", "pose_vec_to_mat", "skew"]


def skew(v: torch.Tensor) -> torch.Tensor:
    """Batch of skew-symmetric matrices from ``(B, 3)`` vectors.

    Args:
        v: ``(B, 3)`` vectors.

    Returns:
        ``(B, 3, 3)`` matrices ``[v]_x`` with ``[v]_x w == v x w``.
    """
    zero = torch.zeros_like(v[:, 0])
    row0 = torch.stack([zero, -v[:, 2], v[:, 1]], dim=-1)
    row1 = torch.stack([v[:, 2], zero, -v[:, 0]], dim=-1)
    row2 = torch.stack([-v[:, 1], v[:, 0], zero], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def pose_vec_to_mat(vec: torch.Tensor) -> torch.Tensor:
    """Lift a ``(B, 6)`` axis-angle + translation vector to a ``(B, 4, 4)`` matrix.

    Rodrigues' formula ``R = I + a [k]_x + b [k]_x^2`` is evaluated with the
    Taylor expansions ``a = 1 - t^2/6`` and ``b = 1/2 - t^2/24`` for tiny
    rotation angles, which keeps ``sin(t)/t`` and ``(1 - cos t)/t^2`` finite and
    differentiable at the identity rotation (the starting point of training).

    Args:
        vec: ``(B, 6)`` tensor: first three entries axis-angle, last three
            translation in the units of the depth predictions.

    Returns:
        ``(B, 4, 4)`` rigid transforms.

    Raises:
        ValueError: If ``vec`` is not ``(B, 6)``.
    """
    if vec.dim() != 2 or vec.shape[1] != 6:
        raise ValueError(f"pose vector must be (B, 6), got {tuple(vec.shape)}")
    angle = vec[:, :3]
    trans = vec[:, 3:]

    theta = torch.norm(angle, dim=1, keepdim=True)  # (B, 1)
    small = theta < 1e-4  # (B, 1)
    theta_safe = theta.clamp(min=1e-12)

    # K is skew(angle / |angle|), a *unit* axis skew matrix, so Rodrigues reads
    # R = I + sin(theta) K + (1 - cos(theta)) K^2 -- the 1/theta factors are
    # already folded into K. Getting this wrong still produces a valid-looking
    # rotation matrix, which is why the test checks a known angle explicitly.
    K = skew(angle / theta_safe)
    t = theta.unsqueeze(-1)  # (B, 1, 1)
    t2 = t * t
    sin_t = torch.sin(theta).unsqueeze(-1)
    cos_t = torch.cos(theta).unsqueeze(-1)
    tiny = small.unsqueeze(-1)
    a = torch.where(tiny, t - t * t2 / 6.0, sin_t)
    b = torch.where(tiny, t2 / 2.0 - t2 * t2 / 24.0, 1.0 - cos_t)

    eye = torch.eye(3, device=vec.device, dtype=vec.dtype).expand(vec.shape[0], 3, 3)
    R = eye + a * K + b * (K @ K)

    T = torch.zeros(vec.shape[0], 4, 4, device=vec.device, dtype=vec.dtype)
    T[:, :3, :3] = R
    T[:, :3, 3] = trans
    T[:, 3, 3] = 1.0
    return T


class PoseNet(nn.Module):
    """Regress the 6-DoF target->source relative pose for a frame pair."""

    def __init__(
        self,
        num_layers: int = 18,
        pretrained: bool = False,
        translation_scale: float = 1.0,
    ) -> None:
        """Build the pose network.

        Args:
            num_layers: Encoder depth (18).
            pretrained: Use ImageNet weights for the encoder trunk.
            translation_scale: Multiplier applied to the predicted translation,
                which sets the prior range of ego-motion magnitudes (metres).
        """
        super().__init__()
        self.encoder = ResNetEncoder(
            num_layers=num_layers, pretrained=pretrained, num_input_images=2
        )
        self.head = nn.Sequential(
            nn.Conv2d(self.encoder.num_ch_enc[-1], 256, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(256, 6)
        # Small but non-zero init. An exactly-zero pose would make the warp
        # independent of depth, so the disparity net would receive no
        # photometric gradient at the start of training.
        nn.init.normal_(self.fc.weight, std=1e-3)
        nn.init.zeros_(self.fc.bias)
        self.translation_scale = float(translation_scale)

    def forward(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """Predict the target->source transform for a batch of frame pairs.

        Args:
            target: ``(B, 3, H, W)`` target frames in ``[0, 1]``.
            source: ``(B, 3, H, W)`` source frames in ``[0, 1]``.

        Returns:
            ``(B, 4, 4)`` transforms taking target camera coordinates into the
            source camera frame.
        """
        feat = self.encoder(torch.cat([target, source], dim=1))[-1]
        pooled = F.adaptive_avg_pool2d(self.head(feat), 1).flatten(1)
        vec = self.fc(pooled)
        vec = torch.cat([vec[:, :3], vec[:, 3:] * self.translation_scale], dim=1)
        return pose_vec_to_mat(vec)
