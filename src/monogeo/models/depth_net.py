"""Self-supervised monocular depth and ego-motion networks.

``DepthNet`` is a ResNet-18 encoder with a multi-scale up-convolution decoder
that predicts a disparity map through a sigmoid; ``PoseNet`` shares the same
encoder trunk over a concatenated frame pair and regresses a 6-DoF relative
pose (axis-angle + translation) which is lifted to a 4x4 matrix by
:func:`pose_vec_to_mat`.

Both are deliberately small so the whole training loop finishes in minutes on a
CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DepthNet", "ResNetEncoder", "UpConvBlock", "disp_to_depth"]


# --------------------------------------------------------------------------- #
# shared encoder
# --------------------------------------------------------------------------- #
class ResNetEncoder(nn.Module):
    """torchvision ResNet trunk returning all five feature stages.

    ``forward`` returns ``[c0, c1, c2, c3, c4]`` at strides
    ``[1/4, 1/4, 1/8, 1/16, 1/32]`` with channels ``[64, 64, 128, 256, 512]``.

    When ``num_input_images`` is 2 the first convolution is widened to 6 input
    channels by duplicating (and averaging) the pretrained 3-channel kernel, which
    is how Monodepth2 lets a single encoder trunk see a whole frame pair.
    """

    def __init__(
        self, num_layers: int = 18, pretrained: bool = False, num_input_images: int = 1
    ) -> None:
        """Build the trunk.

        Args:
            num_layers: Only ``18`` is supported (keeps the CPU budget honest).
            pretrained: Load ImageNet weights. Off by default so the repository
                works with no network access; turn it on for a small accuracy
                win if the weights are already cached.
            num_input_images: Number of RGB frames stacked on the input channel
                axis (1 for depth, 2 for pose).

        Raises:
            ValueError: If ``num_layers`` is not 18.
        """
        super().__init__()
        if int(num_layers) != 18:
            raise ValueError(f"only resnet18 is supported, got num_layers={num_layers}")
        import torchvision

        k = int(num_input_images)
        weights = None
        if pretrained:
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        try:
            net = torchvision.models.resnet18(weights=weights)
        except Exception as exc:  # pragma: no cover - offline / no cached weights
            if pretrained:
                print(
                    f"[monogeo.models] ImageNet weights unavailable ({exc}); training from scratch"
                )
            net = torchvision.models.resnet18(weights=None)

        if k > 1:
            old = net.conv1
            new = nn.Conv2d(
                3 * k, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False
            )
            new.weight.data = old.weight.data.repeat(1, k, 1, 1) / float(k)
            net.conv1 = new

        # normalisation happens inside the model, like Monodepth2
        base_mean = torch.tensor([0.485, 0.456, 0.406]).repeat(k).view(1, 3 * k, 1, 1)
        base_std = torch.tensor([0.229, 0.224, 0.225]).repeat(k).view(1, 3 * k, 1, 1)
        self.register_buffer("mean", base_mean, persistent=False)
        self.register_buffer("std", base_std, persistent=False)

        self.layer0 = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.num_ch_enc = [64, 64, 128, 256, 512]
        self.num_input_images = k

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Encode an image (or frame pair) into five feature maps."""
        if x.shape[1] != 3 * self.num_input_images:
            raise ValueError(
                f"encoder expects {3 * self.num_input_images} input channels, got {x.shape[1]}"
            )
        x = (x - self.mean) / self.std
        c0 = self.layer0(x)
        c1 = self.layer1(c0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return [c0, c1, c2, c3, c4]


# --------------------------------------------------------------------------- #
# depth
# --------------------------------------------------------------------------- #
def disp_to_depth(disp: torch.Tensor, min_depth: float, max_depth: float) -> torch.Tensor:
    """Convert a sigmoid disparity in ``[0, 1]`` to depth in metres.

    The mapping is ``depth = 1 / (min_disp + (max_disp - min_disp) * disp)``
    with ``min_disp = 1 / max_depth`` and ``max_disp = 1 / min_depth``, so the
    output is monotone in the disparity and bounded by the configured depth
    range. Monotonicity in ``1/depth`` (rather than in ``depth``) is what makes
    the photometric loss well conditioned at long range.

    Args:
        disp: ``(B, 1, H, W)`` sigmoid disparities in ``[0, 1]``.
        min_depth: Closest representable depth, in metres.
        max_depth: Furthest representable depth, in metres.

    Returns:
        ``(B, 1, H, W)`` depth in metres.
    """
    min_disp = 1.0 / float(max_depth)
    max_disp = 1.0 / float(min_depth)
    return 1.0 / (min_disp + (max_disp - min_disp) * disp.clamp(1e-6, 1.0 - 1e-6))


class UpConvBlock(nn.Module):
    """Upsample followed by a 3x3 convolution, ReLU and reflection padding."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        """Create the up-convolution block."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample by 2 then convolve."""
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False))


class DepthNet(nn.Module):
    """Multi-scale disparity network (Monodepth2-style architecture).

    ``forward`` returns a list of disparity maps ordered from the finest scale
    (full resolution) to the coarsest, together with the depth at the finest
    scale for convenience.
    """

    def __init__(
        self,
        num_layers: int = 18,
        pretrained: bool = False,
        num_scales: int = 4,
        min_depth: float = 0.5,
        max_depth: float = 120.0,
    ) -> None:
        """Build the encoder/decoder.

        Args:
            num_layers: ResNet depth for the encoder (18).
            pretrained: Use ImageNet weights for the encoder.
            num_scales: Number of decoder outputs to supervise (1..5).
            min_depth: Closest representable depth in metres.
            max_depth: Furthest representable depth in metres.
        """
        super().__init__()
        self.encoder = ResNetEncoder(num_layers=num_layers, pretrained=pretrained)
        n = self.encoder.num_ch_enc
        # decoder: /32 -> /16 -> /8 -> /4 -> /2 -> /1
        self.up4 = UpConvBlock(n[4], n[3])
        self.iconv4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(2 * n[3], n[3], 3, bias=False),
            nn.BatchNorm2d(n[3]),
            nn.ReLU(inplace=True),
        )
        self.up3 = UpConvBlock(n[3], n[2])
        self.iconv3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(2 * n[2], n[2], 3, bias=False),
            nn.BatchNorm2d(n[2]),
            nn.ReLU(inplace=True),
        )
        self.up2 = UpConvBlock(n[2], n[1])
        self.iconv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(2 * n[1], n[1], 3, bias=False),
            nn.BatchNorm2d(n[1]),
            nn.ReLU(inplace=True),
        )
        self.up1 = UpConvBlock(n[1], n[0])
        self.up0 = UpConvBlock(n[0], 32)

        def disp_conv(ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.ReflectionPad2d(1), nn.Conv2d(ch, 1, 3, bias=True), nn.Sigmoid()
            )

        # decoder outputs: /1, /2, /4, /8, /16 (finest first)
        self.disp1 = disp_conv(n[0])  # /2
        self.disp2 = disp_conv(n[1])  # /4
        self.disp3 = disp_conv(n[2])  # /8
        self.disp4 = disp_conv(n[3])  # /16
        self.disp0 = disp_conv(32)  # /1

        self.num_scales = int(num_scales)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Predict disparities.

        Args:
            x: ``(B, 3, H, W)`` RGB images in ``[0, 1]``.

        Returns:
            ``(disparities, depth)`` where ``disparities`` is a list ordered
            finest-first and ``depth`` is the full-resolution depth in metres.
        """
        feats = self.encoder(x)
        _c0, c1, c2, c3, c4 = feats

        x4 = self.iconv4(torch.cat([self.up4(c4), c3], 1))  # /16
        x3 = self.iconv3(torch.cat([self.up3(x4), c2], 1))  # /8
        x2 = self.iconv2(torch.cat([self.up2(x3), c1], 1))  # /4
        # The ResNet trunk's highest-resolution features (c0/c1) both live at
        # /4, so the final two decoder stages have no skip connection: they are
        # pure up-convolutions supervised by the photometric loss. This keeps
        # the decoder cheap enough for CPU training at these resolutions.
        x1 = self.up1(x2)  # /2
        x0 = self.up0(x1)  # /1

        disps = [self.disp0(x0), self.disp1(x1), self.disp2(x2), self.disp3(x3), self.disp4(x4)]
        disps = disps[: self.num_scales]
        depth = disp_to_depth(disps[0], self.min_depth, self.max_depth)
        return disps, depth
