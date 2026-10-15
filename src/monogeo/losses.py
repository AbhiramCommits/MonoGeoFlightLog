"""Self-supervised losses for monocular depth and ego-motion.

Everything here is the *explicit* formulation used by Monodepth2:

1. a photometric similarity that mixes SSIM and L1,
2. a per-pixel minimum over the source frames (so an occluded pixel only has to
   be explained by one neighbour),
3. an auto-mask that discards pixels which the *unwarped* source already
   explains (static scenes, objects moving with the camera), and
4. an edge-aware smoothness prior on the disparity, mean-normalised so the
   network is not incentivised to shrink the disparity everywhere.

No term is imported from a library: SSIM uses the separable 3x3 Gaussian-ish
window from the original paper and everything else is written out.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "automask_min_reprojection",
    "disparity_smoothness",
    "identity_photometric_error",
    "min_over_sources",
    "photometric_error",
    "photometric_loss",
    "ssim",
]

#: Separable 1D window from the SSIM paper (11-tap Gaussian, here a 3-tap
#: reduction which is what Monodepth2 uses for speed at high resolution).
_SSIM_WINDOW_1D = (0.27901, 0.44198, 0.27901)


def _ssim_window(window_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build the separable SSIM smoothing kernel ``(1, 1, k, k)``."""
    if window_size == 3:
        w = torch.tensor(_SSIM_WINDOW_1D, device=device, dtype=dtype)
    else:
        coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
        sigma = max(1.0, window_size / 5.0)
        w = torch.exp(-(coords**2) / (2 * sigma * sigma))
        w = w / w.sum()
    k2d = torch.outer(w, w)
    return k2d.view(1, 1, window_size, window_size)


def ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 3) -> torch.Tensor:
    """Structural similarity between two images.

    Args:
        x: ``(B, C, H, W)`` images in ``[0, 1]``.
        y: ``(B, C, H, W)`` images in ``[0, 1]``.
        window_size: Side length of the smoothing window.

    Returns:
        ``(B, 1, H, W)`` SSIM map (mean over channels), higher is better.

    Raises:
        ValueError: If the two inputs have different shapes.
    """
    if x.shape != y.shape:
        raise ValueError(f"ssim inputs differ: {tuple(x.shape)} vs {tuple(y.shape)}")
    pad = (window_size - 1) // 2
    win = _ssim_window(window_size, x.device, x.dtype)
    C1, C2 = 0.01**2, 0.03**2

    # local statistics are per-channel, so the filter runs on each channel
    # independently: reshape to (B*C, 1, H, W) and restore afterwards
    b, c = x.shape[0], x.shape[1]
    hw = x.shape[-2:]

    def filt(z: torch.Tensor) -> torch.Tensor:
        return F.conv2d(z.reshape(-1, 1, *hw), win, padding=pad).reshape(b, c, *hw)

    mu_x = filt(x)
    mu_y = filt(y)
    sigma_x2 = filt(x * x) - mu_x * mu_x
    sigma_y2 = filt(y * y) - mu_y * mu_y
    sigma_xy = filt(x * y) - mu_x * mu_y

    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return (num / den).mean(dim=1, keepdim=True)


def photometric_error(
    target: torch.Tensor,
    warped: torch.Tensor,
    ssim_weight: float = 0.85,
    l1_weight: float = 0.15,
    window_size: int = 3,
) -> torch.Tensor:
    """Per-pixel photometric error between a target and a warped source.

    ``error = ssim_weight * (1 - SSIM) / 2 + l1_weight * |target - warped|``.
    The ``/2`` keeps both terms in ``[0, 1]``: perfect agreement gives ``0`` and
    a maximally dissimilar pair approaches ``1``.

    Args:
        target: ``(B, 3, H, W)`` target image in ``[0, 1]``.
        warped: ``(B, 3, H, W)`` source image sampled at target pixels.
        ssim_weight: Weight of the structural term.
        l1_weight: Weight of the absolute-difference term.
        window_size: SSIM window side length.

    Returns:
        ``(B, 1, H, W)`` non-negative error map.
    """
    s = ssim(target, warped, window_size=window_size)
    l1 = torch.abs(target - warped).mean(dim=1, keepdim=True)
    return ssim_weight * (1.0 - s) / 2.0 + l1_weight * l1


def min_over_sources(errors: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
    """Per-pixel minimum of several ``(B, 1, H, W)`` error maps.

    Args:
        errors: A list of error maps (or a single stacked ``(B, S, 1, H, W)``
            tensor).

    Returns:
        ``(B, 1, H, W)`` element-wise minimum.
    """
    if isinstance(errors, torch.Tensor):
        return errors.min(dim=1).values
    return torch.stack(errors, dim=1).min(dim=1).values


def identity_photometric_error(
    target: torch.Tensor, sources: torch.Tensor, **kwargs: float
) -> list[torch.Tensor]:
    """Photometric error between the target and *unwarped* source frames.

    Args:
        target: ``(B, 3, H, W)`` target image.
        sources: ``(B, S, 3, H, W)`` source images.
        **kwargs: Forwarded to :func:`photometric_error`.

    Returns:
        List of ``(B, 1, H, W)`` error maps, one per source.
    """
    return [photometric_error(target, sources[:, i], **kwargs) for i in range(sources.shape[1])]


def automask_min_reprojection(
    target: torch.Tensor,
    warped_sources: list[torch.Tensor],
    unwarped_sources: torch.Tensor,
    generator: torch.Generator | None = None,
    tie_break_noise: float = 1e-5,
    **kwargs: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Monodepth2's auto-masking minimum-reprojection loss.

    A pixel is only trained on if warping a source frame onto it produces a
    *lower* photometric error than leaving the source unwarped. Pixels where the
    unwarped source wins are either static (a car moving with the camera, a
    textureless wall) or occluded, and including them teaches the network that
    "nothing moves", which collapses the ego-motion to zero.

    A tiny random tie-break is added to the identity error so that exactly equal
    errors do not systematically favour either branch; the generator is seeded
    so training runs stay reproducible.

    Args:
        target: ``(B, 3, H, W)`` target image.
        warped_sources: List of ``S`` warped source images.
        unwarped_sources: ``(B, S, 3, H, W)`` unwarped source images.
        generator: Torch generator for the tie-break noise.
        tie_break_noise: Standard deviation of the tie-break noise.
        **kwargs: Forwarded to :func:`photometric_error`.

    Returns:
        ``(loss, mask)``: ``(B, 1, H, W)`` masked per-pixel loss and a
        ``(B, 1, H, W)`` float mask that is ``1`` where the warp won.
    """
    s = len(warped_sources)
    reproj = torch.stack(
        [photometric_error(target, w, **kwargs) for w in warped_sources], dim=1
    )  # (B, S, 1, H, W)
    identity = torch.stack(identity_photometric_error(target, unwarped_sources, **kwargs), dim=1)

    noise = torch.randn(
        identity.shape, device=identity.device, dtype=identity.dtype, generator=generator
    )
    identity = identity + noise * tie_break_noise

    combined = torch.cat([identity, reproj], dim=1)
    best, idx = combined.min(dim=1)
    mask = (idx >= s).to(best.dtype).unsqueeze(1)  # (B, 1, H, W)
    return best * mask, mask


def disparity_smoothness(
    disp: torch.Tensor,
    img: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Edge-aware smoothness prior on the disparity.

    ``sum |grad disp| * exp(-|grad img|)``, so the prior is switched off across
    image edges. The disparity is divided by its own mean first; without that
    normalisation the loss is trivially minimised by pushing the whole
    disparity map towards zero.

    Args:
        disp: ``(B, 1, H, W)`` disparity (or inverse depth).
        img: ``(B, 3, H, W)`` image providing the edge weights.
        normalize: Mean-normalise the disparity before differentiating.

    Returns:
        Scalar smoothness loss.
    """
    d = disp
    if normalize:
        d = d / (d.mean(dim=(2, 3), keepdim=True) + 1e-7)

    mean_img = img.mean(dim=1, keepdim=True)
    grad_d_x = torch.abs(d[:, :, :, 1:] - d[:, :, :, :-1])
    grad_d_y = torch.abs(d[:, :, 1:, :] - d[:, :, :-1, :])
    grad_i_x = torch.abs(mean_img[:, :, :, 1:] - mean_img[:, :, :, :-1])
    grad_i_y = torch.abs(mean_img[:, :, 1:, :] - mean_img[:, :, :-1, :])

    loss_x = (grad_d_x * torch.exp(-grad_i_x)).mean()
    loss_y = (grad_d_y * torch.exp(-grad_i_y)).mean()
    return loss_x + loss_y


def photometric_loss(
    target: torch.Tensor,
    warped_sources: list[torch.Tensor],
    unwarped_sources: torch.Tensor | None = None,
    use_min_reproj: bool = True,
    use_automask: bool = True,
    generator: torch.Generator | None = None,
    ssim_weight: float = 0.85,
    l1_weight: float = 0.15,
) -> dict[str, torch.Tensor]:
    """Full photometric loss for one scale.

    Args:
        target: ``(B, 3, H, W)`` target image in ``[0, 1]``.
        warped_sources: Warped source images (one per source frame).
        unwarped_sources: ``(B, S, 3, H, W)`` source images, needed for
            auto-masking.
        use_min_reproj: Take the per-pixel minimum over sources.
        use_automask: Apply Monodepth2's auto-mask.
        generator: Seeded generator for the auto-mask tie-break.
        ssim_weight: SSIM weight inside the photometric term.
        l1_weight: L1 weight inside the photometric term.

    Returns:
        Dict with ``loss`` (scalar), ``per_pixel`` ``(B, 1, H, W)``,
        ``mask`` ``(B, 1, H, W)`` and ``mask_fraction`` (scalar).
        ``mask`` is all-ones when auto-masking is disabled.
    """
    kwargs = {"ssim_weight": ssim_weight, "l1_weight": l1_weight}
    if use_automask and unwarped_sources is not None:
        per_pixel, mask = automask_min_reprojection(
            target, warped_sources, unwarped_sources, generator=generator, **kwargs
        )
    else:
        if use_min_reproj:
            per_pixel = min_over_sources(
                [photometric_error(target, w, **kwargs) for w in warped_sources]
            )
        else:
            per_pixel = torch.stack(
                [photometric_error(target, w, **kwargs) for w in warped_sources], dim=1
            ).mean(dim=1)
        mask = torch.ones_like(per_pixel)

    denom = mask.sum().clamp(min=1.0)
    loss = (per_pixel * mask).sum() / denom
    return {
        "loss": loss,
        "per_pixel": per_pixel,
        "mask": mask,
        "mask_fraction": mask.mean(),
    }
