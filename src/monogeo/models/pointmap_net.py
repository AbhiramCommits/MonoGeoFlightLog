"""Compact DUSt3R/MASt3R-style two-view pointmap regressor.

Architecture (deliberately small so it trains on a CPU in minutes):

1. **Patchify** both frames with a strided convolution (``patch_size`` 16).
2. A **shared (Siamese) ViT encoder** of ``encoder_depth`` pre-norm transformer
   blocks with learned positional embeddings.
3. A **cross-attention decoder**: ``decoder_depth`` blocks that alternate
   self-attention *within* each view and cross-attention *between* the two
   views, so each view's tokens can be explained using the other's geometry.
4. **Two heads** that upsample the token grid back to full resolution with a
   pixel-shuffle and emit a per-pixel 3D pointmap *in frame 1's camera frame*
   plus a per-pixel log-confidence.

Supervision follows DUSt3R: the ground-truth pointmaps are divided by a single
per-sample scale (the mean distance of the reference view's valid points), the
residual is L1, and it is weighted by the predicted confidence with a
``-alpha * log(conf)`` regulariser so that low-error pixels are encouraged to
take high confidence (the optimum is ``conf = alpha / error``).

Because the output is only defined up to that per-scene scale, every metric in
this repository is reported both before and after median scaling -- that is the
whole point of the scale-alignment analysis in step 4.

If local ``dust3r``/``mast3r`` checkpoints are available,
:meth:`PointmapNet.from_pretrained_if_available` will load them instead; the
repository never requires them, and the README states which path was used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PointmapNet", "PointmapOutput", "PointmapPairDataset"]


class _Block(nn.Module):
    """Pre-norm transformer block (self-attention + MLP)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        """Build the block."""
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention and the MLP with residual connections."""
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class _CrossBlock(nn.Module):
    """Self-attention within each view, then cross-attention between views.

    The same module instance is applied to both views (weight sharing), so the
    "two views" are treated symmetrically and the parameter count stays small.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        """Build the block."""
        super().__init__()
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_cross_q = nn.LayerNorm(dim)
        self.norm_cross_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Update both views' tokens jointly."""
        h1, h2 = self.norm_self(x1), self.norm_self(x2)
        x1 = x1 + self.self_attn(h1, h1, h1, need_weights=False)[0]
        x2 = x2 + self.self_attn(h2, h2, h2, need_weights=False)[0]

        x1 = (
            x1
            + self.cross_attn(
                self.norm_cross_q(x1),
                self.norm_cross_kv(x2),
                self.norm_cross_kv(x2),
                need_weights=False,
            )[0]
        )
        x2 = (
            x2
            + self.cross_attn(
                self.norm_cross_q(x2),
                self.norm_cross_kv(x1),
                self.norm_cross_kv(x1),
                need_weights=False,
            )[0]
        )

        x1 = x1 + self.mlp(self.norm_mlp(x1))
        x2 = x2 + self.mlp(self.norm_mlp(x2))
        return x1, x2


class _PointmapHead(nn.Module):
    """Upsample tokens to full resolution and emit xyz + log-confidence."""

    def __init__(self, dim: int, hidden: int, patch_size: int, out_ch: int = 4) -> None:
        """Build the head."""
        super().__init__()
        self.patch_size = int(patch_size)
        self.out_ch = int(out_ch)
        self.proj = nn.Linear(dim, hidden)
        self.net = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, out_ch * patch_size * patch_size, 1),
        )
        self.act = nn.PixelShuffle(patch_size)

    def forward(
        self, tokens: torch.Tensor, hw: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map ``(B, N, D)`` tokens to ``(B, H, W, 3)`` points and ``(B, H, W)`` confidence."""
        b, _n, _ = tokens.shape
        h, w = hw
        x = self.proj(tokens).transpose(1, 2).reshape(b, -1, h, w)
        x = self.act(self.net(x))  # (B, 4, H, W)
        pts = x[:, :3]
        log_conf = x[:, 3:4]
        conf = 1.0 + F.softplus(log_conf)
        return pts.permute(0, 2, 3, 1).contiguous(), conf[:, 0]


class PointmapOutput(dict):
    """Result of a :class:`PointmapNet` forward pass.

    Keys: ``pts1`` (B,H,W,3), ``pts2`` (B,H,W,3), ``conf1`` (B,H,W),
    ``conf2`` (B,H,W), ``norm`` (B,) the predicted scene scale (``1``, since the
    normalisation is applied to the targets), and ``log_conf1``/``log_conf2``.
    """


class PointmapNet(nn.Module):
    """Two-view pointmap regressor (compact DUSt3R-style)."""

    def __init__(
        self,
        img_size: tuple[int, int] = (128, 416),
        patch_size: int = 16,
        embed_dim: int = 128,
        encoder_depth: int = 4,
        decoder_depth: int = 2,
        num_heads: int = 4,
        head_hidden: int = 128,
    ) -> None:
        """Build the network.

        Args:
            img_size: ``(height, width)`` the positional embedding is sized for.
            patch_size: Patch side length; ``img_size`` must be divisible by it.
            embed_dim: Token width.
            encoder_depth: Number of shared encoder blocks.
            decoder_depth: Number of cross-attention decoder blocks.
            num_heads: Attention heads.
            head_hidden: Hidden width of the per-view head.

        Raises:
            ValueError: If ``img_size`` is not divisible by ``patch_size``.
        """
        super().__init__()
        h, w = int(img_size[0]), int(img_size[1])
        if h % patch_size or w % patch_size:
            raise ValueError(f"img_size {(h, w)} must be divisible by patch_size {patch_size}")
        self.img_size = (h, w)
        self.patch_size = int(patch_size)
        self.patch_grid = (h // patch_size, w // patch_size)
        self.embed_dim = int(embed_dim)

        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        n_patches = self.patch_grid[0] * self.patch_grid[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim) + 0.02)
        self.encoder = nn.ModuleList(
            [_Block(embed_dim, num_heads) for _ in range(int(encoder_depth))]
        )
        self.decoder = nn.ModuleList(
            [_CrossBlock(embed_dim, num_heads) for _ in range(int(decoder_depth))]
        )
        self.head = _PointmapHead(embed_dim, int(head_hidden), patch_size)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained_if_available(
        cls, path: str | Path | None = None, **kwargs: Any
    ) -> tuple[PointmapNet, str]:
        """Build the compact model, optionally loading local DUSt3R/MASt3R weights.

        Args:
            path: Optional path to a checkpoint. When ``None`` the method looks
                for a local ``dust3r``/``mast3r`` installation.
            **kwargs: Forwarded to the constructor.

        Returns:
            ``(model, provenance)`` where ``provenance`` is a human-readable
            string recorded in the results files, e.g. ``"compact (trained here)"``.
        """
        model = cls(**kwargs)
        if path is not None and Path(path).exists():
            try:
                ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
                state = ckpt.get("state_dict", ckpt)
                model.load_state_dict(state, strict=False)
                return model, f"pretrained weights from {path}"
            except Exception as exc:  # pragma: no cover - depends on user data
                return model, f"compact (failed to load {path}: {exc})"
        return model, "compact (trained here)"

    # ------------------------------------------------------------------ #
    def _tokenize(self, img: torch.Tensor) -> torch.Tensor:
        """Patchify ``(B, 3, H, W)`` and add positional embeddings."""
        f = self.patch_embed(img).flatten(2).transpose(1, 2)
        return f + self.pos_embed[:, : f.shape[1]]

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> PointmapOutput:
        """Predict pointmaps for both views in frame 1's camera frame.

        Args:
            img1: ``(B, 3, H, W)`` first frame in ``[0, 1]``.
            img2: ``(B, 3, H, W)`` second frame, same size.

        Returns:
            :class:`PointmapOutput`.

        Raises:
            ValueError: If the inputs are not the size the model was built for.
        """
        if tuple(img1.shape[-2:]) != self.img_size:
            raise ValueError(
                f"expected images of size {self.img_size}, got {tuple(img1.shape[-2:])}"
            )
        t1 = self._tokenize(img1)
        t2 = self._tokenize(img2)
        for blk in self.encoder:
            t1 = blk(t1)
            t2 = blk(t2)
        for blk in self.decoder:
            t1, t2 = blk(t1, t2)
        pts1, conf1 = self.head(t1, self.patch_grid)
        pts2, conf2 = self.head(t2, self.patch_grid)
        return PointmapOutput(
            pts1=pts1,
            pts2=pts2,
            conf1=conf1,
            conf2=conf2,
            norm=torch.ones(img1.shape[0], device=img1.device, dtype=img1.dtype),
        )


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class PointmapPairDataset(torch.utils.data.Dataset):
    """Frame pairs with ground-truth pointmaps for pointmap supervision.

    Each item contains the two frames, the two ground-truth pointmaps (both in
    frame 1's camera frame), validity masks, the normalisation scale and the
    intrinsics.
    """

    def __init__(
        self,
        sequence: Any,
        gap: int = 1,
        max_samples: int | None = None,
        indices: list[int] | None = None,
        sigma_scale: float = 0.0,
    ) -> None:
        """Build the pair index.

        Args:
            sequence: A :class:`monogeo.data.Sequence`.
            gap: Frame gap for each pair.
            max_samples: Optional cap on the number of pairs.
            indices: Optional explicit list of first-frame indices.
            sigma_scale: If positive, multiply the (correct) scale by
                ``exp(N(0, sigma))`` so the network must learn to be robust to a
                scale prior it cannot observe (DUSt3R-style scale augmentation).
        """
        self.sequence = sequence
        self.gap = int(gap)
        self.sigma_scale = float(sigma_scale)
        n = len(sequence)
        if indices is None:
            indices = list(range(n - self.gap))
        self.indices = [(int(i), int(i) + self.gap) for i in indices if int(i) + self.gap < n]
        if max_samples is not None:
            self.indices = self.indices[: int(max_samples)]
        if not self.indices:
            raise ValueError("no valid frame pairs for pointmap training")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        """Return one supervised pointmap pair."""
        from ..pointmap_ops import two_view_pointmap_targets

        i, j = self.indices[item]
        seq = self.sequence
        img1 = torch.from_numpy(seq.frames[i].astype(np.float32) / 255.0).permute(2, 0, 1)
        img2 = torch.from_numpy(seq.frames[j].astype(np.float32) / 255.0).permute(2, 0, 1)

        pm1, pm2, v1, v2, norm = two_view_pointmap_targets(seq, i, j)
        if self.sigma_scale > 0.0:
            rng = np.random.default_rng(1000 * i + j)
            norm = norm * float(np.exp(rng.normal(0.0, self.sigma_scale)))
        return {
            "img1": img1,
            "img2": img2,
            "pm1": torch.from_numpy((pm1 / norm).astype(np.float32)),
            "pm2": torch.from_numpy((pm2 / norm).astype(np.float32)),
            "valid1": torch.from_numpy(v1),
            "valid2": torch.from_numpy(v2),
            "norm": torch.tensor(float(norm), dtype=torch.float32),
            "K": torch.from_numpy(np.asarray(seq.K, np.float32)),
            "index": torch.tensor([i, j], dtype=torch.int64),
        }
