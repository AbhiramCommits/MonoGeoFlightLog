"""Self-supervised training of the depth + ego-motion networks.

The loss is the Monodepth2 formulation, computed on the *central* frame of each
triplet only (the depth network runs on the target frame, the pose network on
the target/source pairs). Restricting the loss to the central frame is a
deliberate CPU-budget choice: it cuts the depth forward pass by a factor of
three with no change to the loss definition, and every frame is still visited as
a target as the triplets slide along the sequence.

Nothing here uses ground-truth depth or pose. Those are only used afterwards, for
evaluation, and for the sanity check in :mod:`tests.test_train_depth` that the
warp direction is right.
"""

from __future__ import annotations

import csv
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .cli import cfg_get, save_json
from .data import SequenceDataset, load_sequence
from .geometry import BackprojectDepth, Project3D, resize_like, scale_intrinsics, warp_image
from .losses import disparity_smoothness, photometric_loss
from .models import DepthNet, PoseNet

__all__ = ["DepthTrainer", "build_models", "run_train_depth"]


def build_models(
    cfg: dict[str, Any], device: torch.device | str = "cpu"
) -> tuple[DepthNet, PoseNet]:
    """Construct the depth and pose networks from a config dict.

    Args:
        cfg: Configuration with ``depth`` and ``train_depth`` sections.
        device: Torch device string or object.

    Returns:
        ``(depth_net, pose_net)`` on ``device``.
    """
    depth_net = DepthNet(
        num_layers=int(cfg_get(cfg, "depth.encoder_layers", 18)),
        pretrained=bool(cfg_get(cfg, "depth.pretrained", False)),
        num_scales=int(cfg_get(cfg, "depth.num_scales", 4)),
        min_depth=float(cfg_get(cfg, "depth.min_depth", 0.5)),
        max_depth=float(cfg_get(cfg, "depth.max_depth", 120.0)),
    )
    pose_net = PoseNet(
        num_layers=int(cfg_get(cfg, "depth.encoder_layers", 18)),
        pretrained=bool(cfg_get(cfg, "depth.pretrained", False)),
        translation_scale=float(cfg_get(cfg, "depth.translation_scale", 1.0)),
    )
    return depth_net.to(device), pose_net.to(device)


class DepthTrainer:
    """Owns the nets, the optimiser and the multi-scale loss computation."""

    def __init__(
        self,
        cfg: dict[str, Any],
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
        """Build the trainer.

        Args:
            cfg: Full configuration dict.
            device: Torch device.
            seed: Seed for weight init, shuffling and the auto-mask tie-break.
        """
        self.cfg = cfg
        self.device = torch.device(device)
        self.seed = int(seed)
        self.depth_net, self.pose_net = build_models(cfg, self.device)

        self.min_depth = float(cfg_get(cfg, "depth.min_depth", 0.5))
        self.max_depth = float(cfg_get(cfg, "depth.max_depth", 120.0))
        self.num_scales = self.depth_net.num_scales
        self.w_smooth = float(cfg_get(cfg, "train_depth.w_smoothness", 0.001))
        self.ssim_w = float(cfg_get(cfg, "train_depth.ssim_weight", 0.85))
        self.l1_w = float(cfg_get(cfg, "train_depth.l1_weight", 0.15))
        self.w_photo = float(cfg_get(cfg, "train_depth.w_photometric", 1.0))
        self.use_automask = bool(cfg_get(cfg, "train_depth.use_automask", True))
        self.use_min_reproj = bool(cfg_get(cfg, "train_depth.use_min_reproj", True))
        # supervise the three finest decoder outputs
        self.supervised_scales = list(range(min(3, self.num_scales)))
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)

    # ------------------------------------------------------------------ #
    def _scale_factors(self, height: int, width: int) -> list[tuple[float, int, int]]:
        """Per-supervised-scale ``(factor, height, width)`` tuples."""
        out = []
        for s in self.supervised_scales:
            f = 2.0**s
            out.append((1.0 / f, max(1, int(round(height / f))), max(1, int(round(width / f)))))
        return out

    def compute_losses(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run both networks and evaluate the multi-scale self-supervised loss.

        Args:
            batch: A batch produced by :class:`monogeo.data.SequenceDataset`.

        Returns:
            Dict with ``total``, ``photometric``, ``smoothness`` (all scalars)
            and ``mask_fraction`` (scalar).
        """
        target = batch["target"].to(self.device)
        sources = batch["sources"].to(self.device)  # (B, S, 3, H, W)
        K = batch["K"].to(self.device)
        batch["inv_K"].to(self.device)
        n_src = sources.shape[1]
        b, _, h, w = target.shape

        disps, _ = self.depth_net(target)
        disp0 = disps[0]

        total_photo = target.new_zeros(())
        total_smooth = target.new_zeros(())
        mask_frac = target.new_zeros(())

        poses = []
        for i in range(n_src):
            poses.append(self.pose_net(target, sources[:, i]))

        for factor, sh, sw in self._scale_factors(h, w):
            idx = int(round(1.0 / factor)) - 1
            disp_s = disps[idx] if idx < len(disps) else disp0
            if disp_s.shape[-2:] != (sh, sw):
                disp_s = F.interpolate(disp_s, size=(sh, sw), mode="bilinear", align_corners=False)
            depth_s = 1.0 / (
                1.0 / self.max_depth
                + (1.0 / self.min_depth - 1.0 / self.max_depth) * disp_s.clamp(1e-6, 1 - 1e-6)
            )
            K_s = scale_intrinsics(K, factor)
            inv_K_s = torch.linalg.inv(K_s)
            target_s = resize_like(target, sh, sw)
            sources_s = resize_like(sources.reshape(b * n_src, 3, h, w), sh, sw).reshape(
                b, n_src, 3, sh, sw
            )

            backproject = BackprojectDepth(sh, sw).to(self.device)
            project = Project3D(sh, sw).to(self.device)
            points = backproject(depth_s, inv_K_s)  # target pixels -> target-camera 3D

            warped = []
            for i in range(n_src):
                # ``points`` are in the target frame and ``poses[i]`` maps
                # target -> source, so this samples the source image at target
                # pixels: the forward warp of the photometric loss.
                coords = project(points, K_s, poses[i])
                warped.append(warp_image(sources_s[:, i], coords))

            out = photometric_loss(
                target_s,
                warped,
                unwarped_sources=sources_s,
                use_min_reproj=self.use_min_reproj,
                use_automask=self.use_automask,
                generator=self.generator,
                ssim_weight=self.ssim_w,
                l1_weight=self.l1_w,
            )
            total_photo = total_photo + out["loss"]
            mask_frac = mask_frac + out["mask_fraction"]
            total_smooth = total_smooth + disparity_smoothness(disp_s, target_s)

        n = float(len(self._scale_factors(h, w)))
        photo = total_photo / n
        smooth = total_smooth / n
        return {
            "total": self.w_photo * photo + self.w_smooth * smooth,
            "photometric": photo,
            "smoothness": smooth,
            "mask_fraction": mask_frac / n,
        }

    @torch.no_grad()
    def evaluate(self, loader: Iterable[dict[str, torch.Tensor]]) -> dict[str, float]:
        """Mean losses over a loader, with the model in eval mode."""
        was_training = self.depth_net.training
        self.depth_net.eval()
        self.pose_net.eval()
        acc: dict[str, float] = {"total": 0.0, "photometric": 0.0, "smoothness": 0.0}
        count = 0
        for batch in loader:
            out = self.compute_losses(batch)
            for k in acc:
                acc[k] += float(out[k])
            count += 1
        if was_training:
            self.depth_net.train()
            self.pose_net.train()
        return {k: v / max(1, count) for k, v in acc.items()}


def _to_float(x: torch.Tensor) -> float:
    """Detach a scalar tensor to a Python float."""
    return float(x.detach().cpu())


def run_train_depth(args: Any) -> int:
    """``monogeo train-depth``: train the depth + pose nets and log the curve.

    Args:
        args: Parsed argparse namespace.

    Returns:
        Process exit code (0 on success).
    """
    cfg = getattr(args, "cfg", {}) or {}
    seed = int(getattr(args, "seed", 0) or 0)
    torch.manual_seed(seed)
    np.random.seed(seed)

    size = (int(cfg_get(cfg, "data.height", 128)), int(cfg_get(cfg, "data.width", 416)))
    root = cfg_get(cfg, "data.data_root", "data")
    cache = bool(cfg_get(cfg, "data.cache", True))
    train_name = cfg_get(cfg, "data.sequence", "forward")
    val_name = cfg_get(cfg, "data.eval_sequence", "orbit")
    num_frames = int(cfg_get(cfg, "data.num_frames", 48))
    max_samples = int(cfg_get(cfg, "train_depth.max_samples", 240))

    epochs = int(getattr(args, "epochs", None) or cfg_get(cfg, "train_depth.epochs", 8))
    batch_size = int(getattr(args, "batch_size", None) or cfg_get(cfg, "train_depth.batch_size", 4))
    lr = float(getattr(args, "lr", None) or cfg_get(cfg, "train_depth.lr", 1e-4))
    out_dir = Path(
        getattr(args, "out_dir", None) or cfg_get(cfg, "train_depth.out_dir", "runs/depth")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[train-depth] seed={seed} size={size[1]}x{size[0]} epochs={epochs} batch={batch_size} lr={lr}"
    )
    t_data = time.perf_counter()
    train_seq = load_sequence(
        train_name,
        size=size,
        num_frames=num_frames,
        seed=seed,
        root=root,
        cache=cache,
        fov_deg=float(cfg_get(cfg, "synth.fov_deg", 62.0)),
        distortion=cfg_get(cfg, "synth.distortion", None),
        verbose=bool(getattr(args, "verbose", False)),
    )
    val_seq = load_sequence(
        val_name,
        size=size,
        num_frames=num_frames,
        seed=seed,
        root=root,
        cache=cache,
        fov_deg=float(cfg_get(cfg, "synth.fov_deg", 62.0)),
        distortion=cfg_get(cfg, "synth.distortion", None),
        verbose=False,
    )
    # Split the *training* sequence so the reported val curve is honest, and
    # keep the separate eval sequence untouched for step 4/5.
    n_train_seq = len(train_seq)
    split = max(4, int(0.7 * n_train_seq))
    train_idx = list(range(1, min(split, n_train_seq - 1)))
    val_idx = list(range(max(1, split - 1), n_train_seq - 1))

    train_ds = SequenceDataset(
        train_seq, frame_ids=(0, -1, 1), max_samples=max_samples, indices=train_idx
    )
    val_ds = SequenceDataset(val_seq, frame_ids=(0, -1, 1), indices=val_idx[:12])
    data_secs = time.perf_counter() - t_data
    print(
        f"[train-depth] train triplets {len(train_ds)} ({train_seq.source}), "
        f"val triplets {len(val_ds)} ({val_seq.source}), data {data_secs:.1f}s"
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    trainer = DepthTrainer(cfg, device="cpu", seed=seed)
    params = list(trainer.depth_net.parameters()) + list(trainer.pose_net.parameters())
    n_params = sum(p.numel() for p in params)
    opt = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 2), gamma=0.5)
    print(
        f"[train-depth] parameters {n_params/1e6:.2f} M, supervised scales {trainer.supervised_scales}"
    )

    log_path = out_dir / "log.csv"
    history: list[dict[str, float]] = []
    t_train = time.perf_counter()
    first_loss = None
    for epoch in range(1, epochs + 1):
        trainer.depth_net.train()
        trainer.pose_net.train()
        running = {"total": 0.0, "photometric": 0.0, "smoothness": 0.0, "mask_fraction": 0.0}
        steps = 0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            out = trainer.compute_losses(batch)
            out["total"].backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
            opt.step()
            for k in running:
                running[k] += _to_float(out[k])
            steps += 1
        scheduler.step()
        train_losses = {k: v / max(1, steps) for k, v in running.items()}
        val_losses = trainer.evaluate(val_loader)
        if first_loss is None:
            first_loss = train_losses["photometric"]
        row = {
            "epoch": float(epoch),
            "train_total": train_losses["total"],
            "train_photometric": train_losses["photometric"],
            "train_smoothness": train_losses["smoothness"],
            "train_mask_fraction": train_losses["mask_fraction"],
            "val_total": val_losses["total"],
            "val_photometric": val_losses["photometric"],
            "val_smoothness": val_losses["smoothness"],
            "lr": float(opt.param_groups[0]["lr"]),
            "steps": float(steps),
            "elapsed_s": time.perf_counter() - t_train,
        }
        history.append(row)
        print(
            f"[train-depth] epoch {epoch}/{epochs}: train {row['train_total']:.5f} "
            f"(photo {row['train_photometric']:.5f}, smooth {row['train_smoothness']:.5f}, "
            f"mask {row['train_mask_fraction']:.3f}) | val {row['val_total']:.5f} "
            f"| {row['elapsed_s']:.0f}s"
        )
        _write_log(log_path, history)

    torch.save(
        {
            "state_dict": trainer.depth_net.state_dict(),
            "min_depth": trainer.min_depth,
            "max_depth": trainer.max_depth,
            "num_scales": trainer.num_scales,
            "num_layers": 18,
            "resolution_hw": list(size),
            "config": {k: cfg_get(cfg, k) for k in ("depth",)},
        },
        out_dir / "depth.pt",
    )
    torch.save(
        {
            "state_dict": trainer.pose_net.state_dict(),
            "num_layers": 18,
            "translation_scale": trainer.pose_net.translation_scale,
        },
        out_dir / "pose.pt",
    )

    last = history[-1]
    reduction = 100.0 * (first_loss - last["train_photometric"]) / max(1e-9, first_loss)
    summary = {
        "step": "train-depth",
        "sequence": train_seq.name,
        "val_sequence": val_seq.name,
        "data_source": train_seq.source,
        "resolution_hw": list(size),
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "num_parameters_millions": n_params / 1e6,
        "num_train_triplets": len(train_ds),
        "num_val_triplets": len(val_ds),
        "supervised_scales": trainer.supervised_scales,
        "use_automask": trainer.use_automask,
        "use_min_reproj": trainer.use_min_reproj,
        "w_smoothness": trainer.w_smooth,
        "photometric_first_epoch": first_loss,
        "photometric_last_epoch": last["train_photometric"],
        "photometric_reduction_pct": reduction,
        "train_total_last": last["train_total"],
        "val_total_last": last["val_total"],
        "val_photometric_last": last["val_photometric"],
        "training_seconds": time.perf_counter() - t_train,
        "history": history,
        "checkpoints": {
            "depth": str(out_dir / "depth.pt"),
            "pose": str(out_dir / "pose.pt"),
            "log": str(log_path),
        },
    }
    save_json(out_dir / "summary.json", summary)
    print(
        f"[train-depth] photometric loss {first_loss:.5f} -> {last['train_photometric']:.5f} "
        f"({reduction:.1f}% reduction) in {summary['training_seconds']:.0f}s"
    )
    print(f"[train-depth] saved {out_dir/'depth.pt'}, {out_dir/'pose.pt'}, {log_path}")
    return 0


def _write_log(path: Path, history: list[dict[str, float]]) -> None:
    """Rewrite the CSV loss log (small enough that rewriting is simplest)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
