"""Training the two-view pointmap regressor with confidence-weighted L1.

The objective is DUSt3R's: the ground-truth pointmaps are divided by one
per-sample scale, the residual is L1 (robust to the long tails of aerial depth),
and it is weighted by the predicted per-pixel confidence with a
``-alpha * log(conf)`` regulariser. Minimising that in ``conf`` gives
``conf = alpha / residual``, so the network learns to be confident exactly where
it is accurate, and the confidence map is meaningful enough to drive the
pose/focal estimators in :mod:`monogeo.pointmap_ops`.

Training is seeded and runs on CPU in a couple of minutes.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .cli import cfg_get, save_json
from .data import load_sequence
from .models.pointmap_net import PointmapNet, PointmapPairDataset

__all__ = ["pointmap_loss", "run_train_pointmap"]


def pointmap_loss(
    output: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    alpha: float = 0.2,
) -> dict[str, torch.Tensor]:
    """Confidence-weighted L1 on scale-normalised pointmaps (DUSt3R objective).

    ``loss = mean( conf * |pred - gt| - alpha * log(conf) )`` over valid pixels,
    averaged over both views. The ``-alpha*log(conf)`` term is what stops the
    trivial solution ``conf = 0``: optimising it in closed form gives
    ``conf = alpha / residual``.

    Args:
        output: Model output with ``pts1``, ``pts2``, ``conf1``, ``conf2``.
        target: Batch with scale-normalised ``pm1``, ``pm2``, ``valid1``,
            ``valid2``.
        alpha: Confidence regularisation weight.

    Returns:
        Dict with ``total`` (scalar), ``regression`` (confidence-weighted L1),
        ``mean_conf`` and ``viewport`` breakdown.
    """
    total = output["pts1"].new_zeros(())
    reg = output["pts1"].new_zeros(())
    conf_sum = output["pts1"].new_zeros(())
    denom = 0.0
    per_view: dict[str, float] = {}

    for view in ("1", "2"):
        pred = output[f"pts{view}"]
        gt = target[f"pm{view}"]
        conf = output[f"conf{view}"]
        valid = target[f"valid{view}"]

        err = torch.abs(pred - gt).mean(dim=-1)  # (B, H, W)
        w = valid.to(err.dtype)
        n_valid = w.sum().clamp(min=1.0)
        term = (conf * err - alpha * torch.log(conf.clamp(min=1e-6))) * w
        loss_view = term.sum() / n_valid

        total = total + loss_view
        reg = reg + (conf * err * w).sum() / n_valid
        conf_sum = conf_sum + (conf * w).sum() / n_valid
        denom += 1.0
        per_view[f"regression_view{view}"] = float(((conf * err * w).sum() / n_valid).detach())

    total = total / denom
    reg = reg / denom
    conf_mean = conf_sum / denom
    return {
        "total": total,
        "regression": reg,
        "mean_conf": conf_mean,
        "per_view": per_view,
    }


def run_train_pointmap(args: Any) -> int:
    """``monogeo train-pointmap``: train the compact pointmap model.

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
    num_frames = int(cfg_get(cfg, "data.num_frames", 120))

    epochs = int(getattr(args, "epochs", None) or cfg_get(cfg, "train_pointmap.epochs", 12))
    batch_size = int(
        getattr(args, "batch_size", None) or cfg_get(cfg, "train_pointmap.batch_size", 2)
    )
    lr = float(getattr(args, "lr", None) or cfg_get(cfg, "train_pointmap.lr", 3e-4))
    alpha = float(cfg_get(cfg, "train_pointmap.conf_reg", 0.2))
    gap = int(cfg_get(cfg, "train_pointmap.gap", 1))
    max_samples = int(cfg_get(cfg, "train_pointmap.max_samples", 160))
    out_dir = Path(
        getattr(args, "out_dir", None) or cfg_get(cfg, "train_pointmap.out_dir", "runs/pointmap")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[train-pointmap] seed={seed} size={size[1]}x{size[0]} epochs={epochs} batch={batch_size} lr={lr}"
    )

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
        num_frames=max(16, num_frames // 4),
        seed=seed,
        root=root,
        cache=cache,
        fov_deg=float(cfg_get(cfg, "synth.fov_deg", 62.0)),
        distortion=cfg_get(cfg, "synth.distortion", None),
        verbose=False,
    )
    split = max(4, int(0.7 * len(train_seq)))
    train_ds = PointmapPairDataset(
        train_seq, gap=gap, max_samples=max_samples, indices=list(range(1, split))
    )
    val_ds = PointmapPairDataset(
        val_seq, gap=gap, indices=list(range(1, min(13, len(val_seq) - gap)))
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    print(
        f"[train-pointmap] train pairs {len(train_ds)} ({train_seq.source}), val pairs {len(val_ds)} ({val_seq.source})"
    )

    model, provenance = PointmapNet.from_pretrained_if_available(
        path=cfg_get(cfg, "pointmap.pretrained_path"),
        img_size=size,
        patch_size=int(cfg_get(cfg, "pointmap.patch_size", 16)),
        embed_dim=int(cfg_get(cfg, "pointmap.embed_dim", 128)),
        encoder_depth=int(cfg_get(cfg, "pointmap.encoder_depth", 4)),
        decoder_depth=int(cfg_get(cfg, "pointmap.decoder_depth", 2)),
        num_heads=int(cfg_get(cfg, "pointmap.num_heads", 4)),
        head_hidden=int(cfg_get(cfg, "pointmap.head_hidden", 128)),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train-pointmap] provenance: {provenance}; parameters {n_params/1e6:.2f} M")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    log_path = out_dir / "log.csv"
    history: list[dict[str, float]] = []
    t0 = time.perf_counter()
    first = None
    for epoch in range(1, epochs + 1):
        model.train()
        acc = {"total": 0.0, "regression": 0.0, "mean_conf": 0.0}
        steps = 0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            out = model(batch["img1"], batch["img2"])
            losses = pointmap_loss(out, batch, alpha=alpha)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            for k in acc:
                acc[k] += float(losses[k].detach())
            steps += 1
        sched.step()
        train = {k: v / max(1, steps) for k, v in acc.items()}

        model.eval()
        vacc = {"total": 0.0, "regression": 0.0, "mean_conf": 0.0}
        vsteps = 0
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch["img1"], batch["img2"])
                losses = pointmap_loss(out, batch, alpha=alpha)
                for k in vacc:
                    vacc[k] += float(losses[k].detach())
                vsteps += 1
        val = {k: v / max(1, vsteps) for k, v in vacc.items()}
        if first is None:
            # the confidence-weighted L1 is the meaningful regression number;
            # `total` also contains the -alpha*log(conf) term, which is unbounded
            # below and can legitimately go negative when the fit is good
            first = train["regression"]
        row = {
            "epoch": float(epoch),
            "train_total": train["total"],
            "train_regression": train["regression"],
            "train_mean_conf": train["mean_conf"],
            "val_total": val["total"],
            "val_regression": val["regression"],
            "lr": float(opt.param_groups[0]["lr"]),
            "steps": float(steps),
            "elapsed_s": time.perf_counter() - t0,
        }
        history.append(row)
        print(
            f"[train-pointmap] epoch {epoch}/{epochs}: train {row['train_total']:.5f} "
            f"(reg {row['train_regression']:.5f}, conf {row['train_mean_conf']:.3f}) "
            f"| val {row['val_total']:.5f} | {row['elapsed_s']:.0f}s"
        )
        _write_log(log_path, history)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "img_size": list(size),
            "patch_size": model.patch_size,
            "embed_dim": model.embed_dim,
            "provenance": provenance,
        },
        out_dir / "pointmap.pt",
    )
    last = history[-1]
    reduction = 100.0 * (first - last["train_regression"]) / max(1e-9, first)
    summary = {
        "step": "train-pointmap",
        "sequence": train_seq.name,
        "val_sequence": val_seq.name,
        "data_source": train_seq.source,
        "provenance": provenance,
        "resolution_hw": list(size),
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "conf_reg_alpha": alpha,
        "num_parameters_millions": n_params / 1e6,
        "num_train_pairs": len(train_ds),
        "num_val_pairs": len(val_ds),
        "loss_first_epoch": first,
        "loss_last_epoch": last["train_total"],
        "loss_reduction_pct": reduction,
        "val_total_last": last["val_total"],
        "training_seconds": time.perf_counter() - t0,
        "history": history,
        "checkpoint": str(out_dir / "pointmap.pt"),
    }
    save_json(out_dir / "summary.json", summary)
    print(
        f"[train-pointmap] confidence-weighted L1 {first:.5f} -> {last['train_regression']:.5f} "
        f"({reduction:.1f}% reduction); total objective {last['train_total']:.5f} "
        f"(negative is expected: the -alpha*log(conf) term is unbounded below) "
        f"in {summary['training_seconds']:.0f}s"
    )
    print(f"[train-pointmap] saved {out_dir/'pointmap.pt'} and {log_path}")
    return 0


def _write_log(path: Path, history: list[dict[str, float]]) -> None:
    """Rewrite the CSV loss log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
