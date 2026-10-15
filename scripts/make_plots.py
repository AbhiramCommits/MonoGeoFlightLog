#!/usr/bin/env python
"""Make the figures in ``results/figures/`` from the JSON results and checkpoints.

Four figures, all driven by measured data rather than hardcoded numbers:

``depth_error_bars.png``   per-method depth error with and without median scaling
``corruption_curves.png``  metric vs corruption severity, one line per method
``trajectory.png``         ground-truth camera path vs each method after alignment
``qualitative_strip.png``  frame / predicted depth / pointmap depth / error map

Usage::

    python scripts/make_plots.py [--results results] [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from monogeo.cli import cfg_get, load_config
from monogeo.data import load_sequence, slice_sequence, splits
from monogeo.metrics import align_trajectory
from monogeo.pointmap_ops import median_scale

METHOD_STYLE = {
    "classical_two_view": ("#1f77b4", "classical two-view"),
    "self_supervised_monodepth": ("#d62728", "self-supervised monodepth"),
    "pointmap": ("#2ca02c", "pointmap"),
    "classical_two_view_gap2": ("#7f7f7f", "classical (gap 2)"),
}


def _read(path: Path) -> dict | None:
    """Read JSON if present."""
    if not path.exists():
        print(f"[make_plots] skipping missing {path}")
        return None
    return json.loads(path.read_text())


def _unflatten(rows: list) -> np.ndarray:
    """List of 16-element rows -> ``(N, 4, 4)`` array."""
    return np.asarray(rows, np.float64).reshape(-1, 4, 4)


def plot_depth_error_bars(metrics: dict, out: Path) -> None:
    """Grouped bars of AbsRel / RMSE / delta per method."""
    methods = list(metrics["methods"].keys())
    labels = [METHOD_STYLE.get(m, (None, m))[1] for m in methods]
    absrel_raw = [metrics["methods"][m]["depth_and_pair_pose"]["abs_rel"] for m in methods]
    absrel_scaled = [
        metrics["methods"][m]["depth_and_pair_pose"]["abs_rel_median_scaled"] for m in methods
    ]
    delta_scaled = [
        metrics["methods"][m]["depth_and_pair_pose"]["delta1_median_scaled"] for m in methods
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(methods))
    w = 0.38
    axes[0].bar(x - w / 2, absrel_raw, w, label="raw (no alignment)", color="#9ecae1")
    axes[0].bar(x + w / 2, absrel_scaled, w, label="after median scaling", color="#3182bd")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=18, ha="right")
    axes[0].set_ylabel("AbsRel (lower is better)")
    axes[0].set_title("Depth error per method")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, delta_scaled, 0.5, color="#31a354")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=18, ha="right")
    axes[1].set_ylabel(r"$\delta < 1.25$ (higher is better)")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title(r"Accuracy $\delta<1.25$ after median scaling")
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[make_plots] wrote {out}")


def plot_corruption_curves(stress: dict, out: Path) -> None:
    """Metric vs severity, one line per method, one panel per corruption."""
    corruptions = stress["corruptions_tested"]
    severities = stress["severities_tested"]
    methods = list(stress["clean"].keys())
    if not corruptions:
        return
    fig, axes = plt.subplots(
        1, len(corruptions), figsize=(3.6 * len(corruptions), 3.6), sharey=True
    )
    if len(corruptions) == 1:
        axes = [axes]
    for ax, cname in zip(axes, corruptions, strict=False):
        for method in methods:
            color, label = METHOD_STYLE.get(method, (None, method))
            xs, ys = [], []
            for sev in severities:
                block = stress["corrupted"].get(f"{cname}_sev{sev}")
                if not block or method not in block["methods"]:
                    continue
                xs.append(sev)
                ys.append(block["methods"][method]["metrics"]["abs_rel_median_scaled"])
            if xs:
                ax.plot(xs, ys, "o-", color=color, label=label)
        # clean reference
        for method in methods:
            color, _ = METHOD_STYLE.get(method, (None, method))
            ax.axhline(
                stress["clean"][method]["abs_rel_median_scaled"],
                color=color,
                ls=":",
                lw=0.8,
                alpha=0.6,
            )
        ax.set_title(cname)
        ax.set_xlabel("severity")
        ax.set_xticks(list(severities))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("AbsRel after median scaling")
    axes[-1].legend(fontsize=7, loc="upper left")
    fig.suptitle("Depth error vs corruption severity (dotted = clean reference)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[make_plots] wrote {out}")


def plot_trajectory(metrics: dict, out: Path) -> None:
    """Ground-truth camera path vs each method after Umeyama alignment."""
    gt = _unflatten(metrics["ground_truth_trajectory"])
    est_all = metrics.get("estimated_trajectories", {})
    if gt.size == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    gt_c = np.stack([np.linalg.inv(p)[:3, 3] for p in gt])
    for name, rows in est_all.items():
        if not rows:
            continue
        est = _unflatten(rows)
        n = min(len(est), len(gt))
        try:
            aligned, scale, _, _ = align_trajectory(est[:n], gt[:n])
        except ValueError:
            continue
        c = np.stack([np.linalg.inv(p)[:3, 3] for p in aligned])
        color, label = METHOD_STYLE.get(name, (None, name))
        axes[0].plot(c[:, 0], c[:, 2], "-", color=color, label=f"{label} (s={scale:.2f})", lw=1.4)
        axes[1].plot(c[:, 1], c[:, 2], "-", color=color, lw=1.4)
    axes[0].plot(gt_c[:, 0], gt_c[:, 2], "k--", label="ground truth", lw=1.8)
    axes[1].plot(gt_c[:, 1], gt_c[:, 2], "k--", lw=1.8)
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("z (m)")
    axes[0].set_title("Top view (x-z)")
    axes[1].set_xlabel("y (m)")
    axes[1].set_title("Side view (y-z)")
    for ax in axes:
        ax.axis("equal")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Composed trajectory per method, aligned to ground truth by Umeyama", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[make_plots] wrote {out}")


def plot_qualitative_strip(cfg: dict, out: Path, index: int | None = None) -> None:
    """Frame / classical depth / monodepth / pointmap depth / GT / error map."""
    import torch

    from monogeo.compare import load_depth_model, load_pointmap_model
    from monogeo.pointmap_ops import depth_from_pointmap
    from monogeo.twoview import (
        depth_from_triangulation,
        estimate_pose,
        match_features,
        scale_from_gt,
        triangulate,
    )

    size = (int(cfg_get(cfg, "data.height", 128)), int(cfg_get(cfg, "data.width", 416)))
    full = load_sequence(
        cfg_get(cfg, "data.sequence", "forward"),
        size=size,
        num_frames=int(cfg_get(cfg, "data.num_frames", 120)),
        root=cfg_get(cfg, "data.data_root", "data"),
        fov_deg=float(cfg_get(cfg, "synth.fov_deg", 62.0)),
        distortion=cfg_get(cfg, "synth.distortion", None),
        verbose=False,
    )
    _, held = splits(full, float(cfg_get(cfg, "eval.held_out_fraction", 0.7)))
    seq = slice_sequence(full, held[0], held[-1] + 1)
    i = int(index) if index is not None else len(seq) // 2
    j = min(i + 1, len(seq) - 1)
    gt = seq.depths[i]
    mask = (gt > float(cfg_get(cfg, "eval.min_depth", 1.0))) & (
        gt < float(cfg_get(cfg, "eval.max_depth", 120.0))
    )

    depth_net = load_depth_model(cfg)
    pm_net = load_pointmap_model(cfg)[0]
    with torch.no_grad():
        img = torch.from_numpy(seq.frames[i].astype(np.float32) / 255).permute(2, 0, 1)[None]
        _, d_mono = depth_net(img)
        out_pm = pm_net(
            img, torch.from_numpy(seq.frames[j].astype(np.float32) / 255).permute(2, 0, 1)[None]
        )
    d_mono = d_mono[0, 0].numpy()
    pm_depth = depth_from_pointmap(out_pm["pts1"][0].numpy())

    matches = match_features(seq.frames[i], seq.frames[j], max_features=4000)
    d_tri = np.zeros_like(gt)
    if len(matches) >= 8:
        tv = estimate_pose(matches, seq.K)
        pts3d, valid = triangulate(matches, seq.K, tv.R, tv.t, mask=tv.inlier_mask)
        if valid.sum() >= 3:
            sc = scale_from_gt(pts3d, gt, points1=matches.points1[valid])
            d_tri = depth_from_triangulation(pts3d * sc["scale"], matches.points1[valid], gt.shape)

    s_mono = median_scale(d_mono, gt, mask)
    s_pm = median_scale(pm_depth, gt, mask)
    panels = [
        ("frame", None),
        ("classical triangulation\n(metric, sparse)", d_tri),
        ("self-supervised monodepth\n(median-scaled)", d_mono * s_mono),
        ("pointmap\n(median-scaled)", pm_depth * s_pm),
        ("ground truth", gt),
        ("pointmap rel. error\n|pred-gt|/gt", None),
    ]
    vmin, vmax = float(np.percentile(gt[mask], 2)), float(np.percentile(gt[mask], 98))
    err = np.abs(pm_depth * s_pm - gt) / np.maximum(gt, 1e-6)
    err[~mask] = np.nan

    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 2.6))
    for ax, (title, data) in zip(axes, panels, strict=False):
        if data is None and title == "frame":
            ax.imshow(seq.frames[i])
        elif title.startswith("pointmap rel. error"):
            im = ax.imshow(np.clip(err, 0, 0.5), cmap="magma", vmin=0, vmax=0.5)
            plt.colorbar(im, ax=ax, fraction=0.046)
        else:
            show = data if data is not None else np.zeros_like(gt)
            show = np.where(show > 0, show, np.nan)
            im = ax.imshow(show, cmap="turbo", vmin=vmin, vmax=vmax)
            if title == "ground truth":
                plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"Frame {i} of held-out `{seq.name}` — depth in metres (turbo), error as relative error",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[make_plots] wrote {out}")


def main(argv: list[str] | None = None) -> int:
    """Entry point: build all figures that the available results support."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--figures", default=None)
    args = ap.parse_args(argv)

    results_dir = Path(args.results)
    figures_dir = Path(args.figures) if args.figures else results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)

    metrics = _read(results_dir / "metrics.json")
    stress = _read(results_dir / "stress_tests.json")

    if metrics:
        plot_depth_error_bars(metrics, figures_dir / "depth_error_bars.png")
        plot_trajectory(metrics, figures_dir / "trajectory.png")
    if stress:
        plot_corruption_curves(stress, figures_dir / "corruption_curves.png")
    try:
        plot_qualitative_strip(cfg, figures_dir / "qualitative_strip.png")
    except FileNotFoundError as exc:
        print(f"[make_plots] skipping qualitative strip: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
