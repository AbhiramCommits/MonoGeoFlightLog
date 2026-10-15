"""Assemble ``results/RESULTS.md`` from the JSON files in ``results/``.

Every number in the generated report is read from a results file that some
other command wrote. Nothing is typed in by hand, so the report cannot drift
away from the measurements. Missing result files are reported as such and their
sections are skipped, which lets the report be generated at any point in the
pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["build_report", "run_report"]


def _read(path: Path) -> dict[str, Any] | None:
    """Read a JSON file, returning ``None`` if it is absent or unreadable."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:  # pragma: no cover - corrupt file
        print(f"[report] could not parse {path}: {exc}")
        return None


def _f(value: Any, digits: int = 4, dash: str = "n/a") -> str:
    """Format a number for a Markdown table cell."""
    if value is None:
        return dash
    try:
        v = float(value)
    except (TypeError, ValueError):
        return dash
    if v != v:  # NaN
        return dash
    if v == 0:
        return "0"
    if abs(v) >= 1e5 or (abs(v) < 1e-3 and v != 0):
        return f"{v:.{digits}e}"
    return f"{v:.{digits}f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a Markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _section_calibration(res: dict[str, Any]) -> str:
    """Calibration table: estimated vs true intrinsics and distortion."""
    rep = res["report"]
    intr = rep["intrinsics"]
    out = ["### Calibration\n"]
    rows = []
    for name in ("fx", "fy", "cx", "cy"):
        rows.append(
            [
                name,
                _f(intr["true"][name], 3),
                _f(intr["estimated"][name], 3),
                _f(intr["abs_error_px"][name], 3),
                _f(intr["rel_error_pct"][name], 3),
            ]
        )
    out.append(
        _table(["param", "true (px)", "estimated (px)", "abs error (px)", "rel error (%)"], rows)
    )
    d = rep["distortion"]
    rows = [
        [k, _f(d["true"][k], 5), _f(d["estimated"][k], 5), _f(d["abs_error"][k], 5)]
        for k in ("k1", "k2", "p1", "p2", "k3")
    ]
    out.append("\n")
    out.append(_table(["coeff", "true", "estimated", "abs error"], rows))
    disp = rep.get("distortion_field_displacement_px", {})
    cb = res.get("checkerboard", {})
    out.append(
        f"\n- RMS reprojection error: **{_f(res.get('rms_reprojection_px'), 4)} px** "
        f"over {cb.get('views_used', '?')}/{cb.get('views_offered', '?')} detected views\n"
    )
    out.append(
        f"- distortion *field* displacement vs truth: mean {_f(disp.get('mean'), 4)} px, "
        f"max {_f(disp.get('max'), 4)} px\n"
    )
    out.append(
        f"- checkerboard: {cb.get('inner_corners_rows_cols')} inner corners, "
        f"{_f(cb.get('square_size_m'), 3)} m squares, images {res.get('image_size_wh')} (w,h)\n"
    )
    if "distortion_field_note" in rep:
        out.append(f"\n> {rep['distortion_field_note']}\n")
    return "".join(out)


def _section_twoview(res: dict[str, Any], gap1: dict[str, Any] | None) -> str:
    """Classical two-view baseline table plus the short-baseline reference."""
    agg = res["aggregate_over_pairs"]
    rows = [
        ["matches (median)", _f(agg.get("n_matches_median"), 1)],
        ["inliers (median)", _f(agg.get("n_inliers_median"), 1)],
        ["inlier ratio (median)", _f(agg.get("inlier_ratio_median"), 4)],
        ["median reprojection error (px)", _f(agg.get("reprojection_median_px"), 4)],
        ["RMS reprojection error (px)", _f(agg.get("reprojection_rms_px"), 4)],
        ["rotation error (deg)", _f(agg.get("rotation_error_deg_median"), 4)],
        ["translation angular error (deg)", _f(agg.get("translation_angle_error_deg_median"), 4)],
        ["recovered scale factor", _f(agg.get("scale_from_gt_median"), 4)],
        ["true baseline (m)", _f(agg.get("true_baseline_m_median"), 4)],
        ["runtime (ms/pair)", _f(agg.get("runtime_ms_median"), 2)],
    ]
    out = [
        "### Classical two-view baseline\n",
        f"Evaluated on {agg.get('num_pairs')}/{agg.get('num_pairs_attempted')} frame pairs of the "
        f"`{res.get('sequence')}` sequence at {res.get('resolution_hw')} (h,w), "
        f"frame gap {res.get('frame_gap')}. `matches`/`inliers` are medians over pairs.\n\n",
        _table(["quantity", "value"], rows),
    ]
    if gap1 is not None:
        g = gap1["aggregate_over_pairs"]
        out.append(
            "\nSame pipeline with a **one-frame** baseline (the gap the learned models are "
            "trained on), for comparison:\n\n"
        )
        out.append(
            _table(
                ["quantity", "gap 1"],
                [
                    ["matches (median)", _f(g.get("n_matches_median"), 1)],
                    ["inlier ratio (median)", _f(g.get("inlier_ratio_median"), 4)],
                    ["median reprojection error (px)", _f(g.get("reprojection_median_px"), 4)],
                    ["rotation error (deg)", _f(g.get("rotation_error_deg_median"), 4)],
                    [
                        "translation angular error (deg)",
                        _f(g.get("translation_angle_error_deg_median"), 4),
                    ],
                    ["recovered scale factor", _f(g.get("scale_from_gt_median"), 4)],
                    ["true baseline (m)", _f(g.get("true_baseline_m_median"), 4)],
                ],
            )
        )
    return "".join(out)


def _section_detph(eval_res: dict[str, Any]) -> str:
    """Depth accuracy per method, with and without alignment."""
    rows = []
    for method, m in eval_res["methods"].items():
        h = m["depth_and_pair_pose"]
        rows.append(
            [
                f"`{method}`",
                _f(h["abs_rel"], 3),
                _f(h["sq_rel_median_scaled"], 3),
                _f(h["rmse_median_scaled"], 2),
                _f(h["rmse_log_median_scaled"], 3),
                _f(h["delta1_median_scaled"], 3),
                _f(h["abs_rel_median_scaled"], 3),
                _f(h["delta1"], 3),
            ]
        )
    return (
        "### Depth accuracy per method\n"
        f"Held-out frames {eval_res.get('held_out_frames')} of `{eval_res.get('parent_sequence')}`, "
        f"{eval_res.get('num_frames')} frames at {eval_res.get('resolution_hw')} (h,w), "
        f"depth range {eval_res.get('metric_range_m')} m, gap 1. All metrics are medians over "
        f"pairs; `delta<1.25` values are fractions (higher is better), the rest are errors "
        f"(lower is better).\n\n"
        + _table(
            [
                "method",
                "AbsRel (raw)",
                "SqRel (scaled)",
                "RMSE (scaled)",
                "RMSE-log (scaled)",
                "d<1.25 (scaled)",
                "AbsRel (scaled)",
                "d<1.25 (raw)",
            ],
            rows,
        )
        + "\n*raw* = the network/triangulation output as-is (up to an unknown scale); "
        "*scaled* = after a single per-pair median scale factor. These are the standard "
        "Eigen et al. metrics.\n"
    )


def _section_pose(eval_res: dict[str, Any]) -> str:
    """Pose accuracy: per-pair errors plus ATE/RPE over the composed trajectory."""
    rows = []
    for method, m in eval_res["methods"].items():
        h = m["depth_and_pair_pose"]
        rows.append(
            [
                f"`{method}`",
                _f(h["rotation_deg"], 3),
                _f(h["translation_deg"], 3),
                _f(h.get("ate", {}).get("rmse"), 3),
                _f(h.get("rpe", {}).get("translation_rmse"), 4),
                _f(h.get("rpe", {}).get("rotation_rmse_deg"), 3),
            ]
        )
    return (
        "### Pose accuracy per method\n"
        "Per-pair errors are rotation (deg) and translation *direction* (deg). ATE and RPE are "
        "computed over the trajectory each method composes from its own per-pair transforms, "
        "after a 7-DoF Umeyama alignment of the camera centres "
        "(a monocular trajectory has no observable scale or global frame).\n\n"
        + _table(
            [
                "method",
                "rotation (deg)",
                "translation dir (deg)",
                "ATE RMSE (m)",
                "RPE trans (m/frame)",
                "RPE rot (deg/frame)",
            ],
            rows,
        )
    )


def _section_scale(compare_res: dict[str, Any]) -> str:
    """Scale analysis: factors, drift, and what alignment costs."""
    rows = []
    for method, m in compare_res["scale_analysis"]["per_method"].items():
        rows.append(
            [
                f"`{method}`",
                _f(m["scale_factor_from_median_ratio"], 3),
                _f(m["scale_drift_std_over_mean"], 3),
                _f(m["metrics_no_alignment"]["abs_rel"], 3),
                _f(m["metrics_with_median_scaling"]["abs_rel"], 3),
                _f(m["metrics_with_umeyama_alignment"]["abs_rel"], 3),
                _f(m["metrics_no_alignment"]["delta1"], 3),
                _f(m["metrics_with_median_scaling"]["delta1"], 3),
            ]
        )
    focal = compare_res.get("focal_analysis", {})
    frows = []
    for method, f in focal.items():
        frows.append(
            [
                f"`{method}`",
                _f(f.get("focal_used_by_pipeline"), 2),
                _f(f.get("focal_estimated_from_pointmap"), 2),
                _f(f.get("abs_error_px"), 2),
                _f(f.get("rel_error_pct"), 2),
            ]
        )
    out = [
        "### Scale analysis\n",
        compare_res["scale_analysis"]["note"],
        "\n\n",
        _table(
            [
                "method",
                "scale factor",
                "drift (std/mean)",
                "AbsRel (none)",
                "AbsRel (median)",
                "AbsRel (umeyama)",
                "d<1.25 (none)",
                "d<1.25 (median)",
            ],
            rows,
        ),
        "\n### Estimated vs calibrated focal length\n\n",
        _table(
            [
                "method",
                "focal used (px)",
                "focal estimated (px)",
                "abs error (px)",
                "rel error (%)",
            ],
            frows,
        ),
        "\nThe classical and self-supervised methods *consume* the calibrated focal length; only "
        "the pointmap model estimates it, from the pointmap geometry alone.\n",
    ]
    return "".join(out)


def _section_stress(stress_res: dict[str, Any]) -> str:
    """Degradation tables for corruptions and domain shifts."""
    methods = list(stress_res["clean"].keys())
    corr_tags = list(stress_res["corrupted"].keys())
    rows = []
    for method in methods:
        row = [f"`{method}`"]
        for tag in corr_tags:
            block = stress_res["corrupted"][tag]["methods"].get(method, {})
            row.append(_f(block.get("degradation_pct", {}).get("abs_rel_median_scaled"), 1))
        rows.append(row)
    out = [
        "### Stress tests\n",
        "Degradation of median-scaled AbsRel (%) relative to the clean held-out evaluation. "
        "**Positive always means worse.**\n\n",
        _table(["method", *corr_tags], rows),
    ]
    drows = []
    for method in methods:
        row = [f"`{method}`"]
        for label in stress_res["domain_shift"]:
            block = stress_res["domain_shift"][label]["methods"].get(method, {})
            row.append(_f(block.get("degradation_pct", {}).get("abs_rel_median_scaled"), 1))
        drows.append(row)
    out.append("\nDomain shift (no image corruption, just a different flight regime):\n\n")
    out.append(_table(["method", *list(stress_res["domain_shift"].keys())], drows))
    out.append("\n")
    rows = []
    for tag, block in stress_res["corrupted"].items():
        for method, m in block["methods"].items():
            rows.append(
                [
                    tag,
                    f"`{method}`",
                    _f(m["metrics"]["abs_rel_median_scaled"], 3),
                    _f(m["metrics"]["delta1_median_scaled"], 3),
                    _f(m["metrics"]["rotation_deg"], 2),
                    _f(m["degradation_pct"]["abs_rel_median_scaled"], 1),
                ]
            )
    out.append("\nAbsolute values per corruption and severity:\n\n")
    out.append(
        _table(
            [
                "variant",
                "method",
                "AbsRel (scaled)",
                "d<1.25 (scaled)",
                "rotation (deg)",
                "AbsRel degradation (%)",
            ],
            rows,
        )
    )
    return "".join(out)


def _section_timing(
    py: dict[str, Any] | None, cpp: dict[str, Any] | None, cmp_res: dict[str, Any] | None
) -> str:
    """C++ vs Python inference timing and parity."""
    if py is None and cpp is None:
        return "### C++ vs Python inference\n\n_No timing results present._\n"
    rows = []
    if py is not None:
        rows.append(
            [
                "Python (PyTorch)",
                _f(py.get("mean_ms"), 3),
                _f(py.get("p50_ms"), 3),
                _f(py.get("p95_ms"), 3),
                f"{py.get('runs', '?')} runs, torch threads {py.get('threads', '?')}",
            ]
        )
    if cpp is not None:
        rows.append(
            [
                "C++ (backend below)",
                _f(cpp.get("mean_ms"), 3),
                _f(cpp.get("p50_ms"), 3),
                _f(cpp.get("p95_ms"), 3),
                f"{cpp.get('runs', '?')} runs, backend {cpp.get('backend', '?')}"
                f", threads {cpp.get('threads', '?')}",
            ]
        )
    out = [
        "### C++ vs Python inference\n\n",
        _table(["implementation", "mean (ms)", "p50 (ms)", "p95 (ms)", "notes"], rows),
    ]
    if py is not None and isinstance(py.get("onnxruntime"), dict):
        ort = py["onnxruntime"]
        out.append(
            f"\n- ONNX Runtime on the same graph: mean {_f(ort.get('mean_ms'), 3)} ms, "
            f"p50 {_f(ort.get('p50_ms'), 3)} ms, p95 {_f(ort.get('p95_ms'), 3)} ms\n"
        )
    if cmp_res is not None:
        out.append(
            f"\n- speedup (Python mean / C++ mean): **{_f(cmp_res.get('speedup_mean'), 3)}x**\n"
            f"- max abs output difference: **{_f(cmp_res.get('max_abs_diff'), 8)}** "
            f"(parity threshold {_f(cmp_res.get('parity_threshold'), 6)})\n"
            f"- mean abs output difference: {_f(cmp_res.get('mean_abs_diff'), 8)}\n"
        )
    return "".join(out)


def _section_learning(depth_sum: dict[str, Any] | None, pm_sum: dict[str, Any] | None) -> str:
    """Training summaries, so the README can quote the convergence."""
    rows = []
    if depth_sum is not None:
        rows.append(
            [
                "self-supervised depth + pose",
                f"{depth_sum.get('epochs')} epochs, batch {depth_sum.get('batch_size')}, "
                f"lr {_f(depth_sum.get('lr'), 6)}",
                _f(depth_sum.get("photometric_first_epoch"), 4),
                _f(depth_sum.get("photometric_last_epoch"), 4),
                _f(depth_sum.get("photometric_reduction_pct"), 1) + " %",
                _f(depth_sum.get("training_seconds"), 0) + " s",
            ]
        )
    if pm_sum is not None:
        rows.append(
            [
                "two-view pointmap",
                f"{pm_sum.get('epochs')} epochs, batch {pm_sum.get('batch_size')}, "
                f"lr {_f(pm_sum.get('lr'), 6)}",
                _f(pm_sum.get("loss_first_epoch"), 4),
                _f(pm_sum.get("loss_last_epoch"), 4),
                _f(pm_sum.get("loss_reduction_pct"), 1) + " %",
                _f(pm_sum.get("training_seconds"), 0) + " s",
            ]
        )
    if not rows:
        return ""
    return (
        "### Training convergence\n\n"
        + _table(
            [
                "model",
                "schedule",
                "loss (first epoch)",
                "loss (last epoch)",
                "reduction",
                "wall time",
            ],
            rows,
        )
        + "\nThe pointmap number is its confidence-weighted L1 regression term; the full DUSt3R "
        "objective also contains a `-alpha*log(conf)` term which is unbounded below and can go "
        "negative once the fit is good.\n"
    )


def build_report(results_dir: Path, figures_dir: Path) -> str:
    """Build the Markdown report body from the results directory.

    Args:
        results_dir: Directory holding the ``*.json`` result files.
        figures_dir: Directory holding the ``*.png`` figures.

    Returns:
        The Markdown document as a string.
    """
    calib = _read(results_dir / "calibration.json")
    twoview = _read(results_dir / "two_view_baseline.json")
    twoview_gap1 = _read(results_dir / "two_view_baseline_gap1.json")
    comparison = _read(results_dir / "method_comparison.json")
    metrics = _read(results_dir / "metrics.json")
    stress = _read(results_dir / "stress_tests.json")
    py_timing = _read(results_dir / "python_timing.json")
    cpp_timing = _read(results_dir / "cpp_timing.json")
    cpp_cmp = _read(results_dir / "cpp_vs_python.json")
    depth_sum = _read(results_dir.parent / "runs" / "depth" / "summary.json")
    pm_sum = _read(results_dir.parent / "runs" / "pointmap" / "summary.json")

    from . import __version__

    out = [
        "# MonoGeoFlightLog — measured results\n\n",
        f"Generated by `monogeo report` (package version {__version__}). Every value below is read "
        "from a JSON file in this directory; nothing is typed by hand. Regenerate with "
        "`bash scripts/reproduce.sh`.\n\n",
    ]

    if calib:
        src = calib.get("data_source", "synthetic")
        out.append(f"All numbers in this report were produced on the **{src}** renderer.\n\n")
        out.append(_section_calibration(calib) + "\n")
    if twoview:
        out.append(_section_twoview(twoview, twoview_gap1) + "\n")
    if metrics:
        out.append(_section_detph(metrics) + "\n")
        out.append(_section_pose(metrics) + "\n")
    if comparison:
        out.append(_section_scale(comparison) + "\n")
    if stress:
        out.append(_section_stress(stress) + "\n")
    out.append(_section_timing(py_timing, cpp_timing, cpp_cmp) + "\n")
    learn = _section_learning(depth_sum, pm_sum)
    if learn:
        out.append(learn + "\n")

    figs = sorted(figures_dir.glob("*.png")) if figures_dir.exists() else []
    if figs:
        out.append("### Figures\n\n")
        for fig in figs:
            title = fig.stem.replace("_", " ")
            out.append(f"![{title}](figures/{fig.name})\n\n")

    missing = [
        p
        for p in (
            "calibration.json",
            "two_view_baseline.json",
            "method_comparison.json",
            "metrics.json",
            "stress_tests.json",
        )
        if not (results_dir / p).exists()
    ]
    if missing:
        out.append(f"\n_Missing result files at build time: {', '.join(missing)}._\n")
    return "".join(out)


def run_report(args: Any) -> int:
    """``monogeo report``: write ``results/RESULTS.md``.

    Args:
        args: Parsed argparse namespace.

    Returns:
        Process exit code (0 on success).
    """
    out_path = Path(getattr(args, "out", "results/RESULTS.md"))
    results_dir = out_path.parent
    figures_dir = results_dir / "figures"
    text = build_report(results_dir, figures_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(f"[report] wrote {out_path} ({len(text)} bytes)")
    return 0
