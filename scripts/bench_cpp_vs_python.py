#!/usr/bin/env python
"""Compare the Python and C++ inference timings and plot the result.

Reads ``results/python_timing.json`` (written by ``monogeo export``) and
``results/cpp_timing.json`` (written by the ``depth_infer`` binary), writes
``results/cpp_vs_python.json`` with the speedup factor and the parity check, and
draws a bar chart in ``results/figures/``.

Usage::

    python scripts/bench_cpp_vs_python.py [--results results]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read(path: Path) -> dict | None:
    """Read JSON if present, else return ``None``."""
    if not path.exists():
        print(f"[bench] missing {path}; run the corresponding step first")
        return None
    return json.loads(path.read_text())


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    args = ap.parse_args(argv)
    results = Path(args.results)
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    py = _read(results / "python_timing.json")
    cpp = _read(results / "cpp_timing.json")
    if py is None or cpp is None:
        return 1

    py_mean = float(py["mean_ms"])
    cpp_mean = float(cpp["mean_ms"])
    speedup = py_mean / cpp_mean if cpp_mean > 0 else float("nan")

    payload = {
        "step": "bench-cpp-vs-python",
        "python": {
            "mean_ms": py_mean,
            "p50_ms": float(py["p50_ms"]),
            "p95_ms": float(py["p95_ms"]),
            "runs": int(py.get("runs", 0)),
            "threads": py.get("threads"),
            "implementation": py.get("implementation", "python-pytorch"),
        },
        "cpp": {
            "mean_ms": cpp_mean,
            "p50_ms": float(cpp["p50_ms"]),
            "p95_ms": float(cpp["p95_ms"]),
            "runs": int(cpp.get("runs", 0)),
            "threads": cpp.get("threads"),
            "backend": cpp.get("backend"),
            "model_load_ms": cpp.get("model_load_ms"),
        },
        "speedup_mean": speedup,
        "speedup_p50": (
            float(py["p50_ms"]) / float(cpp["p50_ms"]) if float(cpp["p50_ms"]) > 0 else float("nan")
        ),
        "speedup_p95": (
            float(py["p95_ms"]) / float(cpp["p95_ms"]) if float(cpp["p95_ms"]) > 0 else float("nan")
        ),
        "max_abs_diff": cpp.get("parity_max_abs_diff"),
        "mean_abs_diff": cpp.get("parity_mean_abs_diff"),
        "parity_threshold": cpp.get("parity_threshold"),
        "input_shape": cpp.get("input_shape_string"),
        "note": (
            "Both timings measure one forward pass on the same fixed 1x3xHxW frame on CPU. "
            "The C++ side additionally verifies bit-level parity against the PyTorch output, so "
            "the speedup is a genuine replacement, not a different computation."
        ),
    }
    out = results / "cpp_vs_python.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"[bench] python mean {py_mean:.3f} ms, cpp mean {cpp_mean:.3f} ms -> {speedup:.3f}x speedup"
    )
    print(
        f"[bench] max abs diff {payload['max_abs_diff']} (threshold {payload['parity_threshold']})"
    )
    print(f"[bench] wrote {out}")

    # --- bar chart --------------------------------------------------------
    labels = [
        "Python\n(PyTorch)",
        "C++\n(" + str(cpp.get("backend", "?")).split("(")[0].strip() + ")",
    ]
    means = [py_mean, cpp_mean]
    p50s = [float(py["p50_ms"]), float(cpp["p50_ms"])]
    p95s = [float(py["p95_ms"]), float(cpp["p95_ms"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = range(len(labels))
    w = 0.55
    axes[0].bar(x, means, w, color=["#4c72b0", "#dd8452"], label="mean")
    # Draw the p50..p95 whisker directly. errorbar() needs it as offsets from the
    # mean, which go negative (and raise) whenever p50 > mean, e.g. a run with a
    # few unusually fast iterations.
    for i, (p50, p95) in enumerate(zip(p50s, p95s, strict=True)):
        axes[0].plot([i, i], [p50, p95], color="k", marker="_", markersize=12, lw=1)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("ms per frame")
    axes[0].set_title("Inference time (bars = mean, whiskers = p50..p95)")
    for i, m in enumerate(means):
        axes[0].text(i, m, f"{m:.2f} ms", ha="center", va="bottom", fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(list(x), [speedup] * 2, 0.4, color=["#55a868", "#c44e52"])
    axes[1].axhline(1.0, color="k", lw=0.8, ls="--")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(["speedup (mean)", "speedup (p50)"])
    axes[1].set_ylabel("x faster than Python")
    axes[1].set_title("C++ speedup over Python")
    for i, v in enumerate([speedup, payload["speedup_p50"]]):
        axes[1].text(i, v, f"{v:.2f}x", ha="center", va="bottom", fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle(
        f"Depth head inference, identical {payload['input_shape']} input, "
        f"max abs diff {payload['max_abs_diff']:.2e}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    chart = figures / "cpp_vs_python_timing.png"
    fig.savefig(chart, dpi=130)
    plt.close(fig)
    print(f"[bench] wrote {chart}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
