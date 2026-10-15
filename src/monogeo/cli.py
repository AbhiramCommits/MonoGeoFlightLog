"""Command line interface for MonoGeoFlightLog.

Every stage of the pipeline is reachable as ``monogeo <subcommand>``. The
subcommands are registered incrementally as the pipeline is built; each one
delegates to a module-level ``run_*`` function so the same code path is used by
``scripts/reproduce.sh`` and by the tests.

Usage::

    monogeo calibrate   --out results/calibration.json
    monogeo two-view    --sequence forward --out results/two_view_baseline.json
    monogeo train-depth --config configs/default.yaml
    monogeo train-pointmap --config configs/default.yaml
    monogeo compare     --out results/method_comparison.json
    monogeo evaluate    --out results/metrics.json
    monogeo stress      --out results/stress_tests.json
    monogeo export      --out artifacts/depth_net.onnx
    monogeo report      --out results/RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

__all__ = ["build_parser", "load_config", "main", "save_json", "set_threads"]


# --------------------------------------------------------------------------- #
# small shared helpers
# --------------------------------------------------------------------------- #
def set_threads(num_threads: int | None) -> None:
    """Pin torch/BLAS thread counts so CPU timings are reproducible.

    Args:
        num_threads: Thread count, or ``None`` to leave the environment alone.
    """
    if num_threads is None:
        return
    import os

    n = int(num_threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    try:
        import torch

        torch.set_num_threads(n)
        torch.set_num_interop_threads(1)
    except Exception:  # pragma: no cover - interop threads can only be set once
        pass


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Read a YAML config, returning an empty dict when ``path`` is ``None``.

    Args:
        path: Path to a YAML file, or ``None``.

    Returns:
        The parsed configuration as a nested dictionary.
    """
    if path is None:
        return {}
    import yaml

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    with p.open() as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg


def cfg_get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Fetch ``cfg["a"]["b"]`` from the dotted key ``"a.b"``."""
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def save_json(path: str | Path, payload: Any) -> Path:
    """Write ``payload`` as pretty-printed JSON, creating parent directories.

    Args:
        path: Destination file.
        payload: JSON-serialisable object (numpy scalars/arrays are coerced).

    Returns:
        The written path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    return out


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy/torch objects into JSON-serialisable ones."""
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return _jsonable(obj.detach().cpu().numpy())
    except Exception:  # pragma: no cover
        pass
    if isinstance(obj, Path):
        return str(obj)
    return obj


# --------------------------------------------------------------------------- #
# subcommand handlers
# --------------------------------------------------------------------------- #
def _run_calibrate(args: argparse.Namespace) -> int:
    from .calib import run_calibrate

    return run_calibrate(args)


def _run_twoview(args: argparse.Namespace) -> int:
    from .twoview import run_two_view

    return run_two_view(args)


def _run_train_depth(args: argparse.Namespace) -> int:
    from .train_depth import run_train_depth

    return run_train_depth(args)


def _run_train_pointmap(args: argparse.Namespace) -> int:
    from .train_pointmap import run_train_pointmap

    return run_train_pointmap(args)


def _run_compare(args: argparse.Namespace) -> int:
    from .compare import run_compare

    return run_compare(args)


def _run_evaluate(args: argparse.Namespace) -> int:
    from .evaluate import run_evaluate

    return run_evaluate(args)


def _run_stress(args: argparse.Namespace) -> int:
    from .evaluate import run_stress

    return run_stress(args)


def _run_export(args: argparse.Namespace) -> int:
    from .export import run_export

    return run_export(args)


def _run_report(args: argparse.Namespace) -> int:
    from .report import run_report

    return run_report(args)


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="configs/default.yaml", help="YAML config file")
    p.add_argument("--seed", type=int, default=None, help="override the config seed")
    p.add_argument(
        "--threads", type=int, default=None, help="pin CPU threads (default: config num_threads)"
    )
    p.add_argument("--verbose", action="store_true", help="print per-step progress")


def build_parser() -> argparse.ArgumentParser:
    """Construct the full argument parser for the ``monogeo`` entry point."""
    parser = argparse.ArgumentParser(
        prog="monogeo",
        description=(
            "Monocular geometry pipeline for aerial/driving flight logs: "
            "classical two-view, self-supervised depth + ego-motion, and "
            "end-to-end pointmap regression."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p = sub.add_parser("calibrate", help="checkerboard calibration + report")
    _add_common(p)
    p.add_argument("--out", default="results/calibration.json", help="output JSON path")
    p.add_argument("--views", type=int, default=None, help="number of calibration views")
    p.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    p.set_defaults(func=_run_calibrate)

    p = sub.add_parser("two-view", help="classical essential-matrix two-view baseline")
    _add_common(p)
    p.add_argument("--sequence", default=None, help="sequence name (default: config data.sequence)")
    p.add_argument("--out", default="results/two_view_baseline.json", help="output JSON path")
    p.add_argument("--pairs", type=int, default=None, help="number of frame pairs to average")
    p.add_argument("--gap", type=int, default=None, help="frame gap between the pair")
    p.set_defaults(func=_run_twoview)

    p = sub.add_parser("train-depth", help="train the self-supervised depth + pose networks")
    _add_common(p)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--out-dir", default=None)
    p.set_defaults(func=_run_train_depth)

    p = sub.add_parser("train-pointmap", help="train the two-view pointmap regressor")
    _add_common(p)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--out-dir", default=None)
    p.set_defaults(func=_run_train_pointmap)

    p = sub.add_parser("compare", help="compare all three methods on held-out data")
    _add_common(p)
    p.add_argument("--out", default="results/method_comparison.json")
    p.add_argument("--sequence", default=None)
    p.add_argument("--frames", type=int, default=None)
    p.set_defaults(func=_run_compare)

    p = sub.add_parser("evaluate", help="depth and pose metrics on clean held-out data")
    _add_common(p)
    p.add_argument("--out", default="results/metrics.json")
    p.add_argument("--sequence", default=None)
    p.add_argument("--frames", type=int, default=None)
    p.set_defaults(func=_run_evaluate)

    p = sub.add_parser("stress", help="corruption and domain-shift stress tests")
    _add_common(p)
    p.add_argument("--out", default="results/stress_tests.json")
    p.add_argument("--sequence", default=None)
    p.add_argument("--frames", type=int, default=None)
    p.set_defaults(func=_run_stress)

    p = sub.add_parser("export", help="export the depth head to ONNX and time Python inference")
    _add_common(p)
    p.add_argument("--out", default=None, help="ONNX path (default: config export.onnx_path)")
    p.add_argument("--runs", type=int, default=None)
    p.set_defaults(func=_run_export)

    p = sub.add_parser("report", help="emit results/RESULTS.md from the results/*.json files")
    _add_common(p)
    p.add_argument("--out", default="results/RESULTS.md")
    p.set_defaults(func=_run_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``monogeo`` console script.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    cfg = load_config(getattr(args, "config", None))
    args.cfg = cfg
    if args.seed is None:
        args.seed = int(cfg_get(cfg, "seed", 0))
    if args.threads is None:
        args.threads = cfg_get(cfg, "num_threads", None)
    set_threads(args.threads)
    try:
        return int(args.func(args) or 0)
    except NotImplementedError as exc:
        print(
            f"[monogeo] {args.command}: not available in this build step ({exc})", file=sys.stderr
        )
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
