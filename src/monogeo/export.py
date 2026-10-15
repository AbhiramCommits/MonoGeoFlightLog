"""ONNX export of the depth head, plus a Python timing baseline.

The exported graph takes a **fixed** ``1 x 3 x H x W`` float32 input in ``[0, 1]``
RGB and returns metric depth ``1 x 1 x H x W`` in metres. Emitting depth rather
than disparity keeps the C++ side trivial and makes the parity check meaningful:
both implementations must produce the same metres, not just the same activations.

Along with the ``.onnx`` file this step writes

* ``artifacts/sample_input.bin``  -- the exact float32 tensor fed to the model,
  so the C++ program can run on bit-identical input and the parity check tests
  the *inference*, not the preprocessing;
* ``artifacts/sample_frame.png``  -- the same frame as an image, so the C++
  program can also exercise its OpenCV preprocessing path;
* ``artifacts/python_output.bin`` -- the PyTorch output for that input;
* ``results/python_timing.json``  -- mean/p50/p95 ms over >= 50 runs with
  ``torch.no_grad()`` and threads pinned.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .cli import cfg_get, save_json
from .compare import load_depth_model
from .data import load_sequence, slice_sequence, splits

__all__ = ["DepthHeadExport", "percentiles", "run_export"]


class DepthHeadExport(nn.Module):
    """Wrapper that makes the depth head a clean single-input/single-output graph.

    ``DepthNet`` returns a list of multi-scale disparities plus a depth map; ONNX
    export of that is awkward and the C++ side only ever wants the metric depth.
    """

    def __init__(self, depth_net: nn.Module) -> None:
        """Wrap a trained :class:`monogeo.models.DepthNet`.

        Args:
            depth_net: The trained depth network.
        """
        super().__init__()
        self.depth_net = depth_net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return metric depth in metres.

        Args:
            x: ``(1, 3, H, W)`` float32 RGB in ``[0, 1]``.

        Returns:
            ``(1, 1, H, W)`` depth in metres.
        """
        _, depth = self.depth_net(x)
        return depth


def percentiles(values: list[float]) -> dict[str, float]:
    """Summarise a list of timings in milliseconds.

    Args:
        values: Per-run durations in milliseconds.

    Returns:
        Dict with ``mean``, ``p50``, ``p95``, ``min``, ``max`` and ``runs``.
    """
    if not values:
        return {
            "mean": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "runs": 0,
        }
    arr = np.asarray(values, np.float64)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "runs": int(arr.size),
    }


def run_export(args: Any) -> int:
    """``monogeo export``: write the ONNX graph, reference tensors and Python timings.

    Args:
        args: Parsed argparse namespace.

    Returns:
        Process exit code (0 on success, 1 if ONNX and PyTorch disagree).
    """
    cfg = getattr(args, "cfg", {}) or {}
    seed = int(getattr(args, "seed", 0) or 0)
    torch.manual_seed(seed)
    np.random.seed(seed)

    size = (int(cfg_get(cfg, "data.height", 128)), int(cfg_get(cfg, "data.width", 416)))
    height, width = size
    runs = int(getattr(args, "runs", None) or cfg_get(cfg, "export.runs", 50))
    warmup = int(cfg_get(cfg, "export.warmup", 10))
    opset = int(cfg_get(cfg, "export.onnx_opset", 17))
    onnx_path = Path(
        getattr(args, "out", None) or cfg_get(cfg, "export.onnx_path", "artifacts/depth_net.onnx")
    )
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    net = DepthHeadExport(load_depth_model(cfg))
    net.eval()

    # A real held-out frame is more representative than random noise, and it lets
    # the C++ side exercise its OpenCV preprocessing path on the same content.
    full = load_sequence(
        cfg_get(cfg, "data.sequence", "forward"),
        size=size,
        num_frames=int(cfg_get(cfg, "data.num_frames", 120)),
        root=cfg_get(cfg, "data.data_root", "data"),
        fov_deg=float(cfg_get(cfg, "synth.fov_deg", 62.0)),
        distortion=cfg_get(cfg, "synth.distortion", None),
        verbose=bool(getattr(args, "verbose", False)),
    )
    _, held = splits(full, float(cfg_get(cfg, "eval.held_out_fraction", 0.7)))
    slice_ = slice_sequence(full, held[0], held[-1] + 1)
    frame_index = len(slice_) // 2
    frame_hwc = np.ascontiguousarray(slice_.frames[frame_index])
    x = (
        torch.from_numpy(frame_hwc.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
    )
    print(
        f"[export] input {tuple(x.shape)} from held-out frame {held[0] + frame_index} of {full.name}"
    )

    import cv2

    cv2.imwrite(
        str(onnx_path.parent / "sample_frame.png"), cv2.cvtColor(frame_hwc, cv2.COLOR_RGB2BGR)
    )
    x.numpy().astype(np.float32).tofile(onnx_path.parent / "sample_input.bin")

    with torch.no_grad():
        ref = net(x)
    ref_np = ref.numpy().astype(np.float32)
    ref_np.tofile(onnx_path.parent / "python_output.bin")
    print(f"[export] PyTorch depth range {float(ref.min()):.3f} .. {float(ref.max()):.3f} m")

    dynamic_axes = None
    export_kwargs: dict[str, Any] = dict(
        input_names=["input"],
        output_names=["depth"],
        opset_version=opset,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    # torch >= 2.6 defaults to the dynamo exporter, which needs the extra
    # `onnxscript` package. This repository declares only `onnx`/`onnxruntime`,
    # so ask for the TorchScript exporter explicitly and fall back if the
    # installed torch predates the `dynamo` flag.
    try:
        torch.onnx.export(net, (x,), str(onnx_path), dynamo=False, **export_kwargs)
    except TypeError:  # pragma: no cover - older torch
        torch.onnx.export(net, (x,), str(onnx_path), **export_kwargs)
    print(f"[export] wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.2f} MB, opset {opset})")

    import onnx

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    print(f"[export] onnx.checker OK, {len(model.graph.node)} nodes")

    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = int(cfg_get(cfg, "num_threads", 4))
    sess = ort.InferenceSession(str(onnx_path), sess_options, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input": x.numpy().astype(np.float32)})[0]
    diff = np.abs(ort_out - ref_np)
    print(f"[export] ONNX vs PyTorch: max abs diff {diff.max():.3e}, mean {diff.mean():.3e}")
    if float(diff.max()) > 1e-3:
        print("[export] ERROR: ONNX Runtime and PyTorch disagree beyond 1e-3")
        return 1

    # --- Python timing (>= 50 runs, no_grad, pinned threads) ---------------
    threads = cfg_get(cfg, "num_threads", 4)
    for _ in range(warmup):
        with torch.no_grad():
            net(x)
    times: list[float] = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            net(x)
            times.append((time.perf_counter() - t0) * 1000.0)
    torch_timing = percentiles(times)
    # `mean_ms`/`p50_ms`/`p95_ms` mirror the key names the C++ program writes, so
    # the benchmark script can consume both files the same way.
    torch_timing.update(
        {
            "mean_ms": float(torch_timing["mean"]),
            "p50_ms": float(torch_timing["p50"]),
            "p95_ms": float(torch_timing["p95"]),
            "implementation": "python-pytorch",
            "runs": int(runs),
            "warmup": int(warmup),
            "threads": int(threads) if threads is not None else "default",
            "input_shape": [1, 3, height, width],
            "device": "cpu",
            "method": "torch.no_grad(), one frame per timed call, wall clock around the forward pass",
        }
    )

    ort_times: list[float] = []
    for _ in range(warmup):
        sess.run(None, {"input": x.numpy().astype(np.float32)})
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {"input": x.numpy().astype(np.float32)})
        ort_times.append((time.perf_counter() - t0) * 1000.0)
    ort_timing = percentiles(ort_times)
    ort_timing.update(
        {
            "mean_ms": float(ort_timing["mean"]),
            "p50_ms": float(ort_timing["p50"]),
            "p95_ms": float(ort_timing["p95"]),
            "implementation": "python-onnxruntime",
            "runs": int(runs),
            "warmup": int(warmup),
            "threads": int(sess_options.intra_op_num_threads),
            "input_shape": [1, 3, height, width],
            "device": "cpu",
            "method": "onnxruntime CPUExecutionProvider, one frame per timed call",
        }
    )

    payload = {
        "step": "export",
        "onnx_path": str(onnx_path),
        "artifacts": {
            "sample_input_bin": str(onnx_path.parent / "sample_input.bin"),
            "sample_frame_png": str(onnx_path.parent / "sample_frame.png"),
            "python_output_bin": str(onnx_path.parent / "python_output.bin"),
        },
        "opset": opset,
        "output": "depth in metres, shape (1, 1, H, W)",
        "parity_onnx_vs_pytorch": {
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
        },
        "frame_index": int(held[0] + frame_index),
        "sequence": full.name,
        "seed": seed,
        **torch_timing,
        "onnxruntime": ort_timing,
    }
    out = save_json(Path("results") / "python_timing.json", payload)
    print(
        f"[export] python pytorch  mean {torch_timing['mean']:.3f} ms  p50 {torch_timing['p50']:.3f}  "
        f"p95 {torch_timing['p95']:.3f}  ({runs} runs)"
    )
    print(
        f"[export] python onnxrt   mean {ort_timing['mean']:.3f} ms  p50 {ort_timing['p50']:.3f}  "
        f"p95 {ort_timing['p95']:.3f}"
    )
    print(f"[export] wrote {out}")
    return 0
