"""Tests for the ONNX export helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn

from monogeo.export import lower_clip_to_relu
from monogeo.models import disp_to_depth


class _ClampHead(nn.Module):
    """1x1 conv into the same clamp + depth mapping the real depth head uses."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor([[[[3.0]], [[-2.0]], [[1.5]]]]))
            self.conv.bias.fill_(-0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return disp_to_depth(self.conv(x), 0.1, 100.0)


class _OneSidedClamp(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=0.25) * 2.0


def _export(net: nn.Module, x: torch.Tensor, path: Path) -> onnx.ModelProto:
    kwargs = dict(input_names=["input"], output_names=["out"], opset_version=17)
    try:
        torch.onnx.export(net, (x,), str(path), dynamo=False, **kwargs)
    except TypeError:  # pragma: no cover - older torch
        torch.onnx.export(net, (x,), str(path), **kwargs)
    return onnx.load(str(path))


def _run(model: onnx.ModelProto, x: torch.Tensor) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(None, {"input": x.numpy()})[0]


@pytest.mark.parametrize("net", [_ClampHead(), _OneSidedClamp()], ids=["two-sided", "min-only"])
def test_lower_clip_to_relu_matches_pytorch(net: nn.Module, tmp_path: Path) -> None:
    """The Relu lowering removes every Clip and computes the same values, saturated ones included."""
    net.eval()
    x = torch.rand(1, 3, 16, 24, generator=torch.Generator().manual_seed(0)) * 2.0 - 0.5
    with torch.no_grad():
        ref = net(x).numpy()

    model = _export(net, x, tmp_path / "m.onnx")
    assert any(n.op_type == "Clip" for n in model.graph.node)
    clip_out = _run(model, x)

    assert lower_clip_to_relu(model) >= 1
    onnx.checker.check_model(model)
    assert not any(n.op_type == "Clip" for n in model.graph.node)

    if isinstance(net, _ClampHead):
        # Both clamp bounds must actually be exercised for this test to mean anything.
        with torch.no_grad():
            pre = net.conv(x)
        assert bool((pre < 1e-6).any()) and bool((pre > 1.0 - 1e-6).any())

    lowered_out = _run(model, x)
    # Same graph semantics as the original Clip, to float rounding ...
    np.testing.assert_allclose(lowered_out, clip_out, rtol=1e-6, atol=1e-6)
    # ... and therefore the same answer as PyTorch (looser: the conv itself rounds
    # differently, and depth is steep near the far clamp bound).
    np.testing.assert_allclose(lowered_out, ref, rtol=1e-4, atol=1e-4)
