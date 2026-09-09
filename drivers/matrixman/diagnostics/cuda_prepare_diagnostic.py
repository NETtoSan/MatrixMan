"""Focused checks for the opt-in CUDA BatchNorm preparation experiment."""

from __future__ import annotations

import torch
from torch import nn

from drivers import matrixman
from drivers.matrixman.backends.cuda import profiling


class Conv(nn.Module):
    """Minimal Ultralytics-style direct conv -> bn -> activation block."""

    def __init__(self, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1, bias=bias)
        self.bn = nn.BatchNorm2d(4)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def main():
    torch.manual_seed(7)
    for has_bias in (False, True):
        original = Conv(has_bias).eval()
        prepared = matrixman.prepare(original, backend="cuda")
        x = torch.randn(2, 3, 12, 12)
        with torch.no_grad():
            expected = original(x)
            actual = prepared(x)
        error = (expected - actual).abs()
        assert isinstance(original.bn, nn.BatchNorm2d)
        assert isinstance(prepared.bn, nn.Identity)
        assert torch.allclose(expected, actual, atol=1e-5, rtol=1e-5)
        print({"bias": has_bias, "max_abs_error": float(error.max()), "allclose": True})

    training = Conv().train()
    try:
        matrixman.prepare(training, backend="cuda")
    except ValueError as exc:
        print({"training_rejected": True, "reason": str(exc)})
    else:
        raise AssertionError("training model was prepared")

    original = Conv().eval()
    prepared = matrixman.prepare(original, backend="cuda")
    assert isinstance(original.bn, nn.BatchNorm2d)
    assert prepared._matrixman_cuda_preparation_report.folded == 1

    matrixman.prefer("cuda")
    matrixman.init()
    try:
        gpu_input = matrixman.to_device(torch.randn(1, 3, 16, 16))
        matrixman.profile_reset()
        with torch.no_grad():
            original_output = original(gpu_input).cpu()
        before = {name: int(record["calls"]) for name, record in profiling.records.items()}
        matrixman.profile_reset()
        with torch.no_grad():
            prepared_output = prepared(gpu_input).cpu()
        after = {name: int(record["calls"]) for name, record in profiling.records.items()}
        error = (original_output - prepared_output).abs()
        print({
            "cuda_max_abs_error": float(error.max()),
            "cuda_mean_abs_error": float(error.mean()),
            "cuda_allclose": bool(torch.allclose(original_output, prepared_output, atol=1e-5, rtol=1e-5)),
            "before_callbacks": {name: before.get(name, 0) for name in ("Conv2D", "BatchNorm", "SiLU")},
            "after_callbacks": {name: after.get(name, 0) for name in ("Conv2D", "BatchNorm", "SiLU")},
        })
    finally:
        matrixman.shutdown()


if __name__ == "__main__":
    main()
