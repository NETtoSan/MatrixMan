"""Focused CUDA diagnostic for ``torch.arange(..., out=...)``."""

import torch

from drivers import matrixman


def run_diagnostic() -> int:
    matrixman.prefer("cuda")
    source = matrixman.to_device(torch.tensor([0.0], dtype=torch.float32))
    out = source.new_full((5,), 0.0)
    result = torch.arange(5, out=out)
    values = result.cpu()
    expected = torch.arange(5, dtype=torch.float32)
    print("result is out:", result is out)
    print("device:", result.device)
    print("dtype:", result.dtype)
    print("readback:", values.tolist())
    print("allclose:", bool(torch.allclose(values, expected)))
    if result is not out or not torch.allclose(values, expected):
        return 1

    empty = source.new_full((0,), 0.0)
    empty_result = torch.arange(0, out=empty)
    print("zero-size result is out:", empty_result is empty)
    print("zero-size numel:", empty_result.numel())
    assert empty_result is empty and empty_result.numel() == 0
    return 0


if __name__ == "__main__":
    raise SystemExit(run_diagnostic())
