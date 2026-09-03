"""Focused CUDA diagnostic for ``Tensor.new_full``."""

import torch

from drivers import matrixman
from ...tensor import PRIVATEUSE_DEVICE


def run_diagnostic() -> int:
    matrixman.prefer("cuda")
    source = matrixman.to_device(torch.tensor([2.0], dtype=torch.float32))
    result = source.new_full((5,), 0.5)
    values = result.cpu()
    expected = torch.full((5,), 0.5, dtype=torch.float32)
    print("device:", PRIVATEUSE_DEVICE)
    print("dtype:", result.dtype)
    print("shape:", list(result.shape))
    print("readback:", values.tolist())
    print("allclose:", bool(torch.allclose(values, expected)))
    if not torch.allclose(values, expected):
        return 1

    empty = source.new_full((0,), 1.0)
    print("zero-size shape:", list(empty.shape))
    print("zero-size numel:", empty.numel())
    assert empty.numel() == 0
    return 0


if __name__ == "__main__":
    raise SystemExit(run_diagnostic())
