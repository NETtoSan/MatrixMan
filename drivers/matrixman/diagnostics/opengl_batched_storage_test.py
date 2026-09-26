"""Checks for generic batched packed storage and view metadata."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drivers.matrixman.backends.opengl import metadata


def _static_checks() -> None:
    for shape in ((1, 1, 28, 28), (2, 1, 28, 28), (4, 3, 5, 7)):
        metadata.validate_supported_shape(shape)
    assert metadata.contiguous_strides((2, 1, 28, 28)) == (784, 784, 28, 1)
    assert metadata.normalize_shape((2, 784), 2 * 784) == (2, 784)
    print("OpenGL batched storage static checks: PASS")


def _live_checks() -> None:
    from drivers import matrixman

    matrixman.config.backend = "opengl"
    matrixman.init()
    try:
        for shape in ((1, 1, 28, 28), (2, 1, 28, 28), (4, 3, 5, 7)):
            cpu = torch.arange(1, 1 + int(torch.tensor(shape).prod()), dtype=torch.float32).reshape(shape)
            tensor = matrixman.to_device(cpu)
            assert tensor._owner.layout.kind == "packed_rgba"
            torch.testing.assert_close(tensor.cpu(), cpu, rtol=0, atol=0)

        source_cpu = torch.arange(2 * 1 * 28 * 28, dtype=torch.float32).reshape(2, 1, 28, 28)
        source = matrixman.to_device(source_cpu)
        flat = source.flatten(1)
        view = source.view(2, 784)
        torch.testing.assert_close(flat.cpu(), source_cpu.flatten(1), rtol=0, atol=0)
        torch.testing.assert_close(view.cpu(), source_cpu.view(2, 784), rtol=0, atol=0)
        print("OpenGL batched storage live checks: PASS")
    finally:
        matrixman.shutdown()


def main() -> int:
    _static_checks()
    if "--gpu" in sys.argv:
        _live_checks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
