"""Static and optional live checks for packed scalar division."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drivers.matrixman.backends.opengl.ops import arithmetic


def _params(shape, divisor=2.0, offset=0):
    shape = tuple(shape)
    return (shape, offset, tuple(torch.empty(shape).stride()), 8, 8, 8, divisor)


def _static_checks():
    for shape in ((), (5,), (2, 2), (2, 3), (1, 3, 4), (1, 2, 3, 4)):
        source = arithmetic._packed_scalar_div_shader_source(_params(shape, -2.5)).decode("ascii")
        assert "__INPUT_INDEX__" not in source
        assert "-2.5" in source
        assert "divide_at(base)" in source


def _live_checks():
    from drivers import matrixman

    matrixman.config.backend = "opengl"
    matrixman.init()
    try:
        for shape in ((), (5,), (2, 2), (2, 3), (1, 3, 4), (1, 2, 3, 4)):
            count = math.prod(shape) if shape else 1
            cpu = torch.arange(float(count), dtype=torch.float32).reshape(shape)
            actual = matrixman.to_device(cpu) / -2.5
            torch.testing.assert_close(actual.cpu(), cpu / -2.5, rtol=0, atol=0)
        base_cpu = torch.arange(30.0, dtype=torch.float32).reshape(2, 3, 5)
        base = matrixman.to_device(base_cpu)
        actual = base[..., 1:] / 4.0
        torch.testing.assert_close(actual.cpu(), base_cpu[..., 1:] / 4.0, rtol=0, atol=0)
        print("OpenGL packed scalar div live checks: PASS")
    finally:
        matrixman.shutdown()


def main():
    _static_checks()
    print("OpenGL packed scalar div static checks: PASS")
    if "--gpu" in sys.argv:
        _live_checks()


if __name__ == "__main__":
    main()
