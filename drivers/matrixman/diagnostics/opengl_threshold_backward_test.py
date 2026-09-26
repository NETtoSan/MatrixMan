"""Static and optional live checks for packed threshold backward."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drivers.matrixman.backends.opengl.ops import activation


def _params(shape, threshold=0.0, offset=0):
    shape = tuple(shape)
    strides = tuple(torch.empty(shape).stride())
    return (shape, offset, strides, 8, 8, offset, strides, 8, 8, 8, threshold)


def _static_checks():
    for shape in ((), (5,), (2, 3), (2, 2), (1, 3, 4), (1, 2, 3, 4)):
        source = activation._threshold_backward_shader_source(_params(shape, 0.25)).decode("ascii")
        assert "__GRAD_INDEX__" not in source
        assert "__SELF_INDEX__" not in source
        assert "x > 0.25" in source
        assert "? g : 0.0" in source


def _live_checks():
    from drivers import matrixman

    matrixman.config.backend = "opengl"
    matrixman.init()
    try:
        for shape, threshold in (((), 0.0), ((5,), 0.0), ((2, 2), 0.25),
                                 ((1, 3, 4), -0.5), ((1, 2, 3, 4), 1.0)):
            count = math.prod(shape) if shape else 1
            self_cpu = torch.linspace(-1.0, 1.0, count, dtype=torch.float32).reshape(shape)
            grad_cpu = torch.arange(1.0, count + 1.0, dtype=torch.float32).reshape(shape)
            actual = torch.ops.aten.threshold_backward(
                matrixman.to_device(grad_cpu), matrixman.to_device(self_cpu), threshold
            )
            expected = torch.where(self_cpu > threshold, grad_cpu, torch.zeros_like(grad_cpu))
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        print("OpenGL threshold backward live checks: PASS")
    finally:
        matrixman.shutdown()


def main():
    _static_checks()
    print("OpenGL threshold backward static checks: PASS")
    if "--gpu" in sys.argv:
        _live_checks()


if __name__ == "__main__":
    main()
