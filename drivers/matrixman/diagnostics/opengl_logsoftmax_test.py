"""Static and optional live checks for generic packed log-softmax."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drivers.matrixman.backends.opengl.ops import softmax


def _params(shape, dim):
    shape = tuple(shape)
    return (shape, dim, 0, tuple(torch.empty(shape).stride()), 8, 8, 8)


def _static_checks():
    for shape, dim in (((3, 4), 1), ((1, 4), 1), ((2, 5, 3), -1), ((1, 2, 3, 4), 1)):
        normalized = dim if dim >= 0 else len(shape) + dim
        source = softmax._logsoftmax_shader_source(_params(shape, normalized)).decode("ascii")
        assert "__REDUCTION_INDEX__" not in source
        assert "__CURRENT_INDEX__" not in source
        assert "maximum = max" in source
        assert "exp(source_for_reduction" in source
        assert "- maximum - log(total)" in source
    backward = softmax._logsoftmax_backward_shader_source(
        ((3, 4), 1, 0, (4, 1), 8, 8, 0, (4, 1), 8, 8, 8)
    ).decode("ascii")
    assert "summed_grad" in backward
    assert "grad - exp(logp) * summed_grad" in backward
    print("OpenGL log-softmax static checks: PASS")


def _live_checks():
    from drivers import matrixman

    matrixman.config.backend = "opengl"
    matrixman.init()
    try:
        cases = [
            (torch.tensor([[1.25, -0.5, 0.25, 2.0], [-1.0, 0.75, 1.5, -0.25], [0.1, 0.2, 0.3, 0.4]], dtype=torch.float32), 1),
            (torch.tensor([[1000.0, 999.0, -1000.0, 998.0]], dtype=torch.float32), -1),
        ]
        for cpu, dim in cases:
            actual = torch.log_softmax(matrixman.to_device(cpu), dim=dim)
            expected = torch.log_softmax(cpu, dim=dim)
            torch.testing.assert_close(actual.cpu(), expected, rtol=2e-4, atol=2e-4)
        base_cpu = torch.arange(20.0, dtype=torch.float32).reshape(5, 4)
        view = matrixman.to_device(base_cpu)[1:]
        expected = torch.log_softmax(base_cpu[1:], dim=1)
        torch.testing.assert_close(torch.log_softmax(view, dim=1).cpu(), expected, rtol=2e-4, atol=2e-4)
        print("OpenGL log-softmax live checks: PASS")
    finally:
        matrixman.shutdown()


def main():
    _static_checks()
    if "--gpu" in sys.argv:
        _live_checks()


if __name__ == "__main__":
    main()
