"""Static and optional live checks for MatrixMan's packed 2D transpose."""

from __future__ import annotations

import sys

import torch

from drivers.matrixman.backends.opengl.ops import broadcast


def _static_checks() -> None:
    source = broadcast._transpose_shader_source((2, 3, 4, 3, 1, 4, 4, 2)).decode("ascii")
    assert "transpose_at(base)" in source
    assert "source_index = 4" in source
    assert tuple(torch.tensor(2.0).t().shape) == ()
    assert tuple(torch.arange(3.0).t().shape) == (3,)
    matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert torch.equal(matrix.t(), torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]))
    assert torch.equal(matrix.t().t(), matrix)


def _live_checks() -> None:
    from drivers import matrixman

    matrixman.config.backend = "opengl"
    matrixman.init()
    try:
        cpu = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        actual = matrixman.to_device(cpu).t()
        torch.testing.assert_close(actual.cpu(), cpu.t(), rtol=0, atol=0)
        restored = actual.t()
        torch.testing.assert_close(restored.cpu(), cpu, rtol=0, atol=0)
        print("OpenGL transpose live checks: PASS")
    finally:
        matrixman.shutdown()


def main() -> int:
    _static_checks()
    print("OpenGL transpose static checks: PASS")
    if "--gpu" in sys.argv:
        _live_checks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
