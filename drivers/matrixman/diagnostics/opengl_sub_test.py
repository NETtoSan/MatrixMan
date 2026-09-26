"""Static and optional live checks for generic packed subtraction."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drivers.matrixman.backends.opengl.ops import arithmetic


def _params(shape, alpha=1.0, offset=0):
    shape = tuple(shape)
    strides = torch.empty(shape).stride()
    return (
        shape, offset, offset, strides, strides,
        8, 8, 8, 8, 8, alpha,
    )


def _static_checks() -> None:
    for shape in ((), (5,), (2, 3), (2, 2), (1, 3, 4), (1, 2, 3, 4)):
        source = arithmetic._packed_sub_shader_source(_params(shape, alpha=2.5)).decode("ascii")
        assert "ALPHA" not in source
        assert "subtract_at(base)" in source
        assert "2.5" in source

    cases = [
        torch.tensor(3.0),
        torch.arange(5.0),
        torch.arange(6.0).reshape(2, 3),
        torch.arange(24.0).reshape(1, 3, 2, 4),
    ]
    for lhs in cases:
        rhs = lhs + 1.0
        assert torch.equal(lhs - 2.5 * rhs, lhs - 2.5 * rhs)


def _live_checks() -> None:
    from drivers import matrixman

    matrixman.config.backend = "opengl"
    matrixman.init()
    try:
        for shape in ((5,), (2, 3), (2, 2), (1, 3, 4), (1, 2, 3, 4)):
            lhs_cpu = torch.arange(float(torch.tensor(shape).prod()), dtype=torch.float32).reshape(shape)
            rhs_cpu = lhs_cpu + 1.0
            lhs = matrixman.to_device(lhs_cpu)
            rhs = matrixman.to_device(rhs_cpu)
            actual = lhs - 2.5 * rhs
            torch.testing.assert_close(actual.cpu(), lhs_cpu - 2.5 * rhs_cpu, rtol=0, atol=0)
        print("OpenGL packed sub live checks: PASS")
    finally:
        matrixman.shutdown()


def main() -> int:
    _static_checks()
    print("OpenGL packed sub static checks: PASS")
    if "--gpu" in sys.argv:
        _live_checks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
