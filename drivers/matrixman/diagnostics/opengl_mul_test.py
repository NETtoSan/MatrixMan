"""Static and optional live checks for packed elementwise multiplication."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drivers.matrixman.backends.opengl.ops import arithmetic
from drivers.matrixman.diagnostics.opengl_sub_test import _params


def _static_checks() -> None:
    for shape in ((), (5,), (2, 3), (2, 2), (1, 3, 4), (1, 2, 3, 4)):
        params = _params(shape)[:-1]
        source = arithmetic._packed_mul_shader_source(params).decode("ascii")
        assert "__LEFT_INDEX__" not in source
        assert "__RIGHT_INDEX__" not in source
        assert "multiply_at(base)" in source
        assert "read_packed(left_tex" in source
        assert "read_packed(right_tex" in source
        assert "* read_packed" in source

    legacy = arithmetic._packed_broadcast_mul_shader_source(
        ((1, 4, 7), 0, 0, (28, 7, 1), (7, 1), 8, 8, 8, 8, 8)
    ).decode("ascii")
    assert "CHANNELS" not in legacy
    assert "multiply_at(base)" in legacy


def _live_checks() -> None:
    from drivers import matrixman

    matrixman.config.backend = "opengl"
    matrixman.init()
    try:
        for shape in ((), (5,), (2, 3), (2, 2), (1, 3, 4), (1, 2, 3, 4)):
            count = math.prod(shape) if shape else 1
            lhs_cpu = torch.arange(float(count), dtype=torch.float32).reshape(shape)
            rhs_cpu = lhs_cpu + 1.25
            lhs = matrixman.to_device(lhs_cpu)
            rhs = matrixman.to_device(rhs_cpu)
            actual = lhs * rhs
            torch.testing.assert_close(actual.cpu(), lhs_cpu * rhs_cpu, rtol=0, atol=0)

        base = matrixman.to_device(torch.arange(12.0, dtype=torch.float32).reshape(3, 4))
        lhs = base[1:, 1:]
        rhs = matrixman.to_device(torch.full((2, 3), 2.0))
        torch.testing.assert_close(lhs.mul(rhs).cpu(), base.cpu()[1:, 1:] * 2.0, rtol=0, atol=0)
        print("OpenGL packed mul live checks: PASS")
    finally:
        matrixman.shutdown()


def main() -> int:
    _static_checks()
    print("OpenGL packed mul static checks: PASS")
    if "--gpu" in sys.argv:
        _live_checks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
