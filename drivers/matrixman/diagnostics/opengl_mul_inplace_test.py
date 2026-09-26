"""Static and optional live checks for packed ``aten.mul_.Tensor``."""

from __future__ import annotations

import sys

import torch

from drivers.matrixman.backends.opengl.ops import arithmetic
from drivers.matrixman.tensor import MatrixManTensor, readback_tensor


def _static_checks() -> None:
    source = arithmetic._packed_scalar_mul_shader_source(
        ((2, 2), 0, (2, 1), 2, 2, 1, 0.5)
    ).decode("ascii")
    assert "* 0.5" in source
    assert arithmetic._replace_inplace_tensor.__doc__ is None


def _live_checks() -> None:
    from drivers import matrixman

    matrixman.config.backend = "opengl"
    matrixman.init()
    try:
        for shape in ((5,), (2, 3)):
            for scalar in (0.0, 0.5, 2.0, -1.25):
                cpu = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
                self_mm = matrixman.to_device(cpu)
                other_cpu = torch.tensor(scalar, dtype=torch.float32)
                self_id = id(self_mm)
                returned = self_mm.mul_(other_cpu)
                assert returned is self_mm
                assert id(self_mm) == self_id
                assert isinstance(self_mm, MatrixManTensor)
                torch.testing.assert_close(
                    readback_tensor(self_mm, audit_reason="mul_ diagnostic result"),
                    cpu * scalar,
                    rtol=2e-5,
                    atol=2e-5,
                )

        lhs_cpu = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        rhs_cpu = torch.full_like(lhs_cpu, 2.0)
        lhs_mm = matrixman.to_device(lhs_cpu)
        rhs_mm = matrixman.to_device(rhs_cpu)
        rhs_before = readback_tensor(rhs_mm, audit_reason="mul_ rhs")
        lhs_mm.mul_(rhs_mm)
        torch.testing.assert_close(
            readback_tensor(lhs_mm, audit_reason="mul_ tensor result"),
            lhs_cpu * rhs_cpu,
            rtol=2e-5,
            atol=2e-5,
        )
        torch.testing.assert_close(
            readback_tensor(rhs_mm, audit_reason="mul_ rhs unchanged"),
            rhs_before,
            rtol=0,
            atol=0,
        )
        print("OpenGL mul_ live checks: PASS")
    finally:
        matrixman.shutdown()


def main() -> int:
    _static_checks()
    print("OpenGL mul_ static checks: PASS")
    if "--gpu" in sys.argv:
        _live_checks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
