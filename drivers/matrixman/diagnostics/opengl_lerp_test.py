"""Static and optional live checks for packed ``aten.lerp_.Scalar``."""

from __future__ import annotations

import sys

import torch

from drivers.matrixman.backends.opengl.ops import arithmetic
from drivers.matrixman.tensor import MatrixManTensor, readback_tensor


def _static_checks() -> None:
    source = arithmetic._packed_lerp_shader_source(
        ((2, 2), 0, 0, (2, 1), (2, 1), 2, 2, 2, 2, 1, 0.1)
    ).decode("ascii")
    assert "self_value + 0.1" in source
    assert "end_value - self_value" in source
    assert arithmetic._render_packed_lerp_inplace.__doc__


def _live_checks() -> None:
    from drivers import matrixman

    matrixman.config.backend = "opengl"
    matrixman.init()
    try:
        cases = [
            torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float32),
            torch.arange(12, dtype=torch.float32).reshape(3, 4),
        ]
        for weight in (0.0, 1.0, 0.1, -0.25, 1.5):
            for self_cpu in cases:
                end_cpu = torch.flip(self_cpu, dims=[-1]).clone() + 0.5
                self_mm = matrixman.to_device(self_cpu)
                end_mm = matrixman.to_device(end_cpu)
                self_id = id(self_mm)
                end_before = readback_tensor(end_mm, audit_reason="lerp diagnostic end")
                returned = self_mm.lerp_(end_mm, weight)
                assert returned is self_mm
                assert id(self_mm) == self_id
                assert isinstance(self_mm, MatrixManTensor)
                torch.testing.assert_close(
                    readback_tensor(self_mm, audit_reason="lerp diagnostic result"),
                    torch.lerp(self_cpu, end_cpu, weight),
                    rtol=2e-5,
                    atol=2e-5,
                )
                torch.testing.assert_close(
                    readback_tensor(end_mm, audit_reason="lerp diagnostic end unchanged"),
                    end_before,
                    rtol=0,
                    atol=0,
                )
        print("OpenGL lerp_ live checks: PASS")
    finally:
        matrixman.shutdown()


def main() -> int:
    _static_checks()
    print("OpenGL lerp_ static checks: PASS")
    if "--gpu" in sys.argv:
        _live_checks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
