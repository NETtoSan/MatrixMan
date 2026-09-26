"""Static and optional live checks for packed ``aten.addcmul_.default``."""

from __future__ import annotations

import sys

import torch

from drivers.matrixman.backends.opengl.ops import arithmetic
from drivers.matrixman.tensor import MatrixManTensor, readback_tensor


def _static_checks() -> None:
    source = arithmetic._packed_addcmul_shader_source(
        ((2, 2), (0, 0, 0), ((2, 1), (2, 1), (2, 1)), ((2, 2), (2, 2), (2, 2)), 1, 0.25)
    ).decode("ascii")
    assert "self_value + 0.25" in source
    assert "tensor1_value * tensor2_value" in source


def _live_checks() -> None:
    from drivers import matrixman

    matrixman.config.backend = "opengl"
    matrixman.init()
    try:
        for shape, value in (((5,), 0.0), ((2, 3), 1.0), ((2, 3), 0.001)):
            self_cpu = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
            tensor1_cpu = torch.full(shape, 2.0, dtype=torch.float32)
            tensor2_cpu = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape) + 1.0
            self_mm = matrixman.to_device(self_cpu)
            tensor1_mm = matrixman.to_device(tensor1_cpu)
            tensor2_mm = matrixman.to_device(tensor2_cpu)
            self_id = id(self_mm)
            tensor1_before = readback_tensor(tensor1_mm, audit_reason="addcmul tensor1")
            tensor2_before = readback_tensor(tensor2_mm, audit_reason="addcmul tensor2")
            returned = self_mm.addcmul_(tensor1_mm, tensor2_mm, value=value)
            assert returned is self_mm
            assert id(self_mm) == self_id
            assert isinstance(self_mm, MatrixManTensor)
            torch.testing.assert_close(
                readback_tensor(self_mm, audit_reason="addcmul result"),
                self_cpu + value * tensor1_cpu * tensor2_cpu,
                rtol=2e-5,
                atol=2e-5,
            )
            torch.testing.assert_close(
                readback_tensor(tensor1_mm, audit_reason="addcmul tensor1 unchanged"),
                tensor1_before,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                readback_tensor(tensor2_mm, audit_reason="addcmul tensor2 unchanged"),
                tensor2_before,
                rtol=0,
                atol=0,
            )
        print("OpenGL addcmul_ live checks: PASS")
    finally:
        matrixman.shutdown()


def main() -> int:
    _static_checks()
    print("OpenGL addcmul_ static checks: PASS")
    if "--gpu" in sys.argv:
        _live_checks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
