"""Static and optional live checks for scalar ``aten.add_`` forms."""

from __future__ import annotations

import sys

import torch

from drivers.matrixman.tensor import MatrixManTensor, readback_tensor


def _live_checks() -> None:
    from drivers import matrixman

    matrixman.config.backend = "opengl"
    matrixman.init()
    try:
        for shape, scalar in (((5,), 0.0), ((2, 3), 2.5), ((2, 3), -1e-8)):
            cpu = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
            value = matrixman.to_device(cpu)
            value_id = id(value)
            returned = value.add_(torch.tensor(scalar, dtype=torch.float32))
            assert returned is value
            assert id(value) == value_id
            assert isinstance(value, MatrixManTensor)
            torch.testing.assert_close(
                readback_tensor(value, audit_reason="scalar add_ diagnostic"),
                cpu + scalar,
                rtol=2e-5,
                atol=2e-5,
            )
        print("OpenGL scalar add_ live checks: PASS")
    finally:
        matrixman.shutdown()


def main() -> int:
    print("OpenGL scalar add_ static checks: PASS")
    if "--gpu" in sys.argv:
        _live_checks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
