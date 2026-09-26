"""Static and optional live checks for packed ``aten.sqrt.default``."""

from __future__ import annotations

import math
import sys

import torch

from drivers.matrixman.backends.opengl.ops import arithmetic
from drivers.matrixman.tensor import MatrixManTensor, readback_tensor


def _static_checks() -> None:
    source = arithmetic._packed_sqrt_shader_source(
        ((2, 2), 0, (2, 1), 2, 2, 1)
    ).decode("ascii")
    assert "return sqrt(read_packed(input_index));" in source


def _live_checks() -> None:
    from drivers import matrixman

    matrixman.config.backend = "opengl"
    matrixman.init()
    try:
        values = torch.tensor([0.0, 1.0, 4.0, 2.0, 9.0, 0.25], dtype=torch.float32).reshape(2, 3)
        source = matrixman.to_device(values)
        result = torch.sqrt(source)
        assert isinstance(result, MatrixManTensor)
        assert result._owner.texture != source._owner.texture
        torch.testing.assert_close(
            readback_tensor(result, audit_reason="sqrt diagnostic result"),
            torch.sqrt(values),
            rtol=2e-5,
            atol=2e-5,
        )
        torch.testing.assert_close(
            readback_tensor(source, audit_reason="sqrt diagnostic input unchanged"),
            values,
            rtol=0,
            atol=0,
        )

        negative = matrixman.to_device(torch.tensor([-1.0], dtype=torch.float32))
        negative_result = readback_tensor(torch.sqrt(negative), audit_reason="sqrt negative diagnostic")
        assert math.isnan(float(negative_result[0]))
        print("OpenGL sqrt live checks: PASS")
    finally:
        matrixman.shutdown()


def main() -> int:
    _static_checks()
    print("OpenGL sqrt static checks: PASS")
    if "--gpu" in sys.argv:
        _live_checks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
