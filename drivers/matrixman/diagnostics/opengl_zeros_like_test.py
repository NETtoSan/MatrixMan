"""Static and optional live checks for the OpenGL zeros_like implementation."""

from __future__ import annotations

import sys

import torch

from drivers.matrixman.backends.opengl.ops import concat
from drivers.matrixman.tensor import MatrixManTensor, readback_tensor


def _static_checks() -> None:
    source = concat._fill_shader_source((17, 3, 0.0)).decode("ascii")
    assert "return 0.0;" in source
    assert "fill_at(base)" in source
    assert concat.render_zeros_like.__doc__


def _live_checks() -> None:
    from drivers import matrixman

    matrixman.config.backend = "opengl"
    matrixman.init()
    try:
        cpu_inputs = [
            torch.arange(7, dtype=torch.float32),
            torch.arange(12, dtype=torch.float32).reshape(3, 4),
            torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5),
            torch.tensor(3.0, dtype=torch.float32),
        ]
        for cpu in cpu_inputs:
            source = matrixman.to_device(cpu)
            source.requires_grad_(True)
            actual = torch.zeros_like(source)
            assert isinstance(actual, MatrixManTensor)
            assert actual.device == source.device
            assert actual.dtype == source.dtype
            assert tuple(actual.shape) == tuple(cpu.shape)
            assert not actual.requires_grad
            assert actual._owner.texture != source._owner.texture
            readback = readback_tensor(actual, audit_reason="zeros_like diagnostic")
            torch.testing.assert_close(readback, torch.zeros_like(cpu), rtol=0, atol=0)

        base_cpu = torch.arange(30, dtype=torch.float32).reshape(2, 3, 5)
        view = matrixman.to_device(base_cpu)[..., 1:]
        actual = torch.zeros_like(view)
        assert tuple(actual.shape) == (2, 3, 4)
        torch.testing.assert_close(
            readback_tensor(actual, audit_reason="zeros_like offset diagnostic"),
            torch.zeros_like(base_cpu[..., 1:]),
            rtol=0,
            atol=0,
        )
        print("OpenGL zeros_like live checks: PASS")
    finally:
        matrixman.shutdown()


def main() -> int:
    _static_checks()
    print("OpenGL zeros_like static checks: PASS")
    if "--gpu" in sys.argv:
        _live_checks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
