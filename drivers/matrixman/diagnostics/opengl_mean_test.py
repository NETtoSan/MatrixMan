"""CPU-reference and optional live checks for generic OpenGL mean.dim."""

from __future__ import annotations

import sys

import torch

from drivers.matrixman.backends.opengl.ops import reduction


def _source_checks():
    params = ((1, 512, 7, 7), (2, 3), True, 5, (25088, 49, 7, 1), 79, 80, 12)
    source = reduction._mean_shader_source(params).decode("ascii")
    assert "reduction_index < 49" in source
    assert "if (output_index >= 512)" in source
    assert "total / float(49)" in source


def _dimension_checks():
    assert reduction.normalize_reduction_dims(-1, 4) == (3,)
    assert reduction.normalize_reduction_dims([3, -2], 4) == (2, 3)
    assert reduction.normalize_reduction_dims((0, 2), 4) == (0, 2)
    for value, message in (([1, 1], "duplicate"), ([4], "out of range"), ([], "at least one")):
        try:
            reduction.normalize_reduction_dims(value, 4)
        except RuntimeError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"expected mean.dim validation failure for {value!r}")


def _run_live():
    from drivers import matrixman
    from drivers.matrixman.tensor import MatrixManTensor

    matrixman.config.backend = "opengl"
    matrixman.init()
    try:
        cases = [
            (torch.randn((1, 4, 5, 7), generator=torch.Generator().manual_seed(41)), (-2, -1), True),
            (torch.randn((1, 3, 6, 8), generator=torch.Generator().manual_seed(42)), (1, 3), False),
            (torch.randn((1, 5, 9), generator=torch.Generator().manual_seed(43)), 2, False),
        ]
        for value, dims, keepdim in cases:
            actual = torch.ops.aten.mean.dim(matrixman.to_device(value), dims, keepdim)
            expected = torch.mean(value, dim=dims, keepdim=keepdim)
            assert tuple(actual.shape) == tuple(expected.shape)
            torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-5)

        base_cpu = torch.arange(40, dtype=torch.float32).reshape(1, 1, 4, 10)
        base = matrixman.to_device(base_cpu)
        view = MatrixManTensor._from_owner(base._owner, (1, 1, 2, 6), 3)
        actual = torch.ops.aten.mean.dim(view, (-2, -1), True)
        expected = base_cpu.reshape(-1)[3:15].reshape(1, 1, 2, 6).mean(dim=(-2, -1), keepdim=True)
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-5)
        print("OpenGL mean.dim live numerical tests: PASS")
    finally:
        matrixman.shutdown()


def main() -> int:
    _source_checks()
    _dimension_checks()
    value = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4)
    assert torch.equal(value.mean(dim=(-2, -1), keepdim=True), value.mean(dim=(2, 3), keepdim=True))
    if "--gpu" in sys.argv:
        _run_live()
    print("OpenGL mean.dim tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
