"""Static and optional live checks for packed GPU sum/mean reductions."""

from __future__ import annotations

import sys

import torch

from drivers.matrixman.backends.opengl.ops import reduction


def _params(shape, dims, keepdim, offset=0):
    shape = tuple(shape)
    out_shape = tuple(
        1 if keepdim and axis in dims else shape[axis]
        for axis in range(len(shape))
        if keepdim or axis not in dims
    )
    return (
        shape, tuple(dims), bool(keepdim), offset,
        tuple(torch.empty(shape).stride()),
        8, 8, 8 if out_shape else 1,
    )


def _static_checks():
    assert reduction.normalize_reduction_dims(-1, 3, "sum") == (2,)
    assert reduction.normalize_reduction_dims((0, -1), 3, "sum") == (0, 2)
    assert reduction.normalize_reduction_dims([1, 0], 3, "mean") == (0, 1)

    sum_source = reduction._reduction_shader_source(
        _params((2, 3, 4), (0, 2), False), "sum"
    ).decode("ascii")
    mean_source = reduction._reduction_shader_source(
        _params((2, 3, 4), (0, 2), False), "mean"
    ).decode("ascii")
    all_mean_source = reduction._reduction_shader_source(
        _params((2, 3, 4), (0, 1, 2), False), "mean"
    ).decode("ascii")
    assert "return total;" in sum_source
    assert "return total / float(8);" in mean_source
    assert "return total / float(24);" in all_mean_source
    assert "return total / float(8);" not in sum_source
    assert "8 +" in reduction._reduction_shader_source(
        _params((2, 3, 4), (0, 2), False, offset=8), "sum"
    ).decode("ascii")

    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    assert torch.equal(x.sum(), torch.tensor(276.0))
    assert torch.equal(x.sum(dim=1), torch.sum(x, dim=1))
    assert torch.equal(x.sum(dim=(-1, 0), keepdim=True), torch.sum(x, dim=(-1, 0), keepdim=True))
    assert torch.equal(x.mean(dim=1), torch.mean(x, dim=1))
    assert torch.equal(x.mean(), torch.tensor(11.5))
    assert torch.equal(torch.tensor(3.0).mean(), torch.tensor(3.0))


def _live_checks():
    from drivers import matrixman

    matrixman.config.backend = "opengl"
    matrixman.init()
    try:
        cpu = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        x = matrixman.to_device(cpu)
        checks = [
            (x.sum(), cpu.sum()),
            (x.sum(dim=1), cpu.sum(dim=1)),
            (x.sum(dim=(0, -1)), cpu.sum(dim=(0, -1))),
            (x.sum(dim=(0, -1), keepdim=True), cpu.sum(dim=(0, -1), keepdim=True)),
            (x.mean(dim=1), cpu.mean(dim=1)),
            (x.mean(), cpu.mean()),
        ]
        for actual, expected in checks:
            torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)
        base = matrixman.to_device(torch.arange(30, dtype=torch.float32).reshape(2, 3, 5))
        view = base[..., 1:]
        torch.testing.assert_close(view.sum().cpu(), torch.arange(30, dtype=torch.float32).reshape(2, 3, 5)[..., 1:].sum())
        torch.testing.assert_close(view.mean().cpu(), torch.arange(30, dtype=torch.float32).reshape(2, 3, 5)[..., 1:].mean())
        print("OpenGL reduction live checks: PASS")
    finally:
        matrixman.shutdown()


def main() -> int:
    _static_checks()
    print("OpenGL reduction static checks: PASS")
    if "--gpu" in sys.argv:
        _live_checks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
