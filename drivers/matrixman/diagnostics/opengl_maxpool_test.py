"""CPU-reference and optional live checks for generalized OpenGL max-pool."""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from drivers.matrixman.backends.opengl.ops import pooling
from drivers.matrixman.backends.opengl.storage import packed_atlas_size


def _params(shape, kernel, stride, padding, offset=0):
    _, channels, in_h, in_w = shape
    out_h = (in_h + 2 * padding[0] - kernel[0]) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - kernel[1]) // stride[1] + 1
    input_tw, input_th = packed_atlas_size(channels * in_h * in_w)
    output_tw, _ = packed_atlas_size(channels * out_h * out_w)
    return (
        channels, in_h, in_w, out_h, out_w,
        kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1],
        offset, input_tw, input_th, output_tw,
    )


def _check_source(shape, kernel, stride, padding, offset=0):
    params = _params(shape, kernel, stride, padding, offset)
    source = pooling._maxpool_shader_source(params).decode("ascii")
    assert f"ky < {kernel[0]}" in source
    assert f"kx < {kernel[1]}" in source
    assert f"oy * {stride[0]} + ky - {padding[0]}" in source
    assert f"ox * {stride[1]} + kx - {padding[1]}" in source
    if offset:
        assert f"{offset} + ((c * IN_H)".replace("IN_H", str(shape[2])) in source


def _cases():
    torch.manual_seed(404)
    return [
        ((1, 3, 224, 224), (3, 3), (2, 2), (1, 1)),
        ((1, 8, 16, 16), (5, 5), (1, 1), (2, 2)),
        ((1, 4, 15, 17), (3, 5), (1, 1), (0, 0)),
        ((1, 4, 15, 17), (2, 3), (2, 2), (0, 1)),
    ]


def _run_live():
    from drivers import matrixman
    from drivers.matrixman.tensor import MatrixManTensor

    matrixman.config.backend = "opengl"
    matrixman.init()
    try:
        for shape, kernel, stride, padding in _cases():
            value = torch.randn(shape, dtype=torch.float32) - 2.0
            expected = F.max_pool2d(value, kernel, stride, padding, dilation=1, ceil_mode=False)
            actual = F.max_pool2d(
                matrixman.to_device(value), kernel, stride, padding,
                dilation=1, ceil_mode=False,
            ).cpu()
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
            print(
                f"kernel={kernel} stride={stride} padding={padding} "
                f"shape={list(actual.shape)} max_abs_error={(actual - expected).abs().max().item():.6g}"
            )

        base_cpu = torch.randn((1, 4, 8, 9), dtype=torch.float32) - 3.0
        base = matrixman.to_device(base_cpu)
        offset = 5
        view_shape = (1, 2, 4, 4)
        view = MatrixManTensor._from_owner(base._owner, view_shape, offset)
        source = base_cpu.reshape(-1)[offset:offset + 32].reshape(view_shape)
        expected = F.max_pool2d(source, 3, 2, 1)
        actual = F.max_pool2d(view, 3, 2, 1).cpu()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        print("nonzero storage offset: PASS")
    finally:
        matrixman.shutdown()


def main() -> int:
    for shape, kernel, stride, padding in _cases():
        _check_source(shape, kernel, stride, padding)
        value = torch.randn(shape, dtype=torch.float32)
        expected = F.max_pool2d(value, kernel, stride, padding)
        assert tuple(expected.shape)[-2:] == (
            (shape[-2] + 2 * padding[0] - kernel[0]) // stride[0] + 1,
            (shape[-1] + 2 * padding[1] - kernel[1]) // stride[1] + 1,
        )
    _check_source((1, 2, 8, 9), (3, 3), (2, 2), (1, 1), offset=5)
    if "--gpu" in sys.argv:
        _run_live()
    print("OpenGL generalized max_pool2d tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
