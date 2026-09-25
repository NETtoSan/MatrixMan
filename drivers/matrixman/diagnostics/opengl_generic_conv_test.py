"""CPU-reference and optional live checks for generic dense OpenGL Conv2D."""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from drivers.matrixman.backends.opengl import convolution
from drivers.matrixman.backends.opengl.storage import packed_atlas_size


def _shader_params(shape, weight, bias, stride, padding):
    _, cin, height, width = shape
    cout, _weight_cin, kh, kw = weight.shape
    hout = (height + 2 * padding[0] - kh) // stride[0] + 1
    wout = (width + 2 * padding[1] - kw) // stride[1] + 1
    in_tw, in_th = packed_atlas_size(cin * height * width)
    weight_tw, weight_th = packed_atlas_size(weight.numel())
    bias_tw, _ = packed_atlas_size(cout)
    out_tw, _ = packed_atlas_size(cout * hout * wout)
    return (
        cin, height, width, cout, hout, wout, kh, kw,
        stride[0], stride[1], padding[0], padding[1], bias is not None, 1, 0,
        in_tw, in_th, weight_tw, weight_th, bias_tw, out_tw, False,
    )


def _check_cpu_case(shape, cout, kernel, stride, padding, *, with_bias):
    torch.manual_seed(sum(shape) + cout + kernel[0] * 11 + kernel[1] * 17)
    x = torch.randn(shape, dtype=torch.float32)
    weight = torch.randn((cout, shape[1], kernel[0], kernel[1]), dtype=torch.float32)
    bias = torch.randn((cout,), dtype=torch.float32) if with_bias else None
    expected = F.conv2d(x, weight, bias, stride=stride, padding=padding)
    if kernel == (7, 7) and shape == (1, 3, 224, 224):
        assert tuple(expected.shape) == (1, 64, 112, 112)
    descriptor = convolution.classify_convolution(
        in_channels=shape[1], out_channels=cout,
        weight_in_channels=shape[1], groups=1, kernel=kernel,
        stride=stride, padding=padding, dilation=(1, 1),
    )
    assert descriptor.path == "dense"
    params = _shader_params(shape, weight, bias, stride, padding)
    source = convolution._conv_shader_source(params).decode("ascii")
    assert f"for (int ky = 0; ky < {kernel[0]}; ++ky)" in source
    assert f"for (int kx = 0; kx < {kernel[1]}; ++kx)" in source
    assert f"* {kernel[0]} + ky" in source
    tiled_source = convolution._conv_tile_shader_source(
        params, 8, 16, descriptor,
    ).decode("ascii")
    assert "tex_y + 16" in tiled_source and "tex_x + 8" in tiled_source
    return x, weight, bias, expected


def _errors(actual, expected):
    delta = (actual - expected).abs()
    return float(delta.max()), float((delta / expected.abs().clamp_min(1e-12)).max())


def _run_live():
    from drivers import matrixman

    matrixman.config.backend = "opengl"
    matrixman.config.prepared_execution = False
    matrixman.init()
    try:
        cases = [
            ((1, 3, 224, 224), 64, (7, 7), (2, 2), (3, 3), False),
            ((1, 3, 16, 16), 4, (5, 5), (1, 1), (2, 2), True),
            ((1, 3, 16, 18), 4, (3, 5), (1, 1), (1, 2), True),
            ((1, 2, 13, 15), 3, (1, 5), (2, 2), (0, 2), True),
            ((1, 2, 13, 15), 3, (5, 1), (1, 1), (2, 0), False),
        ]
        for shape, cout, kernel, stride, padding, with_bias in cases:
            x, weight, bias, expected = _check_cpu_case(
                shape, cout, kernel, stride, padding, with_bias=with_bias
            )
            actual = F.conv2d(
                matrixman.to_device(x), weight, bias,
                stride=stride, padding=padding,
            ).cpu()
            max_abs, max_rel = _errors(actual, expected)
            print(
                f"kernel={kernel} stride={stride} padding={padding} "
                f"max_abs_error={max_abs:.6g} max_rel_error={max_rel:.6g}"
            )
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
    finally:
        matrixman.shutdown()


def main() -> int:
    _check_cpu_case((1, 3, 224, 224), 64, (7, 7), (2, 2), (3, 3), with_bias=False)
    _check_cpu_case((1, 3, 16, 16), 4, (5, 5), (1, 1), (2, 2), with_bias=True)
    _check_cpu_case((1, 3, 16, 18), 4, (3, 5), (1, 1), (1, 2), with_bias=True)
    _check_cpu_case((1, 2, 13, 15), 3, (1, 5), (2, 2), (0, 2), with_bias=True)
    _check_cpu_case((1, 2, 13, 15), 3, (5, 1), (1, 1), (2, 0), with_bias=False)
    if "--gpu" in sys.argv:
        _run_live()
    print("OpenGL generic Conv2D tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
