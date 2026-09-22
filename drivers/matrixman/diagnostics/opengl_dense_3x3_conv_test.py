"""Audit, correctness checks, and optional live benchmark for dense 3x3 Conv2D."""

from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as F

from drivers.matrixman.backends.opengl import convolution
from drivers.matrixman.backends.opengl.storage import packed_atlas_size


def _params(x, weight, bias, stride, padding, storage_offset=0):
    _, cin, height, width = x.shape
    cout = weight.shape[0]
    out_height = (height + 2 * padding - 3) // stride + 1
    out_width = (width + 2 * padding - 3) // stride + 1
    in_tw, in_th = packed_atlas_size(x.numel())
    weight_tw, weight_th = packed_atlas_size(weight.numel())
    bias_tw, _ = packed_atlas_size(cout)
    out_tw, _ = packed_atlas_size(cout * out_height * out_width)
    return (
        cin, height, width, cout, out_height, out_width, 3, 3,
        stride, stride, padding, padding, bias is not None, 1, storage_offset,
        in_tw, in_th, weight_tw, weight_th, bias_tw, out_tw, False,
    )


def _check_case(shape, cout, stride, padding, *, with_bias):
    torch.manual_seed(sum(shape) + cout + stride + padding + int(with_bias))
    x = torch.randn(shape, dtype=torch.float32)
    weight = torch.randn((cout, shape[1], 3, 3), dtype=torch.float32)
    bias = torch.randn((cout,), dtype=torch.float32) if with_bias else None
    expected = F.conv2d(x, weight, bias, stride=stride, padding=padding)
    descriptor = convolution.classify_convolution(
        in_channels=shape[1], out_channels=cout,
        weight_in_channels=shape[1], groups=1, kernel=(3, 3),
        stride=(stride, stride), padding=(padding, padding), dilation=(1, 1),
    )
    params = _params(x, weight, bias, stride, padding)
    generic = convolution._conv_shader_source(params).decode("ascii")
    specialized = convolution._dense_3x3_shader_source(params).decode("ascii")
    assert descriptor.path == "dense"
    assert "for (int ky" in generic and "for (int kx" in generic
    assert "for (int ky" not in specialized
    assert "for (int kx" not in specialized
    assert "for (int ic" in specialized
    assert "int iy0 = oy *" in specialized
    assert "read_weight(oc, ic, 2, 2)" in specialized
    torch.testing.assert_close(expected, F.conv2d(x, weight, bias, stride=stride, padding=padding))
    return params


def _errors(value, expected):
    delta = (value - expected).abs()
    denominator = expected.abs().clamp_min(torch.finfo(expected.dtype).tiny)
    return float(delta.max()), float((delta / denominator).max())


def _run_live_comparison(iterations=10):
    from drivers import matrixman
    from drivers.matrixman.backend import get_backend

    matrixman.config.backend = "opengl"
    matrixman.config.preparedExecution = False
    matrixman.init()
    try:
        cases = [
            ((1, 16, 160, 160), 16, 1, 1),
            ((1, 12, 80, 80), 12, 2, 1),
            ((1, 24, 40, 40), 24, 1, 0),
            ((1, 64, 40, 40), 64, 1, 1),
        ]
        for shape, cout, stride, padding in cases:
            torch.manual_seed(sum(shape) + cout + stride + padding)
            x_cpu = torch.randn(shape, dtype=torch.float32)
            weight = torch.randn((cout, shape[1], 3, 3), dtype=torch.float32)
            bias = torch.randn((cout,), dtype=torch.float32)
            expected = F.conv2d(x_cpu, weight, bias, stride=stride, padding=padding)
            x = matrixman.to_device(x_cpu)
            args = (
                x, weight, bias, (stride, stride), (padding, padding), (1, 1),
                False, (0, 0), 1,
            )
            specialized = convolution._execute_diagnostic_dense_3x3(args, specialized=True)
            generic = convolution._execute_diagnostic_dense_3x3(args, specialized=False)
            spec_cpu = specialized.cpu()
            generic_cpu = generic.cpu()
            max_abs, max_rel = _errors(spec_cpu, expected)
            generic_abs, generic_rel = _errors(generic_cpu, expected)
            versus_generic_abs, versus_generic_rel = _errors(spec_cpu, generic_cpu)
            print(
                f"shape={list(shape)}\n"
                f"specialized_max_abs_error={max_abs:.6g}\n"
                f"specialized_max_rel_error={max_rel:.6g}\n"
                f"generic_max_abs_error={generic_abs:.6g}\n"
                f"generic_max_rel_error={generic_rel:.6g}\n"
                f"specialized_vs_generic_max_abs_error={versus_generic_abs:.6g}\n"
                f"specialized_vs_generic_max_rel_error={versus_generic_rel:.6g}"
            )
            torch.testing.assert_close(spec_cpu, generic_cpu, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(spec_cpu, expected, rtol=2e-2, atol=2e-3)
            torch.testing.assert_close(generic_cpu, expected, rtol=2e-2, atol=2e-3)

            for _ in range(2):
                convolution._execute_diagnostic_dense_3x3(args, specialized=True)
                convolution._execute_diagnostic_dense_3x3(args, specialized=False)
            get_backend().synchronize()
            timings = {}
            for label, specialized_flag in (("generic", False), ("specialized", True)):
                get_backend().synchronize()
                started = time.perf_counter()
                for _ in range(iterations):
                    convolution._execute_diagnostic_dense_3x3(
                        args, specialized=specialized_flag
                    )
                get_backend().synchronize()
                timings[label] = (time.perf_counter() - started) * 1000.0 / iterations
            speedup = timings["generic"] / timings["specialized"] if timings["specialized"] else 0.0
            atlas_elements = int(expected.numel())
            atlas_width, atlas_height = packed_atlas_size(atlas_elements)
            dispatch = "tiled" if max(atlas_width, atlas_height) > int(matrixman.config.tileLimit if matrixman.config.tileLimit != "auto" else matrixman.config.resolvedTileLimit) else "direct"
            print(
                f"shape={list(shape)} generic_ms={timings['generic']:.3f} "
                f"specialized_ms={timings['specialized']:.3f} speedup={speedup:.3f} "
                f"max_abs_error={max_abs:.6g} max_rel_error={max_rel:.6g} "
                f"generic_max_abs_error={generic_abs:.6g} "
                f"generic_max_rel_error={generic_rel:.6g} "
                f"direct_or_tiled={dispatch}"
            )
    finally:
        matrixman.shutdown()


def main() -> int:
    _check_case((1, 16, 160, 160), 16, 1, 1, with_bias=True)
    _check_case((1, 12, 80, 80), 12, 2, 1, with_bias=False)
    _check_case((1, 24, 40, 40), 24, 1, 0, with_bias=True)
    _check_case((1, 64, 40, 40), 64, 1, 1, with_bias=True)
    _check_case((1, 5, 7, 9), 3, 1, 1, with_bias=True)
    _check_case((1, 5, 7, 9), 3, 2, 0, with_bias=False)
    offset_params = _params(
        torch.zeros((1, 5, 7, 9)), torch.zeros((3, 5, 3, 3)),
        torch.zeros((3,)), 1, 1, storage_offset=7,
    )
    assert "int linear_index = 7 +" in convolution._dense_3x3_shader_source(offset_params).decode("ascii")
    print("OpenGL dense 3x3 Conv tests: PASS")
    if "--gpu" in sys.argv:
        _run_live_comparison()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
