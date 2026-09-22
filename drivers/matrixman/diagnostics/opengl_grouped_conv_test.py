"""Audit, correctness checks, and optional live benchmark for grouped Conv2D."""

from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as F

from drivers.matrixman.backends.opengl import convolution
from drivers.matrixman.backends.opengl.storage import packed_atlas_size


def _params(x, weight, bias, stride, padding, groups, storage_offset=0):
    _, cin, height, width = x.shape
    cout, in_per_group = weight.shape[:2]
    out_height = (height + 2 * padding - 3) // stride + 1
    out_width = (width + 2 * padding - 3) // stride + 1
    in_tw, in_th = packed_atlas_size(x.numel())
    weight_tw, weight_th = packed_atlas_size(weight.numel())
    bias_tw, _ = packed_atlas_size(cout)
    out_tw, _ = packed_atlas_size(cout * out_height * out_width)
    return (
        cin, height, width, cout, out_height, out_width, 3, 3,
        stride, stride, padding, padding, bias is not None, groups, storage_offset,
        in_tw, in_th, weight_tw, weight_th, bias_tw, out_tw, False,
    ), in_per_group


def _errors(value, expected):
    delta = (value - expected).abs()
    denominator = expected.abs().clamp_min(torch.finfo(expected.dtype).tiny)
    return float(delta.max()), float((delta / denominator).max())


def _check_case(shape, cout, groups, stride, padding, *, with_bias):
    cin = shape[1]
    assert cin % groups == 0
    assert cout % groups == 0
    torch.manual_seed(sum(shape) + cout + groups + stride + padding + int(with_bias))
    x = torch.randn(shape, dtype=torch.float32)
    in_per_group = cin // groups
    weight = torch.randn((cout, in_per_group, 3, 3), dtype=torch.float32)
    assert weight.shape[1] == cin // groups
    bias = torch.randn((cout,), dtype=torch.float32) if with_bias else None
    expected = F.conv2d(x, weight, bias, stride=stride, padding=padding, groups=groups)
    descriptor = convolution.classify_convolution(
        in_channels=shape[1], out_channels=cout,
        weight_in_channels=in_per_group, groups=groups, kernel=(3, 3),
        stride=(stride, stride), padding=(padding, padding), dilation=(1, 1),
    )
    params, actual_in_per_group = _params(x, weight, bias, stride, padding, groups)
    generic = convolution._conv_shader_source(params).decode("ascii")
    specialized = convolution._grouped_3x3_shader_source(params).decode("ascii")
    assert descriptor.path == "grouped"
    assert actual_in_per_group == in_per_group
    assert "int group = oc /" in generic
    assert "int input_channel_start = group *" in generic
    assert "for (int ky" in generic and "for (int kx" in generic
    assert "for (int ky" not in specialized
    assert "for (int kx" not in specialized
    assert "int group = oc /" in specialized
    if in_per_group in {2, 3}:
        assert "for (int ic" not in specialized
    torch.testing.assert_close(expected, F.conv2d(
        x, weight, bias, stride=stride, padding=padding, groups=groups
    ))
    return params


def _run_live_comparison(iterations=10):
    from drivers import matrixman
    from drivers.matrixman.backend import get_backend

    matrixman.config.backend = "opengl"
    matrixman.config.preparedExecution = False
    matrixman.init()
    try:
        cases = [
            ((1, 16, 160, 160), 24, 8, 2, 1),
            ((1, 24, 160, 160), 8, 8, 1, 1),
            ((1, 16, 80, 80), 16, 4, 1, 1),
            ((1, 24, 40, 40), 24, 6, 1, 1),
        ]
        for shape, cout, groups, stride, padding in cases:
            cin = shape[1]
            assert cin % groups == 0
            assert cout % groups == 0
            in_per_group = cin // groups
            torch.manual_seed(sum(shape) + cout + groups + stride + padding)
            x_cpu = torch.randn(shape, dtype=torch.float32)
            weight = torch.randn((cout, in_per_group, 3, 3), dtype=torch.float32)
            assert weight.shape[1] == cin // groups
            bias = torch.randn((cout,), dtype=torch.float32)
            expected = F.conv2d(x_cpu, weight, bias, stride=stride, padding=padding, groups=groups)
            x = matrixman.to_device(x_cpu)
            args = (
                x, weight, bias, (stride, stride), (padding, padding), (1, 1),
                False, (0, 0), groups,
            )
            specialized = convolution._execute_diagnostic_grouped(args, specialized=True)
            generic = convolution._execute_diagnostic_grouped(args, specialized=False)
            spec_cpu = specialized.cpu()
            generic_cpu = generic.cpu()
            max_abs, max_rel = _errors(spec_cpu, expected)
            generic_abs, generic_rel = _errors(generic_cpu, expected)
            versus_generic_abs, versus_generic_rel = _errors(spec_cpu, generic_cpu)
            print(
                f"shape={list(shape)} groups={groups} "
                f"in_channels_per_group={in_per_group} "
                f"out_channels_per_group={cout // groups}\n"
                f"specialized_vs_generic_max_abs_error={versus_generic_abs:.6g}\n"
                f"specialized_vs_generic_max_rel_error={versus_generic_rel:.6g}\n"
                f"max_abs_error={max_abs:.6g} max_rel_error={max_rel:.6g}\n"
                f"generic_max_abs_error={generic_abs:.6g} generic_max_rel_error={generic_rel:.6g}"
            )
            torch.testing.assert_close(spec_cpu, generic_cpu, rtol=1e-2, atol=3e-5)
            torch.testing.assert_close(spec_cpu, expected, rtol=2e-2, atol=2e-3)
            torch.testing.assert_close(generic_cpu, expected, rtol=2e-2, atol=2e-3)

            for _ in range(2):
                convolution._execute_diagnostic_grouped(args, specialized=True)
                convolution._execute_diagnostic_grouped(args, specialized=False)
            get_backend().synchronize()
            timings = {}
            for label, specialized_flag in (("generic", False), ("specialized", True)):
                get_backend().synchronize()
                started = time.perf_counter()
                for _ in range(iterations):
                    convolution._execute_diagnostic_grouped(
                        args, specialized=specialized_flag
                    )
                get_backend().synchronize()
                timings[label] = (time.perf_counter() - started) * 1000.0 / iterations
            speedup = timings["generic"] / timings["specialized"] if timings["specialized"] else 0.0
            atlas_width, atlas_height = packed_atlas_size(int(expected.numel()))
            configured_limit = int(
                matrixman.config.tileLimit
                if matrixman.config.tileLimit != "auto"
                else matrixman.config.resolvedTileLimit
            )
            dispatch = "tiled" if max(atlas_width, atlas_height) > configured_limit else "direct"
            print(
                f"shape={list(shape)} groups={groups} "
                f"in_channels_per_group={in_per_group} "
                f"out_channels_per_group={cout // groups} "
                f"generic_ms={timings['generic']:.3f} "
                f"specialized_ms={timings['specialized']:.3f} speedup={speedup:.3f} "
                f"specialized_vs_generic_max_abs_error={versus_generic_abs:.6g} "
                f"specialized_vs_generic_max_rel_error={versus_generic_rel:.6g} "
                f"max_abs_error={max_abs:.6g} max_rel_error={max_rel:.6g} "
                f"generic_max_abs_error={generic_abs:.6g} generic_max_rel_error={generic_rel:.6g} "
                f"direct_or_tiled={dispatch}"
            )
    finally:
        matrixman.shutdown()


def main() -> int:
    _check_case((1, 16, 160, 160), 24, 8, 2, 1, with_bias=True)
    _check_case((1, 24, 160, 160), 8, 8, 1, 1, with_bias=False)
    _check_case((1, 16, 80, 80), 16, 4, 1, 1, with_bias=True)
    _check_case((1, 24, 40, 40), 24, 6, 1, 1, with_bias=True)
    _check_case((1, 10, 7, 9), 15, 5, 1, 1, with_bias=True)
    offset_params, _ = _params(
        torch.zeros((1, 16, 7, 9)), torch.zeros((24, 2, 3, 3)),
        torch.zeros((24,)), 2, 1, 8, storage_offset=7,
    )
    assert "int linear_index = 7 +" in convolution._grouped_3x3_shader_source(offset_params).decode("ascii")
    print("OpenGL grouped Conv tests: PASS")
    if "--gpu" in sys.argv:
        _run_live_comparison()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
