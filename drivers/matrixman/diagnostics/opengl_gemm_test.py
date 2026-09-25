"""CPU-reference and optional live checks for rectangular OpenGL GEMM."""

from __future__ import annotations

import sys

import torch

from drivers.matrixman.backends.opengl.ops import matmul


def _source_checks():
    params = (2, 3, 4, "packed_rgba", "packed_rgba", 5, 7, 10, 20, 9, 8, 4,
              "packed_rgba", 2, 4, 5, 1, "1.5", "0.25")
    source = matmul._shader_source(params).decode("ascii")
    assert "row * 3 + kk" in source
    assert "kk * 4 + col" in source
    assert "acc *= 1.5" in source
    assert "read_bias(bias_tex, 2 + (col))" in source


def _reference_checks():
    generator = torch.Generator().manual_seed(712)
    cases = [((1, 512), (512, 1000)), ((2, 3), (3, 4)), ((8, 3), (3, 2)), ((2, 8), (8, 16)), ((4, 4), (4, 4))]
    for left_shape, right_shape in cases:
        left = torch.randn(left_shape, generator=generator)
        right = torch.randn(right_shape, generator=generator)
        assert torch.allclose(left @ right, torch.mm(left, right))

    left = torch.randn((2, 3), generator=generator)
    right = torch.randn((3, 4), generator=generator)
    bias = torch.randn(4, generator=generator)
    expected = 0.25 * bias + 1.5 * (left @ right)
    assert torch.allclose(torch.addmm(bias, left, right, beta=0.25, alpha=1.5), expected)

    full_bias = torch.randn((2, 4), generator=generator)
    expected = 0.5 * full_bias + 2.0 * (left @ right)
    assert torch.allclose(torch.addmm(full_bias, left, right, beta=0.5, alpha=2.0), expected)

    try:
        matmul._validate_gemm_inputs(None, None)
    except RuntimeError as error:
        assert "both inputs" in str(error)
    else:
        raise AssertionError("invalid GEMM operands were accepted")


def _run_live():
    from drivers import matrixman
    from drivers.matrixman.tensor import MatrixManTensor

    matrixman.config.backend = "opengl"
    matrixman.init()
    try:
        generator = torch.Generator().manual_seed(713)
        cases = [((1, 512), (512, 1000)), ((2, 3), (3, 4)), ((8, 3), (3, 2)), ((2, 8), (8, 16)), ((4, 4), (4, 4))]
        for left_shape, right_shape in cases:
            left_cpu = torch.randn(left_shape, generator=generator)
            right_cpu = torch.randn(right_shape, generator=generator)
            actual = torch.mm(matrixman.to_device(left_cpu), matrixman.to_device(right_cpu)).cpu()
            torch.testing.assert_close(actual, left_cpu @ right_cpu, rtol=5e-4, atol=5e-4)

        left_cpu = torch.randn((2, 3), generator=generator)
        right_cpu = torch.randn((3, 4), generator=generator)
        bias_cpu = torch.randn(4, generator=generator)
        actual = torch.addmm(
            matrixman.to_device(bias_cpu), matrixman.to_device(left_cpu), matrixman.to_device(right_cpu),
            beta=0.25, alpha=1.5,
        ).cpu()
        torch.testing.assert_close(actual, torch.addmm(bias_cpu, left_cpu, right_cpu, beta=0.25, alpha=1.5), rtol=5e-4, atol=5e-4)

        base_left_cpu = torch.arange(30, dtype=torch.float32).reshape(1, 30)
        base_right_cpu = torch.arange(30, dtype=torch.float32).reshape(1, 30)
        left_base = matrixman.to_device(base_left_cpu)
        right_base = matrixman.to_device(base_right_cpu)
        left_view = MatrixManTensor._from_owner(left_base._owner, (2, 3), 2)
        right_view = MatrixManTensor._from_owner(right_base._owner, (3, 2), 4)
        expected = base_left_cpu.reshape(-1)[2:8].reshape(2, 3) @ base_right_cpu.reshape(-1)[4:10].reshape(3, 2)
        torch.testing.assert_close(torch.mm(left_view, right_view).cpu(), expected, rtol=5e-4, atol=5e-4)
        print("OpenGL GEMM live numerical tests: PASS")
    finally:
        matrixman.shutdown()


def main() -> int:
    _source_checks()
    _reference_checks()
    if "--gpu" in sys.argv:
        _run_live()
    print("OpenGL GEMM tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
