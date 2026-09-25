"""CPU-reference and optional live checks for OpenGL ``aten.add_.Tensor``."""

from __future__ import annotations

import sys

import torch


def _reference_cases():
    return [
        torch.tensor([-2.0, 0.0, 3.0, -0.5, 1.25], dtype=torch.float32),
        torch.randn((1, 4, 8, 10), dtype=torch.float32, generator=torch.Generator().manual_seed(312)),
    ]


def _run_live():
    from drivers import matrixman
    from drivers.matrixman.tensor import MatrixManTensor

    matrixman.config.backend = "opengl"
    matrixman.init()
    try:
        for value in _reference_cases():
            rhs_cpu = torch.randn(value.shape, dtype=torch.float32, generator=torch.Generator().manual_seed(313))
            expected = value + 1.75 * rhs_cpu

            lhs = matrixman.to_device(value)
            rhs = matrixman.to_device(rhs_cpu)
            original_owner = lhs._owner
            returned = lhs.add_(rhs, alpha=1.75)
            assert returned is lhs
            assert lhs._owner is not original_owner
            torch.testing.assert_close(lhs.cpu(), expected, rtol=1e-5, atol=1e-5)

            iadd_lhs = matrixman.to_device(value)
            iadd_rhs = matrixman.to_device(rhs_cpu)
            iadd_owner = iadd_lhs._owner
            iadd_lhs += iadd_rhs
            assert iadd_lhs._owner is not iadd_owner
            torch.testing.assert_close(iadd_lhs.cpu(), value + rhs_cpu, rtol=1e-5, atol=1e-5)

            functional_lhs = matrixman.to_device(value)
            functional_rhs = matrixman.to_device(rhs_cpu)
            functional = torch.add(functional_lhs, functional_rhs, alpha=1.75)
            torch.testing.assert_close(functional.cpu(), expected, rtol=1e-5, atol=1e-5)

        base_cpu = torch.arange(40, dtype=torch.float32).reshape(1, 1, 4, 10) - 17
        rhs_cpu = torch.full((1, 1, 2, 6), 2.0, dtype=torch.float32)
        base = matrixman.to_device(base_cpu)
        # The packed functional add supports contiguous logical views with a
        # nonzero storage offset; add_ must preserve the same input handling.
        view = MatrixManTensor._from_owner(base._owner, (1, 1, 2, 6), 3)
        rhs = matrixman.to_device(rhs_cpu)
        expected = base_cpu.reshape(-1)[3:15].reshape(1, 1, 2, 6) + rhs_cpu
        assert view.add_(rhs) is view
        torch.testing.assert_close(view.cpu(), expected, rtol=1e-5, atol=1e-5)
        print("OpenGL add_ live numerical tests: PASS")
    finally:
        matrixman.shutdown()


def main() -> int:
    for value in _reference_cases():
        rhs = torch.ones_like(value)
        expected = value + 1.75 * rhs
        assert torch.equal(expected, value + 1.75 * rhs)
    if "--gpu" in sys.argv:
        _run_live()
    print("OpenGL add_ tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
