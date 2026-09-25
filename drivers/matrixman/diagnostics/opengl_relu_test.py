"""CPU-reference and optional live checks for OpenGL ReLU dispatch."""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from drivers.matrixman.backends.opengl.ops import activation
from drivers.matrixman.backends.opengl.storage import packed_atlas_size


def _check_shader_source():
    input_tw, input_th = packed_atlas_size(24)
    params = (24, 7, input_tw, input_th, input_tw)
    source = activation._relu_shader_source(params).decode("ascii")
    assert "return max(read_packed(linear_index + 7), 0.0);" in source


def _reference_cases():
    torch.manual_seed(991)
    return [
        torch.tensor([-2.0, 0.0, 3.0, -0.5, 1.25], dtype=torch.float32),
        torch.randn((1, 3, 8, 10), dtype=torch.float32),
        torch.randn((2, 5, 4), dtype=torch.float32),
    ]


def _run_live():
    from drivers import matrixman
    from drivers.matrixman.tensor import MatrixManTensor

    matrixman.config.backend = "opengl"
    matrixman.init()
    try:
        for value in _reference_cases():
            expected = F.relu(value)
            functional = F.relu(matrixman.to_device(value))
            actual = functional.cpu()
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

            inplace_input = matrixman.to_device(value)
            original_owner = inplace_input._owner
            result = F.relu(inplace_input, inplace=True)
            assert result is inplace_input
            assert result._owner is not original_owner
            torch.testing.assert_close(result.cpu(), expected, rtol=1e-5, atol=1e-5)

        base_cpu = torch.tensor(
            [[[-2.0, 1.0, -3.0, 4.0], [5.0, -6.0, 7.0, -8.0]]],
            dtype=torch.float32,
        )
        base = matrixman.to_device(base_cpu)
        view_shape = (1, 1, 2, 3)
        offset = 2
        view = MatrixManTensor._from_owner(base._owner, view_shape, offset)
        expected_view = F.relu(base_cpu.reshape(-1)[offset:offset + 6].reshape(view_shape))
        actual_view = F.relu(view).cpu()
        torch.testing.assert_close(actual_view, expected_view, rtol=1e-5, atol=1e-5)
        inplace_view = MatrixManTensor._from_owner(base._owner, view_shape, offset)
        F.relu(inplace_view, inplace=True)
        torch.testing.assert_close(inplace_view.cpu(), expected_view, rtol=1e-5, atol=1e-5)
        print("OpenGL ReLU live numerical tests: PASS")
    finally:
        matrixman.shutdown()


def main() -> int:
    _check_shader_source()
    for value in _reference_cases():
        assert torch.equal(F.relu(value), value.clamp_min(0))
    if "--gpu" in sys.argv:
        _run_live()
    print("OpenGL ReLU tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
