"""Static and optional live checks for GPU-materialized MatrixMan expand."""

from __future__ import annotations

import sys

import torch

from drivers.matrixman.backends.opengl.ops import broadcast


def _static_checks() -> None:
    source = broadcast._broadcast_shader_source(
        ((1, 3), (3, 1), (4, 3), 2, 4, 4, 4)
    ).decode("ascii")
    assert "broadcast_at(base)" in source
    assert "source_index = 2" in source

    cases = [
        (torch.tensor(2.0), (5,)),
        (torch.tensor(2.0), (2, 3)),
        (torch.arange(3.0), (2, 3)),
        (torch.ones(1, 3), (4, 3)),
        (torch.ones(2, 1, 4), (2, 5, 4)),
    ]
    for source_tensor, shape in cases:
        assert tuple(source_tensor.expand(shape).shape) == shape
    assert tuple(torch.ones(2, 1, 4).expand(2, -1, 4).shape) == (2, 1, 4)
    try:
        torch.ones(2, 3).expand(2, 4)
    except RuntimeError:
        pass
    else:
        raise AssertionError("incompatible expansion was accepted")


def _live_checks() -> None:
    from drivers import matrixman

    matrixman.config.backend = "opengl"
    matrixman.init()
    try:
        cpu_cases = [
            (torch.arange(3.0), (2, 3)),
            (torch.ones(1, 3), (4, 3)),
            (torch.ones(2, 1, 4), (2, 5, 4)),
        ]
        for cpu, shape in cpu_cases:
            actual = matrixman.to_device(cpu).expand(shape)
            torch.testing.assert_close(actual.cpu(), cpu.expand(shape), rtol=0, atol=0)
            assert tuple(actual.shape) == shape
            assert actual.device.type == "privateuseone"

        base_cpu = torch.arange(30, dtype=torch.float32).reshape(2, 3, 5)
        view = matrixman.to_device(base_cpu)[..., 1:]
        actual = view.expand(2, 3, 4)
        torch.testing.assert_close(actual.cpu(), base_cpu[..., 1:].expand(2, 3, 4), rtol=0, atol=0)

        scalar = matrixman.to_device(torch.arange(3, dtype=torch.float32)).sum()
        actual = scalar.expand(4)
        torch.testing.assert_close(actual.cpu(), torch.tensor(3.0).expand(4), rtol=0, atol=0)
        print("OpenGL expand live checks: PASS")
    finally:
        matrixman.shutdown()


def main() -> int:
    _static_checks()
    print("OpenGL expand static checks: PASS")
    if "--gpu" in sys.argv:
        _live_checks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
