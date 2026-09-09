"""Numerical and draw-count checks for the OpenGL prepared Conv/BN path."""

from __future__ import annotations

import torch

from drivers import matrixman


def _run(model, value, prepared: bool):
    matrixman.config.backend = "opengl"
    matrixman.config.preparedExecution = prepared
    matrixman.config.profile = True
    matrixman.config.profileDetail = False
    matrixman.init()
    matrixman.profile_reset()
    with torch.no_grad():
        result = model(matrixman.to_device(value)).cpu()
    from drivers.matrixman.backends.opengl import profiling
    counters = dict(profiling.counters)
    matrixman.shutdown()
    return result, counters


def main() -> int:
    torch.manual_seed(7)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1, bias=False),
        torch.nn.BatchNorm2d(4),
        torch.nn.SiLU(inplace=True),
    ).eval()
    value = torch.randn(1, 3, 8, 8)
    with torch.no_grad():
        reference = model(value)

    prepared, prepared_counters = _run(model, value, True)
    ordinary, ordinary_counters = _run(model, value, False)
    prepared_error = float((prepared - reference).abs().max())
    ordinary_error = float((ordinary - reference).abs().max())
    if not torch.allclose(prepared, reference, rtol=1e-4, atol=1e-5):
        raise AssertionError(f"prepared Conv/BN output mismatch: {prepared_error}")
    if not torch.allclose(ordinary, reference, rtol=1e-4, atol=1e-5):
        raise AssertionError(f"ordinary output mismatch: {ordinary_error}")
    if prepared_counters.get("fused_batch_norm_calls", 0) < 1:
        raise AssertionError("prepared Conv/BN path did not fuse BatchNorm")
    prepared_draws = int(prepared_counters.get("draw_calls", 0))
    ordinary_draws = int(ordinary_counters.get("draw_calls", 0))
    if prepared_draws >= ordinary_draws:
        raise AssertionError(
            f"prepared path did not reduce draws: prepared={prepared_draws} ordinary={ordinary_draws}"
        )

    # A source-parameter mutation must produce a new folded resource rather
    # than reusing the old effective Conv parameters.
    matrixman.config.backend = "opengl"
    matrixman.config.preparedExecution = True
    matrixman.config.profile = True
    matrixman.init()
    with torch.no_grad():
        first = model(matrixman.to_device(value)).cpu()
        model[1].weight[0].add_(0.25)
        second = model(matrixman.to_device(value)).cpu()
        mutated_reference = model(value)
    matrixman.shutdown()
    if not torch.allclose(second, mutated_reference, rtol=1e-4, atol=1e-5):
        raise AssertionError("mutated BatchNorm parameter reused stale folded resources")
    if torch.allclose(first, second):
        raise AssertionError("BatchNorm mutation did not affect prepared output")

    print("OpenGL prepared execution tests: PASS")
    print(f"  max_abs_prepared={prepared_error:.8g} max_abs_ordinary={ordinary_error:.8g}")
    print(f"  draws prepared={prepared_draws} ordinary={ordinary_draws}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
