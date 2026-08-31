#!/usr/bin/env python3
"""Smallest MatrixMan tensor example."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Make ``python3 demo/example/pytorch_example.py`` work from any directory.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drivers import matrixman


def main() -> int:
    torch.manual_seed(45)
    x_cpu = torch.randn((2, 2), dtype=torch.float32)
    other_cpu = torch.randn((2, 2), dtype=torch.float32)
    x = matrixman.to_device(x_cpu)
    other = matrixman.to_device(other_cpu)

    # This addition is executed by MatrixMan's GLSL elementwise kernel.
    y = x + other

    # Readback is explicit and happens only after the GPU operation.
    result = y.cpu()

    expected = x_cpu + other_cpu
    print("CPU tensor:", x_cpu)
    print("Other CPU tensor:", other_cpu)
    print("MatrixMan tensor:", type(x).__name__, "device=", x.device)
    print("Result:", result)
    print("Expected:", expected)
    print("torch.allclose:", bool(torch.allclose(result, expected)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
