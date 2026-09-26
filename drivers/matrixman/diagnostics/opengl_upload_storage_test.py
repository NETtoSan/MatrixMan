"""Check that ordinary tensor uploads use canonical packed RGBA storage."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from drivers import matrixman

    matrixman.config.backend = "opengl"
    matrixman.init()
    try:
        for shape in ((2, 2), (2, 3), (4, 4)):
            tensor = matrixman.to_device(torch.arange(1, 1 + torch.tensor(shape).prod(), dtype=torch.float32).reshape(shape))
            assert tensor._owner.layout.kind == "packed_rgba", (shape, tensor._owner.layout.kind)
        print("OpenGL canonical upload storage: PASS")
    finally:
        matrixman.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
