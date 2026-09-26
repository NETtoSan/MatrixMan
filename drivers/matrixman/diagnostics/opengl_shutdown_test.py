"""Focused OpenGL runtime shutdown/lifecycle diagnostic."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drivers import matrixman


def main() -> int:
    matrixman.config.backend = "opengl"

    matrixman.init()
    x = matrixman.to_device(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
    _ = x + x
    matrixman.shutdown()
    matrixman.shutdown()
    print("shutdown twice: PASS")

    try:
        matrixman.init()
        x = matrixman.to_device(torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32))
        _ = x + x
        raise RuntimeError("intentional shutdown diagnostic exception")
    except RuntimeError as error:
        if str(error) != "intentional shutdown diagnostic exception":
            raise
        print("shutdown from exception: PASS")
    finally:
        matrixman.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
