#!/usr/bin/env python3
"""Focused opt-in CPU materialization audit diagnostic."""

from __future__ import annotations

import torch

from drivers import matrixman
from drivers.matrixman import audit
from drivers.matrixman.backend import get_backend


def main() -> int:
    if not audit.enabled():
        raise RuntimeError("set MATRIXMAN_AUDIT_CPU_LEAKS=1 to run this diagnostic")
    if get_backend().name != "opengl":
        raise RuntimeError("matrixman_cpu_leak_audit_demo requires the OpenGL backend")
    x = matrixman.to_device(torch.ones((5,), dtype=torch.float32))
    y = torch.add(x, x)
    cpu = y.cpu()
    print("explicit .cpu() result:", cpu.tolist())
    torch.ops.aten.empty.memory_format(
        [0], dtype=torch.uint8, device=x.device, layout=torch.strided
    )
    if audit.count("unexpected_cpu_materialization"):
        raise AssertionError("unexpected CPU materialization was reported")
    matrixman.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
