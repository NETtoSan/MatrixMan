"""Selected-backend diagnostic for explicit MatrixMan tensor CPU readback."""

from __future__ import annotations

import torch

from drivers import matrixman
from drivers.matrixman.diagnostics.backend_helpers import describe_storage


def check(name, gpu_tensor, expected):
    cpu_tensor = gpu_tensor.cpu()
    if not torch.equal(cpu_tensor, expected):
        raise RuntimeError(f"{name} readback does not match the CPU reference")
    print(
        f"{name}: shape={list(cpu_tensor.shape)} strides={list(cpu_tensor.stride())} "
        f"offset={cpu_tensor.storage_offset()} storage={describe_storage(gpu_tensor)} PASS"
    )


def main() -> int:
    source_cpu = torch.arange(24, dtype=torch.float32).reshape(1, 4, 2, 3).contiguous()
    source = matrixman.to_device(source_cpu)
    check("contiguous", source, source_cpu)

    split, _ = torch.chunk(source, 2, dim=1)
    split_reference, _ = torch.chunk(source_cpu, 2, dim=1)
    check("split-derived", split, split_reference)

    transposed = source.transpose(2, 3)
    check("transposed", transposed, source_cpu.transpose(2, 3))

    expanded_base_cpu = torch.arange(3, dtype=torch.float32).reshape(1, 3).contiguous()
    expanded = matrixman.to_device(expanded_base_cpu).expand(4, 3)
    check("zero-stride expanded", expanded, expanded_base_cpu.expand(4, 3))

    matrixman.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
