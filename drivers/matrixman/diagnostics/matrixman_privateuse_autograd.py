"""Probe PrivateUse1 autograd/device plumbing without testing gradient math."""

from __future__ import annotations

import torch

from drivers import matrixman


def main() -> None:
    x = matrixman.to_device(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
    print("torch_version:", torch.__version__)
    print("device:", x.device)
    print("get_device:", x.get_device())
    try:
        x.requires_grad_(True)
        print("requires_grad: PASS", x.requires_grad)
    except Exception as error:
        print("requires_grad: BLOCKED", type(error).__name__, str(error))
        return

    try:
        y = x + x
        print("graph_output_requires_grad:", y.requires_grad)
        print("graph_output_device:", y.device)
    except Exception as error:
        print("graph: BLOCKED", type(error).__name__, str(error))

    try:
        y.sum().backward()
        print("backward: UNEXPECTEDLY_SUPPORTED")
    except Exception as error:
        print("backward: CLEAN_ERROR", type(error).__name__, str(error))


if __name__ == "__main__":
    main()
