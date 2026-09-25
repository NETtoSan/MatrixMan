"""Small native registration shim for PyTorch PrivateUse1 device plumbing."""

from __future__ import annotations

from pathlib import Path

import torch


_extension = None


def register_hooks() -> None:
    """Build/load and register MatrixMan's PrivateUse1 C++ hooks once."""
    global _extension
    if _extension is None:
        from torch.utils.cpp_extension import IS_WINDOWS, load

        source = Path(__file__).with_name("autograd_privateuse1.cpp")
        build_directory = Path(__file__).with_name("_build")
        build_directory.mkdir(exist_ok=True)
        _extension = load(
            name="matrixman_privateuse1_hooks",
            sources=[str(source)],
            build_directory=str(build_directory),
            extra_cflags=["/O2"] if IS_WINDOWS else ["-O2"],
            verbose=False,
        )
    _extension.register_hooks()
