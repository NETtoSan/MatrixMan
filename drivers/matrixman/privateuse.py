"""Shared backend-neutral PrivateUse1 registration for MatrixMan."""

from __future__ import annotations

import torch


_registered = False


class MatrixManDeviceModule:
    """Minimal device module exposed by PyTorch for MatrixMan."""

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def is_initialized() -> bool:
        return True

    @staticmethod
    def _is_in_bad_fork() -> bool:
        return False

    @staticmethod
    def manual_seed_all(seed: int) -> None:
        del seed


def register_privateuse1_backend() -> None:
    """Register MatrixMan's PrivateUse1 name and device module once."""
    global _registered
    if _registered:
        return

    current = torch._C._get_privateuse1_backend_name()
    if current == "privateuseone":
        torch.utils.rename_privateuse1_backend("matrixman")
    elif current != "matrixman":
        raise RuntimeError(
            f"PrivateUse1 is already registered as {current!r}, not 'matrixman'"
        )

    try:
        torch._register_device_module("matrixman", MatrixManDeviceModule)
    except RuntimeError:
        if not hasattr(torch, "matrixman"):
            raise
    _registered = True

