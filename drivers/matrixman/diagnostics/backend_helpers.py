"""Small backend-neutral helpers shared by MatrixMan diagnostics."""

from __future__ import annotations

import torch

from drivers.matrixman.backend import get_backend
from drivers.matrixman.tensor import readback_tensor as matrixman_readback


def set_trace_if_supported(matrixman, enabled: bool) -> None:
    """Use the legacy trace API only when the selected backend supports it."""
    if get_backend().name == "opengl":
        matrixman.set_trace(enabled)


def describe_storage(tensor) -> str:
    """Return owner-neutral storage metadata for diagnostic output."""
    return (
        f"backend={get_backend().name} {tensor._owner.storage_description} "
        f"offset={tensor._storage_offset} strides={tuple(tensor._logical_strides)}"
    )


def storage_identity(tensor):
    """Return a comparable identity for the allocation behind a MatrixMan tensor."""
    owner = tensor._owner
    if hasattr(owner, "texture"):
        return ("opengl", owner.texture)
    if hasattr(owner, "pointer"):
        return ("cuda", owner.pointer.value)
    return (type(owner).__name__, id(owner))


def readback_tensor(tensor) -> torch.Tensor:
    """Read a MatrixMan tensor through its selected backend's explicit API."""
    return matrixman_readback(tensor)
