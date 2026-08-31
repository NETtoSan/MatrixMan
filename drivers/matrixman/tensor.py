"""Backend-neutral MatrixMan tensor wrapper and metadata helpers."""

from __future__ import annotations

import warnings

import torch


PRIVATEUSE_DEVICE = torch.device("privateuseone:0")


def numel(shape: tuple[int, ...]) -> int:
    result = 1
    for size in shape:
        result *= int(size)
    return result


def contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    strides = []
    for size in reversed(shape):
        strides.append(stride)
        stride *= int(size)
    return tuple(reversed(strides))


def max_storage_index(shape: tuple[int, ...], strides: tuple[int, ...]) -> int:
    if numel(shape) == 0:
        return 0
    return sum((int(size) - 1) * int(stride) for size, stride in zip(shape, strides))


def infer_view_shape(input_shape, requested_shape) -> tuple[int, ...]:
    """Resolve one inferred dimension for a contiguous metadata-only view."""
    requested = tuple(int(size) for size in requested_shape)
    unknown = [index for index, size in enumerate(requested) if size == -1]
    if len(unknown) > 1 or any(size < -1 for size in requested):
        raise ValueError("MatrixMan view accepts at most one -1 dimension")
    input_count = numel(tuple(int(size) for size in input_shape))
    known_count = numel(tuple(size for size in requested if size != -1))
    if unknown:
        if known_count == 0 or input_count % known_count:
            raise ValueError("MatrixMan view cannot infer a compatible dimension")
        resolved = list(requested)
        resolved[unknown[0]] = input_count // known_count
        requested = tuple(resolved)
    if numel(requested) != input_count:
        raise ValueError("MatrixMan view must preserve the number of elements")
    if any(size < 0 for size in requested):
        raise ValueError("MatrixMan view dimensions must be non-negative")
    return requested


def _impl():
    from . import dispatch

    return dispatch


def is_matrixman_tensor(value) -> bool:
    return isinstance(value, MatrixManTensor)


def is_gm45_tensor(value) -> bool:
    """Deprecated compatibility alias for is_matrixman_tensor."""
    warnings.warn(
        "is_gm45_tensor() is deprecated; use is_matrixman_tensor() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return is_matrixman_tensor(value)


def install_tensor_method() -> None:
    """Install the historical Tensor.gm45() convenience method."""
    def upload(value):
        from .gm45_backend import to_device

        return to_device(value)

    setattr(torch.Tensor, "gm45", upload)


class MatrixManTensor(torch.Tensor):
    """Backend-neutral wrapper around a MatrixMan storage owner."""

    @staticmethod
    def __new__(cls, owner, shape, storage_offset=0, logical_strides=None):
        strides = logical_strides or contiguous_strides(shape)
        dtype = getattr(owner, "dtype", torch.float32)
        return torch.Tensor._make_wrapper_subclass(
            cls, shape, strides=strides, dtype=dtype,
            layout=torch.strided, device=PRIVATEUSE_DEVICE, requires_grad=False,
        )

    def __init__(self, owner, shape, storage_offset=0, logical_strides=None):
        self._owner = owner
        self._shape = tuple(shape)
        self._storage_offset = int(storage_offset)
        self._logical_strides = tuple(logical_strides or contiguous_strides(self._shape))

    @staticmethod
    def _from_owner(owner, shape, storage_offset=0, logical_strides=None):
        strides = tuple(logical_strides or contiguous_strides(shape))
        if len(strides) != len(shape):
            raise RuntimeError("MatrixMan tensor logical strides must match shape rank")
        max_index = max_storage_index(shape, strides)
        if storage_offset < 0 or (numel(shape) > 0 and storage_offset + max_index >= owner.layout.numel):
            raise RuntimeError("MatrixMan tensor view storage offset is outside owner storage")
        return MatrixManTensor(owner, shape, storage_offset, strides)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        return _impl().handle_torch_dispatch(cls, func, types, args, kwargs)

    def __repr__(self):
        storage = self._owner.storage_description
        return (
            f"MatrixManTensor(shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device}, "
            f"{storage}, storage={self._owner.layout.kind}, "
            f"offset={self._storage_offset}, strides={self._logical_strides})"
        )


Gm45Tensor = MatrixManTensor
