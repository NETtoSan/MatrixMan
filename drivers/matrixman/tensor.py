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


def _cpu_layout_from_values(values, shape, strides, storage_offset):
    """Place logical readback values into a CPU tensor with matching metadata."""
    import numpy as np

    shape = tuple(int(value) for value in shape)
    strides = tuple(int(value) for value in strides)
    storage_offset = int(storage_offset)
    if any(stride < 0 for stride in strides) or storage_offset < 0:
        raise RuntimeError("MatrixMan CPU readback does not support negative layout metadata")
    storage_size = storage_offset + (max_storage_index(shape, strides) + 1 if numel(shape) else 0)
    base = torch.empty(storage_size, dtype=values.dtype, device="cpu")
    result = torch.as_strided(base, shape, strides, storage_offset)
    source = values.detach()
    if 0 not in strides:
        result.copy_(source)
    else:
        # Expanded views have intentionally overlapping destinations.  A
        # bulk copy is rejected by PyTorch for those layouts, so assign the
        # logical values with Torch scalars while preserving the zero stride.
        for index in np.ndindex(shape):
            result[index] = source[index]
    return result


def readback_tensor(tensor: "MatrixManTensor") -> torch.Tensor:
    """Explicitly copy a MatrixMan tensor to an ordinary CPU tensor."""
    if not isinstance(tensor, MatrixManTensor):
        raise TypeError("MatrixMan CPU readback requires a MatrixManTensor")
    if tensor.dtype != torch.float32:
        raise NotImplementedError(
            f"MatrixMan CPU readback supports float32 only, got {tensor.dtype}"
        )

    shape = tuple(int(value) for value in tensor.shape)
    strides = tuple(int(value) for value in tensor._logical_strides)
    offset = int(tensor._storage_offset)
    owner = tensor._owner
    if owner.layout.kind == "cuda_linear":
        import numpy as np

        raw = owner.execution.from_device(owner.pointer, (owner.layout.numel,))
        if offset < 0 or (numel(shape) and offset + max_storage_index(shape, strides) >= owner.layout.numel):
            raise RuntimeError("MatrixMan CUDA readback view is outside storage")
        values_array = np.empty(shape, dtype=np.float32)
        for index in np.ndindex(shape):
            values_array[index] = raw[offset + sum(i * stride for i, stride in zip(index, strides))]
        return _cpu_layout_from_values(torch.from_numpy(values_array), shape, strides, offset)

    if owner.layout.kind.startswith("matrix") or owner.layout.kind in {"packed_rgba", "empty"}:
        from .backends.opengl.tensor import readback_tensor as opengl_readback

        values = opengl_readback(owner, shape, offset, strides)
        return _cpu_layout_from_values(values, shape, strides, offset)
    raise RuntimeError(f"MatrixMan CPU readback cannot handle owner kind {owner.layout.kind!r}")


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


def expand_shape_strides(input_shape, input_strides, requested_shape):
    """Return PyTorch-style expanded shape/strides without changing storage."""
    source_shape = tuple(int(size) for size in input_shape)
    source_strides = tuple(int(stride) for stride in input_strides)
    requested = tuple(int(size) for size in requested_shape)
    if len(source_shape) != len(source_strides):
        raise ValueError("MatrixMan expand source shape and strides must have the same rank")
    if len(requested) < len(source_shape):
        raise ValueError("MatrixMan expand cannot remove dimensions")

    offset = len(requested) - len(source_shape)
    output_shape = []
    output_strides = []
    for output_index, requested_size in enumerate(requested):
        if output_index < offset:
            if requested_size < 0:
                raise ValueError("MatrixMan expand does not allow -1 on prepended dimensions")
            output_shape.append(requested_size)
            output_strides.append(0)
            continue
        source_index = output_index - offset
        source_size = source_shape[source_index]
        source_stride = source_strides[source_index]
        size = source_size if requested_size == -1 else requested_size
        if size < 0:
            raise ValueError("MatrixMan expand sizes must be non-negative or -1")
        if source_size != size:
            if source_size != 1 or source_size == 0 or size == 0:
                raise ValueError(
                    f"MatrixMan expand cannot change dimension {source_index} "
                    f"from {source_size} to {size}"
                )
            source_stride = 0
        output_shape.append(size)
        output_strides.append(source_stride)
    return tuple(output_shape), tuple(output_strides)


def unsqueeze_shape_strides(input_shape, input_strides, dimension):
    """Insert a size-one dimension using PyTorch-compatible view strides."""
    shape = tuple(int(size) for size in input_shape)
    strides = tuple(int(stride) for stride in input_strides)
    rank = len(shape)
    if len(strides) != rank:
        raise ValueError("MatrixMan unsqueeze source shape and strides must have the same rank")
    dimension = int(dimension)
    if dimension < 0:
        dimension += rank + 1
    if dimension < 0 or dimension > rank:
        raise IndexError("MatrixMan unsqueeze dimension is out of range")
    if dimension == rank:
        inserted_stride = 1
    else:
        inserted_stride = strides[dimension] * shape[dimension]
    return (
        shape[:dimension] + (1,) + shape[dimension:],
        strides[:dimension] + (inserted_stride,) + strides[dimension:],
    )


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
