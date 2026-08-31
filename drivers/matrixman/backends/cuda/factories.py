"""PrivateUse1 factory boundary for the CUDA backend."""

from __future__ import annotations

import math

import torch


_aten_privateuse1_lib = None


def _unsupported(name: str):
    def factory(*args, **kwargs):
        del args, kwargs
        raise NotImplementedError(f"MatrixMan/CUDA: {name} not implemented")

    return factory


def _cuda_context():
    from .backend import CudaBackend
    from ...backend import get_backend

    backend = get_backend()
    if not isinstance(backend, CudaBackend):
        raise RuntimeError("MatrixMan/CUDA: CUDA factory called while CUDA is not selected")
    return backend


def _validate_device(device) -> None:
    if device is not None and str(device) not in {
        "matrixman", "matrixman:0", "privateuseone", "privateuseone:0"
    }:
        raise RuntimeError(f"MatrixMan/CUDA: factory got unsupported device {device}")


def _validate_options(op_name, dtype, layout, device, pin_memory) -> None:
    if dtype not in {None, torch.float32, torch.uint8}:
        raise NotImplementedError(
            f"MatrixMan/CUDA: {op_name} supports float32 and storage-only uint8, got {dtype}"
        )
    if layout not in {None, torch.strided}:
        raise NotImplementedError(
            f"MatrixMan/CUDA: {op_name} supports strided layout only, got {layout}"
        )
    if pin_memory:
        raise NotImplementedError(f"MatrixMan/CUDA: {op_name} does not support pin_memory")
    _validate_device(device)


def _shape(values, op_name):
    try:
        result = tuple(int(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MatrixMan/CUDA: {op_name} received an invalid shape") from exc
    if any(value < 0 for value in result):
        raise ValueError(f"MatrixMan/CUDA: {op_name} shape values must be non-negative")
    return result


def _storage_numel(shape, strides) -> int:
    if any(size == 0 for size in shape):
        return 0
    return 1 + sum((size - 1) * stride for size, stride in zip(shape, strides))


def _dtype_or_default(dtype):
    return torch.float32 if dtype is None else dtype


def _itemsize(dtype) -> int:
    return {torch.float32: 4, torch.uint8: 1}[_dtype_or_default(dtype)]


def empty_cuda(
    size, *, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None
):
    del memory_format
    _validate_options("torch.empty", dtype, layout, device, pin_memory)
    shape = _shape(size, "torch.empty")
    strides = []
    stride = 1
    for value in reversed(shape):
        strides.insert(0, stride)
        stride *= value
    return empty_strided_cuda(
        shape,
        tuple(strides),
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )


def empty_strided_cuda(
    size, stride, *, dtype=None, layout=None, device=None, pin_memory=False
):
    _validate_options("torch.empty_strided", dtype, layout, device, pin_memory)
    shape = _shape(size, "torch.empty_strided")
    strides = tuple(int(value) for value in stride)
    if len(strides) != len(shape) or any(value < 0 for value in strides):
        raise ValueError("MatrixMan/CUDA: torch.empty_strided requires valid non-negative strides")

    backend = _cuda_context()
    from .backend import CudaTensorOwner
    from ...tensor import MatrixManTensor

    storage_numel = _storage_numel(shape, strides)
    dtype = _dtype_or_default(dtype)
    pointer = backend.execution.allocate(max(1, storage_numel) * _itemsize(dtype))
    owner = CudaTensorOwner(
        backend.execution,
        pointer,
        shape,
        strides,
        storage_numel=storage_numel,
        dtype=dtype,
    )
    try:
        return MatrixManTensor._from_owner(
            owner,
            shape,
            logical_strides=strides,
        )
    except Exception:
        owner.release()
        raise


def _arange_count(start: float, end: float, step: float) -> int:
    if step == 0:
        raise ValueError("MatrixMan/CUDA: torch.arange step must be nonzero")
    distance = (end - start) / step
    return max(0, int(math.ceil(distance))) if distance > 0 else 0


def _arange_cuda(start, end, step=1, *, dtype=None, layout=None, device=None, pin_memory=False):
    _validate_options("torch.arange", dtype, layout, device, pin_memory)
    dtype = _dtype_or_default(dtype)
    if dtype != torch.float32:
        raise NotImplementedError(
            f"MatrixMan/CUDA: torch.arange supports float32 only, got {dtype}"
        )
    start = float(start)
    end = float(end)
    step = float(step)
    count = _arange_count(start, end, step)
    output = empty_strided_cuda(
        (count,),
        (1,),
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )
    if count:
        backend = _cuda_context()
        try:
            backend.execution.arange(output._owner.pointer, start, step, count)
        except Exception:
            output._owner.release()
            raise
    return output


def arange_cuda(end, *, dtype=None, layout=None, device=None, pin_memory=False):
    return _arange_cuda(
        0, end, 1, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


def arange_start_cuda(start, end, *, dtype=None, layout=None, device=None, pin_memory=False):
    return _arange_cuda(
        start, end, 1, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


def arange_start_step_cuda(
    start, end, step=1, *, dtype=None, layout=None, device=None, pin_memory=False
):
    return _arange_cuda(
        start, end, step, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


def install_privateuse1_factory_kernels() -> None:
    global _aten_privateuse1_lib
    if _aten_privateuse1_lib is not None:
        return

    lib = torch.library.Library("aten", "IMPL", "PrivateUse1")
    lib.impl("empty.memory_format", empty_cuda)
    lib.impl("empty_strided", empty_strided_cuda)
    lib.impl("arange", arange_cuda)
    lib.impl("arange.start", arange_start_cuda)
    lib.impl("arange.start_step", arange_start_step_cuda)
    _aten_privateuse1_lib = lib
