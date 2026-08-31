"""Metadata-only operations for OpenGL-backed MatrixMan tensors."""

from __future__ import annotations

import math

from . import diagnostics
from ...tensor import MatrixManTensor
from .storage import contiguous_strides


def _validate(shape):
    validate_supported_shape(shape)


def _trace(message):
    diagnostics.trace(message)


def validate_supported_shape(shape: tuple[int, ...]) -> None:
    if len(shape) == 0:
        raise RuntimeError("gm45 does not support scalar tensors yet")
    if len(shape) not in {1, 2, 3, 4}:
        raise RuntimeError("gm45 supports only 1D, 2D, 3D, and 4D NCHW tensors")
    if any(size <= 0 for size in shape):
        raise RuntimeError("gm45 does not support empty dimensions")
    if len(shape) == 4 and shape[0] != 1:
        raise RuntimeError("gm45 4D support is NCHW with batch size 1 only")


def is_contiguous_logical(tensor) -> bool:
    return tensor._logical_strides == contiguous_strides(tuple(int(v) for v in tensor.shape))


def require_contiguous_logical(tensor, op_name: str) -> None:
    if not is_contiguous_logical(tensor):
        raise RuntimeError(
            f"gm45 {op_name} requires contiguous logical storage. "
            "Broadcast-expanded tensors are currently metadata-only and are not supported by this shader path."
        )


def normalize_shape(shape_arg, old_numel: int) -> tuple[int, ...]:
    import torch
    if isinstance(shape_arg, torch.Size):
        raw = list(shape_arg)
    elif isinstance(shape_arg, (list, tuple)):
        raw = list(shape_arg)
    else:
        raw = [shape_arg]
    raw = [int(v) for v in raw]
    unknown = [i for i, v in enumerate(raw) if v == -1]
    if len(unknown) > 1:
        raise RuntimeError("gm45 view/reshape accepts at most one inferred dimension")
    known = math.prod(v for v in raw if v != -1)
    if unknown:
        if old_numel % known != 0:
            raise RuntimeError("gm45 view/reshape shape is incompatible with tensor size")
        raw[unknown[0]] = old_numel // known
    if math.prod(raw) != old_numel:
        raise RuntimeError("gm45 view/reshape shape is incompatible with tensor size")
    shape = tuple(raw)
    _validate(shape)
    return shape


def metadata_view(tensor, shape: tuple[int, ...], op_name: str, logical_strides=None):
    if logical_strides is None and not is_contiguous_logical(tensor):
        raise RuntimeError(f"gm45 {op_name} currently supports only contiguous logical input")
    strides = logical_strides or contiguous_strides(shape)
    _trace(
        f"gm45.{op_name}:\n"
        f"  {list(tensor.shape)} -> {list(shape)}\n"
        f"  metadata only; texture #{tensor._owner.texture} reused; offset={tensor._storage_offset}; "
        f"strides={list(strides)}"
    )
    return MatrixManTensor._from_owner(tensor._owner, shape, tensor._storage_offset, strides)


def metadata_transpose(args):
    tensor = args[0]
    dim0, dim1 = int(args[1]), int(args[2])
    if not isinstance(tensor, MatrixManTensor):
        raise RuntimeError("gm45 transpose requires a MatrixManTensor input")
    rank = len(tensor.shape)
    if dim0 < 0:
        dim0 += rank
    if dim1 < 0:
        dim1 += rank
    if dim0 < 0 or dim0 >= rank or dim1 < 0 or dim1 >= rank:
        raise RuntimeError(f"gm45 transpose dimensions out of range: dim0={args[1]}, dim1={args[2]}, rank={rank}")
    shape = list(int(v) for v in tensor.shape)
    strides = list(tensor._logical_strides)
    shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
    strides[dim0], strides[dim1] = strides[dim1], strides[dim0]
    _trace(
        "gm45.transpose:\n"
        f"  input texture #{tensor._owner.texture} shape={list(tensor.shape)} offset={tensor._storage_offset} "
        f"strides={list(tensor._logical_strides)}\n"
        f"  dims=({dim0}, {dim1}) -> shape={shape} strides={strides}\n"
        "  metadata only; no shader copy or GPU readback"
    )
    return MatrixManTensor._from_owner(tensor._owner, tuple(shape), tensor._storage_offset, tuple(strides))


def metadata_unsqueeze(tensor, dim: int):
    rank = len(tensor.shape)
    normalized_dim = dim + rank + 1 if dim < 0 else dim
    if normalized_dim < 0 or normalized_dim > rank:
        raise RuntimeError(f"gm45 unsqueeze dim out of range: dim={dim}, rank={rank}")
    shape = tuple(int(v) for v in tensor.shape)
    strides = tensor._logical_strides
    inserted_stride = 1 if normalized_dim == rank else strides[normalized_dim] * shape[normalized_dim]
    new_shape = shape[:normalized_dim] + (1,) + shape[normalized_dim:]
    new_strides = strides[:normalized_dim] + (inserted_stride,) + strides[normalized_dim:]
    _validate(new_shape)
    return metadata_view(tensor, new_shape, "unsqueeze", new_strides)


def metadata_squeeze(tensor, dim=None):
    shape = tuple(int(v) for v in tensor.shape)
    strides = tensor._logical_strides
    if dim is None:
        keep = [index for index, size in enumerate(shape) if size != 1]
    else:
        normalized_dim = int(dim)
        if normalized_dim < 0:
            normalized_dim += len(shape)
        if normalized_dim < 0 or normalized_dim >= len(shape):
            raise RuntimeError(f"gm45 squeeze dim out of range: dim={dim}, rank={len(shape)}")
        keep = list(range(len(shape)))
        if shape[normalized_dim] == 1:
            keep.pop(normalized_dim)
    new_shape = tuple(shape[index] for index in keep) or (1,)
    new_strides = tuple(strides[index] for index in keep) or (strides[0] if strides else 1,)
    _validate(new_shape)
    return metadata_view(tensor, new_shape, "squeeze", new_strides)


def metadata_expand(args, kwargs):
    input_tensor = args[0]
    requested = tuple(int(v) for v in args[1])
    implicit = bool(kwargs.get("implicit", args[2] if len(args) > 2 else False))
    if not isinstance(input_tensor, MatrixManTensor):
        raise RuntimeError("gm45 expand requires a MatrixManTensor input")
    old_shape = tuple(int(v) for v in input_tensor.shape)
    old_strides = input_tensor._logical_strides
    if len(requested) != len(old_shape):
        raise RuntimeError("gm45 expand currently supports only same-rank expand")
    new_shape, new_strides, expanded_dims = [], [], []
    has_minus_one = False
    for dim, (old_size, old_stride, requested_size) in enumerate(zip(old_shape, old_strides, requested)):
        if requested_size == -1:
            has_minus_one = True
            new_size = old_size
        else:
            new_size = requested_size
        if new_size == old_size:
            new_stride = old_stride
        elif old_size == 1 and new_size > 1:
            new_stride = 0
            expanded_dims.append(dim)
        else:
            raise RuntimeError(f"gm45 expand cannot expand dimension {dim} from {old_size} to {new_size}")
        new_shape.append(new_size)
        new_strides.append(new_stride)
    shape, strides = tuple(new_shape), tuple(new_strides)
    _validate(shape)
    _trace(
        "gm45.expand:\n"
        f"  input texture #{input_tensor._owner.texture} shape={list(old_shape)} offset={input_tensor._storage_offset} "
        f"strides={list(old_strides)}\n"
        f"  requested={list(requested)} normalized={list(shape)} implicit={implicit}\n"
        f"  expanded_dims={expanded_dims} has_minus_one={has_minus_one}\n"
        f"  metadata only; output strides={list(strides)}; no shader copy or GPU readback"
    )
    return MatrixManTensor._from_owner(input_tensor._owner, shape, input_tensor._storage_offset, strides)


def metadata_split(args, kwargs) -> tuple["MatrixManTensor", ...]:
    input_tensor = args[0]
    split_size_or_sections = args[1]
    dim = int(args[2]) if len(args) > 2 else int(kwargs.get("dim", 0))
    if not isinstance(input_tensor, MatrixManTensor):
        raise RuntimeError("gm45 split requires a MatrixManTensor input")
    if input_tensor._owner.layout.kind != "packed_rgba":
        raise RuntimeError("gm45 split currently supports only packed_rgba tensor storage")
    require_contiguous_logical(input_tensor, "split")

    shape = tuple(int(v) for v in input_tensor.shape)
    if dim < 0:
        dim += len(shape)
    if dim < 0 or dim >= len(shape):
        raise RuntimeError("gm45 split dim out of range")

    if isinstance(split_size_or_sections, int):
        split_size = int(split_size_or_sections)
        if split_size <= 0:
            raise RuntimeError("gm45 split_size must be positive")
        sections = []
        remaining = shape[dim]
        while remaining > 0:
            take = min(split_size, remaining)
            sections.append(take)
            remaining -= take
    elif isinstance(split_size_or_sections, (list, tuple)):
        sections = [int(v) for v in split_size_or_sections]
        split_size = None
        if any(v <= 0 for v in sections) or sum(sections) != shape[dim]:
            raise RuntimeError("gm45 split sections must be positive and sum to the selected dimension")
    else:
        raise RuntimeError("gm45 split expects an int split size or list of section sizes")

    # With contiguous flattened storage, a split is metadata-only only when each
    # output is one contiguous run. Current YOLO uses 4D NCHW channel splitting
    # and 3D [batch, channel, anchor] splitting in the DFL box decoder.
    is_nchw_split = len(shape) == 4 and shape[0] == 1 and dim == 1
    is_dfl_split = len(shape) == 3 and shape[0] == 1 and dim == 1 and split_size == 2
    if not (is_nchw_split or is_dfl_split):
        raise RuntimeError(
            "gm45 split currently supports only batch-1 NCHW channel splits and "
            "the YOLO 3D [batch,channel,anchor] split_size=2 case"
        )

    inner_block = math.prod(shape[dim + 1 :])
    outputs = []
    logical_start = 0
    trace_lines = [
        "gm45.split:",
        f"  input texture #{input_tensor._owner.texture}",
        f"  shape {list(shape)}",
        f"  dim={dim}",
        f"  split_size={split_size if split_size is not None else sections}",
    ]
    for index, section in enumerate(sections):
        out_shape = shape[:dim] + (section,) + shape[dim + 1 :]
        out_offset = input_tensor._storage_offset + logical_start * inner_block
        outputs.append(MatrixManTensor._from_owner(input_tensor._owner, out_shape, out_offset))
        trace_lines.append(f"  output {index} offset={out_offset} shape={list(out_shape)}")
        logical_start += section
    trace_lines.append("  metadata only; no shader copy or GPU readback")
    _trace("\n".join(trace_lines))
    return tuple(outputs)
