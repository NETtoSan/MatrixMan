"""Backend-neutral PrivateUse1 dispatch routing."""

from __future__ import annotations

import torch

from .backend import get_backend
from .tensor import contiguous_strides, infer_view_shape


def _operator_name(func) -> str:
    names = {
        torch.ops.aten.convolution.default: "Conv2D",
        torch.ops.aten.native_batch_norm.default: "BatchNorm",
        torch.ops.aten.silu_.default: "SiLU",
        torch.ops.aten.add.Tensor: "Add",
        torch.ops.aten.cat.default: "Cat",
        torch.ops.aten.upsample_nearest2d.default: "UpsampleNearest2D",
    }
    return names.get(func, str(func))


def handle_torch_dispatch(cls, func, types, args=(), kwargs=None):
    """Send a PrivateUse1 operation to the selected backend only."""
    backend = get_backend()
    if backend.name == "opengl":
        from .backends.opengl import dispatch as opengl_dispatch

        return opengl_dispatch.handle_torch_dispatch(cls, func, types, args, kwargs)
    if backend.name == "cuda":
        if func is torch.ops.aten.view.default:
            if len(args) < 2:
                raise RuntimeError("MatrixMan/CUDA: malformed view arguments")
            input_tensor = args[0]
            if not hasattr(input_tensor, "_owner") or input_tensor._owner.layout.kind != "cuda_linear":
                raise RuntimeError("MatrixMan/CUDA: view requires a CUDA-backed MatrixManTensor")
            input_shape = tuple(int(value) for value in input_tensor.shape)
            if tuple(int(value) for value in input_tensor._logical_strides) != contiguous_strides(input_shape):
                raise NotImplementedError(
                    "MatrixMan/CUDA: view only supports contiguous tensors"
                )
            output_shape = infer_view_shape(input_shape, args[1])
            return type(input_tensor)._from_owner(
                input_tensor._owner,
                output_shape,
                storage_offset=input_tensor._storage_offset,
                logical_strides=contiguous_strides(output_shape),
            )
        if func is torch.ops.aten.add.Tensor:
            if len(args) < 2:
                raise RuntimeError("MatrixMan/CUDA: malformed add arguments")
            alpha = args[2] if len(args) > 2 else (kwargs or {}).get("alpha", 1)
            owner = backend.add(args[0], args[1], alpha)
            return type(args[0])._from_owner(
                owner,
                owner.shape,
                logical_strides=owner.strides,
            )
        if func is torch.ops.aten.cat.default:
            if not args:
                raise RuntimeError("MatrixMan/CUDA: malformed cat arguments")
            tensors = args[0]
            dimension = args[1] if len(args) > 1 else (kwargs or {}).get("dim", 0)
            owner = backend.cat(tensors, dimension)
            return type(tensors[0])._from_owner(
                owner,
                owner.shape,
                logical_strides=owner.strides,
            )
        if func is torch.ops.aten.upsample_nearest2d.default:
            if len(args) < 4:
                raise RuntimeError("MatrixMan/CUDA: malformed upsample_nearest2d arguments")
            owner = backend.upsample_nearest2d(
                args[0],
                args[1],
                args[2],
                args[3],
            )
            return type(args[0])._from_owner(
                owner,
                owner.shape,
                logical_strides=owner.strides,
            )
        if func is torch.ops.aten.convolution.default:
            if len(args) < 9:
                raise RuntimeError("MatrixMan/CUDA: malformed convolution arguments")
            input_tensor, weight, bias = args[:3]
            transposed = bool(args[6])
            output_padding = tuple(int(value) for value in args[7])
            if transposed or any(output_padding):
                raise NotImplementedError(
                    "MatrixMan/CUDA: transposed convolution and output padding are not implemented"
                )
            owner = backend.convolution(
                input_tensor,
                weight,
                bias,
                args[3],
                args[4],
                args[5],
                args[8],
            )
            return type(input_tensor)._from_owner(
                owner,
                owner.shape,
                logical_strides=owner.strides,
            )
        if func is torch.ops.aten.native_batch_norm.default:
            if len(args) < 8:
                raise RuntimeError("MatrixMan/CUDA: malformed BatchNorm arguments")
            input_tensor, weight, bias, running_mean, running_var = args[:5]
            owner = backend.batch_norm(
                input_tensor,
                weight,
                bias,
                running_mean,
                running_var,
                bool(args[5]),
                args[7],
            )
            output = type(input_tensor)._from_owner(
                owner,
                owner.shape,
                logical_strides=owner.strides,
            )
            return output, torch.empty((0,), dtype=torch.float32), torch.empty((0,), dtype=torch.float32)
        if func is torch.ops.aten.silu_.default or func is torch.ops.aten.silu.default:
            if not args:
                raise RuntimeError("MatrixMan/CUDA: malformed SiLU arguments")
            inplace = func is torch.ops.aten.silu_.default
            if inplace:
                backend.silu(args[0], inplace=True)
                return args[0]
            owner = backend.silu(args[0], inplace=False)
            return type(args[0])._from_owner(
                owner,
                owner.shape,
                logical_strides=owner.strides,
            )
        if func is torch.ops.aten.split.Tensor:
            if len(args) < 3:
                raise RuntimeError("MatrixMan/CUDA: malformed split arguments")
            owners = backend.split(args[0], args[1], args[2])
            return tuple(
                type(args[0])._from_owner(
                    owner,
                    owner.shape,
                    logical_strides=owner.strides,
                )
                for owner in owners
            )
        raise NotImplementedError(
            f"MatrixMan/CUDA: {_operator_name(func)} not implemented"
        )
    raise RuntimeError(f"MatrixMan/{backend.name}: no dispatch implementation")
