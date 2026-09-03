"""Backend-neutral PrivateUse1 dispatch routing."""

from __future__ import annotations

import torch

from .backend import get_backend
from .tensor import (
    contiguous_strides,
    expand_shape_strides,
    infer_view_shape,
    readback_tensor,
    unsqueeze_shape_strides,
)


def _operator_name(func) -> str:
    names = {
        torch.ops.aten.convolution.default: "Conv2D",
        torch.ops.aten.native_batch_norm.default: "BatchNorm",
        torch.ops.aten.silu_.default: "SiLU",
        torch.ops.aten.add.Tensor: "Add",
        torch.ops.aten.mm.default: "MatMul",
        torch.ops.aten.sub.Tensor: "Sub",
        torch.ops.aten.div.Tensor: "Div",
        torch.ops.aten.div.Scalar: "Div",
        torch.ops.aten.cat.default: "Cat",
        torch.ops.aten.stack.default: "Stack",
        torch.ops.aten._softmax.default: "Softmax",
        torch.ops.aten.upsample_nearest2d.default: "UpsampleNearest2D",
    }
    return names.get(func, str(func))


def handle_torch_dispatch(cls, func, types, args=(), kwargs=None):
    """Send a PrivateUse1 operation to the selected backend only."""
    kwargs = kwargs or {}
    backend = get_backend()
    if func is torch.ops.aten._to_copy.default:
        if not args or not isinstance(args[0], cls):
            raise RuntimeError("MatrixMan: _to_copy requires a MatrixManTensor source")
        source = args[0]
        target_device = kwargs.get("device")
        if target_device is not None and torch.device(target_device).type != "cpu":
            raise NotImplementedError("MatrixMan only supports explicit MatrixMan-to-CPU transfer")
        target_dtype = kwargs.get("dtype")
        if target_dtype is not None and target_dtype != source.dtype:
            raise NotImplementedError("MatrixMan CPU readback does not support dtype conversion")
        target_layout = kwargs.get("layout")
        if target_layout not in {None, torch.strided}:
            raise NotImplementedError("MatrixMan CPU readback supports strided layout only")
        if kwargs.get("pin_memory", False):
            raise NotImplementedError("MatrixMan CPU readback does not support pin_memory")
        memory_format = kwargs.get("memory_format")
        if memory_format not in {None, torch.preserve_format}:
            raise NotImplementedError("MatrixMan CPU readback supports preserve_format only")
        return readback_tensor(
            source,
            audit_op="aten._to_copy.default",
            audit_reason="explicit MatrixMan-to-CPU transfer",
        )
    if backend.name == "opengl":
        from .backends.opengl import dispatch as opengl_dispatch

        return opengl_dispatch.handle_torch_dispatch(cls, func, types, args, kwargs)
    if backend.name == "cuda":
        if func is torch.ops.aten.unsqueeze.default:
            if len(args) < 2:
                raise RuntimeError("MatrixMan/CUDA: malformed unsqueeze arguments")
            input_tensor = args[0]
            if not hasattr(input_tensor, "_owner") or input_tensor._owner.layout.kind != "cuda_linear":
                raise RuntimeError("MatrixMan/CUDA: unsqueeze requires a CUDA-backed MatrixManTensor")
            output_shape, output_strides = unsqueeze_shape_strides(
                input_tensor.shape,
                input_tensor._logical_strides,
                args[1],
            )
            return type(input_tensor)._from_owner(
                input_tensor._owner,
                output_shape,
                storage_offset=input_tensor._storage_offset,
                logical_strides=output_strides,
            )
        if func is torch.ops.aten._softmax.default:
            if len(args) < 2:
                raise RuntimeError("MatrixMan/CUDA: malformed softmax arguments")
            input_tensor = args[0]
            dimension = args[1]
            half_to_float = args[2] if len(args) > 2 else (kwargs or {}).get("half_to_float", False)
            owner = backend.softmax(input_tensor, dimension, half_to_float)
            return type(input_tensor)._from_owner(
                owner,
                owner.shape,
                logical_strides=owner.strides,
            )
        if func is torch.ops.aten.transpose.int:
            if len(args) < 3:
                raise RuntimeError("MatrixMan/CUDA: malformed transpose arguments")
            input_tensor = args[0]
            if not hasattr(input_tensor, "_owner") or input_tensor._owner.layout.kind != "cuda_linear":
                raise RuntimeError("MatrixMan/CUDA: transpose requires a CUDA-backed MatrixManTensor")
            shape = list(int(value) for value in input_tensor.shape)
            strides = list(int(value) for value in input_tensor._logical_strides)
            rank = len(shape)
            dim0 = int(args[1])
            dim1 = int(args[2])
            if dim0 < 0:
                dim0 += rank
            if dim1 < 0:
                dim1 += rank
            if dim0 < 0 or dim0 >= rank or dim1 < 0 or dim1 >= rank:
                raise IndexError("MatrixMan/CUDA: transpose dimension is out of range")
            shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
            strides[dim0], strides[dim1] = strides[dim1], strides[dim0]
            return type(input_tensor)._from_owner(
                input_tensor._owner,
                tuple(shape),
                storage_offset=input_tensor._storage_offset,
                logical_strides=tuple(strides),
            )
        if func is torch.ops.aten.expand.default:
            if len(args) < 2:
                raise RuntimeError("MatrixMan/CUDA: malformed expand arguments")
            input_tensor = args[0]
            if not hasattr(input_tensor, "_owner") or input_tensor._owner.layout.kind != "cuda_linear":
                raise RuntimeError("MatrixMan/CUDA: expand requires a CUDA-backed MatrixManTensor")
            output_shape, output_strides = expand_shape_strides(
                input_tensor.shape,
                input_tensor._logical_strides,
                args[1],
            )
            return type(input_tensor)._from_owner(
                input_tensor._owner,
                output_shape,
                storage_offset=input_tensor._storage_offset,
                logical_strides=output_strides,
            )
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
        if func is torch.ops.aten.sub.Tensor:
            if len(args) < 2:
                raise RuntimeError("MatrixMan/CUDA: malformed sub arguments")
            alpha = args[2] if len(args) > 2 else (kwargs or {}).get("alpha", 1)
            owner = backend.sub(args[0], args[1], alpha)
            return type(args[0])._from_owner(
                owner,
                owner.shape,
                logical_strides=owner.strides,
            )
        if func is torch.ops.aten.mul.Tensor:
            if len(args) < 2:
                raise RuntimeError("MatrixMan/CUDA: malformed mul arguments")
            owner = backend.mul(args[0], args[1])
            return type(args[0])._from_owner(
                owner,
                owner.shape,
                logical_strides=owner.strides,
            )
        if func is torch.ops.aten.mm.default:
            if len(args) < 2:
                raise RuntimeError("MatrixMan/CUDA: malformed mm arguments")
            owner = backend.matmul(args[0], args[1])
            return type(args[0])._from_owner(
                owner,
                owner.shape,
                logical_strides=owner.strides,
            )
        if func is torch.ops.aten.sigmoid.default:
            if len(args) < 1:
                raise RuntimeError("MatrixMan/CUDA: malformed sigmoid arguments")
            owner = backend.sigmoid(args[0])
            return type(args[0])._from_owner(
                owner,
                owner.shape,
                logical_strides=owner.strides,
            )
        if func is torch.ops.aten.div.Tensor or func is torch.ops.aten.div.Scalar:
            if len(args) < 2:
                raise RuntimeError("MatrixMan/CUDA: malformed div arguments")
            input_tensor, divisor = args[0], args[1]
            if hasattr(divisor, "_owner"):
                raise NotImplementedError(
                    "MatrixMan/CUDA: tensor-tensor div is not implemented"
                )
            owner = backend.div(input_tensor, divisor)
            return type(input_tensor)._from_owner(
                owner,
                owner.shape,
                logical_strides=owner.strides,
            )
        if func is torch.ops.aten.add.Tensor:
            if len(args) < 2:
                raise RuntimeError("MatrixMan/CUDA: malformed add arguments")
            alpha = args[2] if len(args) > 2 else (kwargs or {}).get("alpha", 1)
            owner = backend.add(args[0], args[1], alpha)
            tensor_operand = args[0] if hasattr(args[0], "_owner") else args[1]
            return type(tensor_operand)._from_owner(
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
        if func is torch.ops.aten.stack.default:
            if not args:
                raise RuntimeError("MatrixMan/CUDA: malformed stack arguments")
            tensors = args[0]
            if not isinstance(tensors, (tuple, list)):
                raise RuntimeError("MatrixMan/CUDA: stack expects a Tensor[] sequence")
            dimension = args[1] if len(args) > 1 else (kwargs or {}).get("dim", 0)
            owner = backend.stack(tensors, dimension)
            return type(tensors[0])._from_owner(
                owner,
                owner.shape,
                logical_strides=owner.strides,
            )
        if func is torch.ops.aten.fill_.Scalar:
            if len(args) < 2:
                raise RuntimeError("MatrixMan/CUDA: malformed fill_ arguments")
            return backend.fill(args[0], args[1])
        if func is torch.ops.aten.new_full.default:
            if len(args) < 3:
                raise RuntimeError("MatrixMan/CUDA: malformed new_full arguments")
            from .backends.cuda.factories import new_full_cuda

            input_tensor, size, fill_value = args[:3]
            options = kwargs or {}
            return new_full_cuda(
                input_tensor,
                size,
                fill_value,
                dtype=options.get("dtype"),
                layout=options.get("layout"),
                device=options.get("device"),
                pin_memory=options.get("pin_memory"),
            )
        if func is torch.ops.aten.arange.out:
            if len(args) < 1 or "out" not in (kwargs or {}):
                raise RuntimeError("MatrixMan/CUDA: malformed arange.out arguments")
            from .backends.cuda.factories import arange_out_cuda

            return arange_out_cuda(args[0], out=(kwargs or {})["out"])
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
            if len(args) < 2:
                raise RuntimeError("MatrixMan/CUDA: malformed split arguments")
            # torch.chunk is lowered to split.Tensor on this PyTorch build.
            # The lowered call supplies only (self, split_size) when the
            # default dimension is used; explicit Python keyword arguments
            # may likewise remain in kwargs.
            dimension = args[2] if len(args) > 2 else (kwargs or {}).get("dim", 0)
            owners = backend.split(args[0], args[1], dimension)
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
