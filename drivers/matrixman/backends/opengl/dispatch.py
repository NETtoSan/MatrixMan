"""PyTorch/ATen dispatch bridge for the OpenGL MatrixMan backend."""

from __future__ import annotations

import math

import torch

from . import convolution, diagnostics, metadata, profiling
from . import tensor as tensor_module
from . import operation_context
from .ops import activation, arithmetic, concat, matmul, normalization, pooling, resize, softmax
from .storage import numel
from ...tensor import MatrixManTensor

_render_packed_sub = arithmetic._render_packed_sub
_render_packed_broadcast_mul = arithmetic._render_packed_broadcast_mul
_render_packed_scalar_div = arithmetic._render_packed_scalar_div
_render_packed_sigmoid = activation._render_packed_sigmoid
_render_batch_norm = normalization._render_batch_norm
_render_silu_inplace = activation._render_silu_inplace
_metadata_split = metadata.metadata_split
_render_cat = concat._render_cat
_render_stack = concat._render_stack
_render_fill_scalar = concat._render_fill_scalar
_render_max_pool2d_with_indices = pooling._render_max_pool2d_with_indices
_render_upsample_nearest2d = resize.render_upsample_nearest2d
_render_softmax = softmax._render_softmax
_read_texture = tensor_module.readback_tensor
_metadata_view = metadata.metadata_view
_normalize_shape = metadata.normalize_shape
_metadata_squeeze = metadata.metadata_squeeze
_metadata_unsqueeze = metadata.metadata_unsqueeze
_metadata_expand = metadata.metadata_expand
_metadata_transpose = metadata.metadata_transpose
_validate_supported_shape = metadata.validate_supported_shape
_is_scalar_operand = operation_context.is_scalar_operand
_scalar_value = operation_context.scalar_value
_numel = numel

def _trace(message: str) -> None:
    diagnostics.trace(message)


def _kernel_log(message: str) -> None:
    diagnostics.kernel_log(message)


def _error_log(message: str) -> None:
    diagnostics.error_log(message)


def _record_unsupported(func, args, kwargs) -> None:
    diagnostics.record_unsupported(func, args, kwargs)


def _render_binary(kind: str, left: MatrixManTensor, right: MatrixManTensor, alpha: float = 1.0) -> MatrixManTensor:
    if kind == "add":
        if isinstance(left, MatrixManTensor) and operation_context.is_scalar_operand(right):
            return arithmetic._render_scalar_add(left, operation_context.scalar_value(right), alpha, tensor_first=True)
        if operation_context.is_scalar_operand(left) and isinstance(right, MatrixManTensor):
            return arithmetic._render_scalar_add(right, operation_context.scalar_value(left), alpha, tensor_first=False)
    if not isinstance(left, MatrixManTensor) or not isinstance(right, MatrixManTensor):
        raise RuntimeError(f"gm45 {kind} requires both inputs to be gm45 tensors")
    if left.shape != right.shape:
        raise RuntimeError(f"gm45 {kind} requires equal shapes")
    if kind == "add" and left._owner.layout.kind == "packed_rgba" and right._owner.layout.kind == "packed_rgba":
        if len(left.shape) == 3 and int(left.shape[0]) == 1 and (
            not metadata.is_contiguous_logical(left) or not metadata.is_contiguous_logical(right)
        ):
            return arithmetic._render_packed_strided_add(left, right, alpha)
        return arithmetic._render_packed_add(left, right, alpha)
    return matmul.render_matrix_binary(kind, left, right, alpha)


@profiling.dispatch_timer
def handle_torch_dispatch(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        _trace(f"torch_dispatch -> {func}")

        if func is torch.ops.aten.add.Tensor:
            _kernel_log("Add")
            _trace("  -> MatrixManTensor.__torch_dispatch__")
            _trace("  -> MatrixMan/OpenGL add kernel")
            _trace("  -> GLSL fragment shader arithmetic: out = left + right")
            alpha = kwargs.get("alpha", args[2] if len(args) > 2 else 1)
            if isinstance(alpha, torch.Tensor):
                raise RuntimeError("gm45 add alpha must be a Python scalar")
            return _render_binary("add", args[0], args[1], float(alpha))

        if func is torch.ops.aten.sub.Tensor:
            _kernel_log("Sub")
            _trace("  -> MatrixManTensor.__torch_dispatch__")
            _trace("  -> MatrixMan/OpenGL packed sub kernel")
            _trace("  -> GLSL fragment shader arithmetic: out = left - right")
            if len(args) < 2 or not isinstance(args[0], MatrixManTensor) or not isinstance(args[1], MatrixManTensor):
                raise RuntimeError("gm45 sub currently requires two MatrixManTensor operands")
            return _render_packed_sub(args[0], args[1])

        if func is torch.ops.aten.mul.Tensor:
            _kernel_log("Mul")
            _trace("  -> MatrixManTensor.__torch_dispatch__")
            _trace("  -> MatrixMan/OpenGL packed broadcast mul kernel")
            _trace("  -> GLSL fragment shader arithmetic: out = lhs * rhs")
            if len(args) < 2 or not isinstance(args[0], MatrixManTensor) or not isinstance(args[1], MatrixManTensor):
                raise RuntimeError("gm45 mul currently requires two MatrixManTensor operands")
            return _render_packed_broadcast_mul(args[0], args[1])

        if func is torch.ops.aten.sigmoid.default:
            _kernel_log("Sigmoid")
            _trace("  -> MatrixManTensor.__torch_dispatch__")
            _trace("  -> MatrixMan/OpenGL packed sigmoid kernel")
            _trace("  -> GLSL fragment shader arithmetic: out = 1 / (1 + exp(-x))")
            if len(args) < 1 or not isinstance(args[0], MatrixManTensor):
                raise RuntimeError("gm45 sigmoid requires a MatrixManTensor input")
            return _render_packed_sigmoid(args[0])

        if func is torch.ops.aten.div.Tensor:
            _kernel_log("Div")
            _trace("  -> MatrixManTensor.__torch_dispatch__")
            _trace("  -> MatrixMan/OpenGL packed scalar div kernel")
            _trace("  -> GLSL fragment shader arithmetic: out = input / scalar")
            if len(args) < 2 or not isinstance(args[0], MatrixManTensor) or not _is_scalar_operand(args[1]):
                raise RuntimeError("gm45 div currently requires a MatrixManTensor divided by a scalar")
            rounding_mode = kwargs.get("rounding_mode")
            if rounding_mode is not None:
                raise RuntimeError("gm45 div currently supports rounding_mode=None only")
            return _render_packed_scalar_div(args[0], _scalar_value(args[1]))

        if func is torch.ops.aten.mm.default:
            _kernel_log("MatMul")
            _trace("  -> MatrixManTensor.__torch_dispatch__")
            _trace("  -> MatrixMan/OpenGL matmul kernel")
            _trace("  -> GLSL fragment shader arithmetic: sum(left[row,k] * right[k,col])")
            return matmul.render_matmul(args[0], args[1])

        if func is torch.ops.aten.convolution.default:
            _trace("  -> MatrixManTensor.__torch_dispatch__")
            _trace("  -> MatrixMan/OpenGL Conv2D kernel")
            _trace("  -> GLSL fragment shader arithmetic: bias + sum(input * weight)")
            return convolution.execute(args)

        if func is torch.ops.aten.native_batch_norm.default:
            _kernel_log("BatchNorm")
            _trace("  -> MatrixManTensor.__torch_dispatch__")
            _trace("  -> MatrixMan/OpenGL BatchNorm inference kernel")
            _trace("  -> GLSL fragment shader arithmetic: ((x - mean) / sqrt(var + eps)) * weight + bias")
            return _render_batch_norm(args)

        if func is torch.ops.aten.silu_.default:
            _kernel_log("SiLU")
            _trace("  -> MatrixManTensor.__torch_dispatch__")
            _trace("  -> MatrixMan/OpenGL SiLU kernel")
            _trace("  -> GLSL fragment shader arithmetic: x / (1 + exp(-x))")
            return _render_silu_inplace(args)

        if func is torch.ops.aten.split.Tensor:
            _trace("  -> MatrixManTensor.__torch_dispatch__")
            _trace("  -> MatrixMan/OpenGL metadata-only split view")
            return _metadata_split(args, kwargs)

        if func is torch.ops.aten.cat.default:
            _kernel_log("Cat")
            _trace("  -> MatrixManTensor.__torch_dispatch__")
            _trace("  -> MatrixMan/OpenGL cat copy/repack kernel")
            _trace("  -> GLSL fragment shader copy: packed tensor concatenation")
            return _render_cat(args, kwargs)

        if func is torch.ops.aten.stack.default:
            _kernel_log("Stack")
            _trace("  -> MatrixManTensor.__torch_dispatch__")
            _trace("  -> MatrixMan/OpenGL stack materialization kernel")
            _trace("  -> GLSL fragment shader copy: logical-stride tensor stack")
            return _render_stack(args, kwargs)

        if func is torch.ops.aten.fill_.Scalar:
            _kernel_log("Fill")
            _trace("  -> MatrixManTensor.__torch_dispatch__")
            _trace("  -> MatrixMan/OpenGL fill_ scalar kernel")
            _trace("  -> GLSL fragment shader write: out = scalar")
            return _render_fill_scalar(args)

        if func is torch.ops.aten.new_full.default:
            _kernel_log("NewFull")
            _trace("  -> MatrixMan/OpenGL new_full allocation + scalar fill")
            if "memory_format" in kwargs and kwargs["memory_format"] is not None:
                raise RuntimeError("gm45 new_full does not support memory_format")
            from .factories import new_full_gm45

            return new_full_gm45(
                args[0],
                args[1],
                args[2],
                dtype=kwargs.get("dtype"),
                layout=kwargs.get("layout"),
                device=kwargs.get("device"),
                pin_memory=kwargs.get("pin_memory"),
            )

        if func is torch.ops.aten.arange.out:
            _kernel_log("ArangeOut")
            _trace("  -> MatrixMan/OpenGL arange shader into provided out tensor")
            if len(args) != 1 or "out" not in kwargs:
                raise RuntimeError("gm45 arange.out requires end and out arguments")
            from .factories import arange_out

            return arange_out(args[0], out=kwargs["out"])

        if func is torch.ops.aten.max_pool2d_with_indices.default:
            _kernel_log("MaxPool")
            _trace("  -> MatrixManTensor.__torch_dispatch__")
            _trace("  -> MatrixMan/OpenGL max_pool2d values kernel")
            _trace("  -> GLSL fragment shader arithmetic: max over valid 5x5 window")
            return _render_max_pool2d_with_indices(args)

        if func is torch.ops.aten.upsample_nearest2d.default:
            _kernel_log("Upsample")
            _trace("  -> MatrixManTensor.__torch_dispatch__")
            _trace("  -> MatrixMan/OpenGL upsample_nearest2d kernel")
            _trace("  -> GLSL fragment shader sampling: output[y,x] -> input[y/2,x/2]")
            return _render_upsample_nearest2d(args)

        if func is torch.ops.aten._softmax.default:
            _kernel_log("Softmax")
            _trace("  -> MatrixManTensor.__torch_dispatch__")
            _trace("  -> MatrixMan/OpenGL DFL softmax kernel")
            _trace("  -> GLSL fragment shader arithmetic: stable 16-bin max/exp/sum/normalize")
            return _render_softmax(args)

        if func is torch.ops.aten._to_copy.default:
            _trace("  -> MatrixManTensor.__torch_dispatch__")
            _trace("  -> explicit GPU-to-CPU transfer requested")
            target_device = kwargs.get("device")
            target_dtype = kwargs.get("dtype")
            if target_device is not None and torch.device(target_device).type != "cpu":
                raise RuntimeError("gm45 only supports transfer from gm45 to CPU")
            if target_dtype is not None and target_dtype != torch.float32:
                raise RuntimeError("gm45 only supports float32 CPU readback")
            return _read_texture(args[0]._owner, args[0]._shape, args[0]._storage_offset, args[0]._logical_strides)

        if func is torch.ops.aten.detach.default:
            return args[0]

        if func is torch.ops.aten.view.default:
            tensor_arg = args[0]
            return _metadata_view(tensor_arg, _normalize_shape(args[1], _numel(tuple(tensor_arg.shape))), "view")

        if func is torch.ops.aten.reshape.default:
            tensor_arg = args[0]
            return _metadata_view(tensor_arg, _normalize_shape(args[1], _numel(tuple(tensor_arg.shape))), "reshape")

        if func is torch.ops.aten.flatten.using_ints:
            tensor_arg = args[0]
            shape = tuple(int(v) for v in tensor_arg.shape)
            start_dim = int(args[1]) if len(args) > 1 else int(kwargs.get("start_dim", 0))
            end_dim = int(args[2]) if len(args) > 2 else int(kwargs.get("end_dim", -1))
            if start_dim < 0:
                start_dim += len(shape)
            if end_dim < 0:
                end_dim += len(shape)
            if start_dim < 0 or end_dim >= len(shape) or start_dim > end_dim:
                raise RuntimeError("gm45 flatten dim range is invalid")
            flat_dim = math.prod(shape[start_dim : end_dim + 1])
            new_shape = shape[:start_dim] + (flat_dim,) + shape[end_dim + 1 :]
            _validate_supported_shape(new_shape)
            return _metadata_view(tensor_arg, new_shape, "flatten")

        if func is torch.ops.aten.squeeze.default:
            tensor_arg = args[0]
            return _metadata_squeeze(tensor_arg)

        if func is torch.ops.aten.squeeze.dim:
            tensor_arg = args[0]
            return _metadata_squeeze(tensor_arg, args[1])

        if func is torch.ops.aten.unsqueeze.default:
            tensor_arg = args[0]
            return _metadata_unsqueeze(tensor_arg, int(args[1]))

        if func is torch.ops.aten.expand.default:
            return _metadata_expand(args, kwargs)

        if func is torch.ops.aten.transpose.int:
            return _metadata_transpose(args)

        if func is torch.ops.aten.permute.default:
            _record_unsupported(func, args, kwargs)
            _error_log(f"Unsupported op: {func}")
            raise RuntimeError(
                f"gm45 unsupported metadata operation: {func}. "
                "The current packed contiguous texture layout cannot represent permute/transpose "
                "without either strided texture interpretation support or a GPU copy/render pass."
            )

        _record_unsupported(func, args, kwargs)
        _error_log(f"Unsupported op: {func}")
        raise RuntimeError(
            f"gm45 unsupported operation: {func}. "
            "This prototype supports CPU upload, .cpu()/.to('cpu'), metadata-only "
            "view/reshape/flatten/squeeze/unsqueeze/split, 2D add/matmul, "
            "packed elementwise add, YOLO-subset stack, fill_ scalar, packed NCHW channel cat, packed 3D last-dim cat, "
            "Conv2D, eval BatchNorm, SiLU_, YOLO-subset max_pool2d values, and "
            "YOLO-subset nearest upsample, float32 arange, and YOLO-subset softmax."
        )



class DispatchBridge(torch.Tensor):
    """Compatibility shell for callers that still reference the old bridge."""

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        return handle_torch_dispatch(cls, func, types, args, kwargs)
