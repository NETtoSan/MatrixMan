"""MatrixMan CUDA backend façade."""

from dataclasses import dataclass

import torch

from ...backend import Backend
from . import factories
from .gpumatrix import CudaExecutionBackend, check as check_cuda, detect_device


@dataclass(frozen=True)
class CudaStorageLayout:
    """Logical storage metadata for one contiguous CUDA allocation."""

    numel: int
    kind: str = "cuda_linear"


class CudaTensorOwner:
    """Own one CUDA device allocation used by a MatrixMan tensor."""

    def __init__(
        self,
        execution: CudaExecutionBackend,
        pointer,
        shape,
        strides,
        storage_numel=None,
        dtype=torch.float32,
    ):
        self.execution = execution
        self.pointer = pointer
        self.dtype = dtype
        self.layout = CudaStorageLayout(
            numel=_numel(shape) if storage_numel is None else int(storage_numel)
        )
        self.shape = tuple(shape)
        self.strides = tuple(strides)

    @property
    def storage_description(self) -> str:
        return f"cuda_ptr={self.pointer.value}"

    def release(self) -> None:
        if self.pointer and self.pointer.value and not self.execution.closed:
            self.execution.free(self.pointer)

    def __del__(self):
        try:
            self.release()
        except Exception:
            # Context shutdown is still responsible for reclaiming allocations
            # that outlive their Python owner.
            pass


def _numel(shape) -> int:
    result = 1
    for dimension in shape:
        result *= int(dimension)
    return result


def _pair(value, label: str) -> tuple[int, int]:
    values = tuple(int(item) for item in value) if isinstance(value, (tuple, list)) else (int(value), int(value))
    if len(values) != 2:
        raise NotImplementedError(f"MatrixMan/CUDA: convolution {label} must contain two values")
    return values


def upload_tensor(data: torch.Tensor, execution: CudaExecutionBackend) -> CudaTensorOwner:
    """Upload a CPU float32 tensor using the existing CUDA transfer boundary."""
    if data.device.type != "cpu":
        raise RuntimeError("MatrixMan/CUDA: tensor upload requires a CPU tensor")
    if data.dtype != torch.float32:
        raise TypeError(f"MatrixMan/CUDA: unsupported tensor dtype {data.dtype}; only float32 is supported")
    if not data.is_contiguous():
        raise RuntimeError("MatrixMan/CUDA: tensor upload requires a contiguous tensor")
    array = data.detach().numpy()
    pointer = execution.to_device(array)
    return CudaTensorOwner(execution, pointer, tuple(data.shape), tuple(data.stride()))


class CudaBackend(Backend):
    """Backend interface adapter around the reusable CUDA execution runtime."""

    name = "cuda"

    def __init__(self):
        factories.install_privateuse1_factory_kernels()
        self.execution = CudaExecutionBackend()

    def device_info(self) -> dict[str, str]:
        return {
            "backend": "CUDA",
            "device": self.execution.info["name"],
            "compute_capability": self.execution.info["compute_capability"],
            "memory_mib": self.execution.info["memory_mib"],
        }

    def synchronize(self):
        check_cuda(
            self.execution.driver,
            self.execution.driver.cuCtxSynchronize(),
            "cuCtxSynchronize",
        )

    def convolution(self, input_tensor, weight, bias, stride, padding, dilation, groups):
        """Execute a float32 NCHW convolution entirely through CUDA storage."""
        if not hasattr(input_tensor, "_owner") or input_tensor._owner.layout.kind != "cuda_linear":
            raise RuntimeError("MatrixMan/CUDA: convolution requires a CUDA-backed MatrixManTensor")
        if input_tensor.dtype != torch.float32 or len(input_tensor.shape) != 4:
            raise NotImplementedError("MatrixMan/CUDA: only float32 NCHW convolution is implemented")
        if not isinstance(weight, torch.Tensor) or weight.device.type != "cpu":
            raise RuntimeError("MatrixMan/CUDA: convolution weights must be CPU tensors for upload")
        if weight.dtype != torch.float32 or not weight.is_contiguous() or len(weight.shape) != 4:
            raise NotImplementedError("MatrixMan/CUDA: convolution requires contiguous float32 weights shaped K,C,R,S")
        if bias is not None:
            if not isinstance(bias, torch.Tensor) or bias.device.type != "cpu":
                raise RuntimeError("MatrixMan/CUDA: convolution bias must be a CPU tensor for upload")
            if bias.dtype != torch.float32 or not bias.is_contiguous() or len(bias.shape) != 1:
                raise NotImplementedError("MatrixMan/CUDA: convolution bias must be contiguous float32")

        n, c, h, w = (int(value) for value in input_tensor.shape)
        k, weight_c, r, s = (int(value) for value in weight.shape)
        groups = int(groups)
        stride_h, stride_w = _pair(stride, "stride")
        pad_h, pad_w = _pair(padding, "padding")
        dilation_h, dilation_w = _pair(dilation, "dilation")
        if groups <= 0 or c % groups or k % groups or weight_c != c // groups:
            raise NotImplementedError(
                "MatrixMan/CUDA: convolution groups/weight channels are unsupported "
                "(expected weight C == input C / groups)"
            )
        if bias is not None and int(bias.shape[0]) != k:
            raise ValueError("MatrixMan/CUDA: convolution bias length must equal output channels")
        if min(stride_h, stride_w, dilation_h, dilation_w) <= 0 or min(pad_h, pad_w) < 0:
            raise NotImplementedError("MatrixMan/CUDA: invalid convolution stride, padding, or dilation")
        out_h = (h + 2 * pad_h - dilation_h * (r - 1) - 1) // stride_h + 1
        out_w = (w + 2 * pad_w - dilation_w * (s - 1) - 1) // stride_w + 1
        if out_h <= 0 or out_w <= 0:
            raise ValueError("MatrixMan/CUDA: convolution output dimensions must be positive")

        weight_pointer = self.execution.to_device(weight.detach().numpy())
        bias_pointer = None
        output_pointer = None
        try:
            bias_pointer = (
                self.execution.to_device(bias.detach().numpy())
                if bias is not None
                else type(input_tensor._owner.pointer)()
            )
            output_pointer = self.execution.allocate(n * k * out_h * out_w * 4)
            self.execution.convolution(
                input_tensor._owner.pointer, weight_pointer, bias_pointer, output_pointer,
                n, c, h, w, k, r, s, out_h, out_w,
                stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups,
            )
            strides = (k * out_h * out_w, out_h * out_w, out_w, 1)
            return CudaTensorOwner(
                self.execution,
                output_pointer,
                (n, k, out_h, out_w),
                strides,
            )
        except Exception:
            if output_pointer is not None:
                self.execution.free(output_pointer)
            raise
        finally:
            self.execution.free(weight_pointer)
            if bias_pointer is not None and bias_pointer.value:
                self.execution.free(bias_pointer)

    def batch_norm(self, input_tensor, weight, bias, running_mean, running_var, training, eps):
        """Execute inference-only float32 NCHW BatchNorm in CUDA storage."""
        if training:
            raise NotImplementedError("MatrixMan/CUDA: BatchNorm training mode is not implemented")
        if not hasattr(input_tensor, "_owner") or input_tensor._owner.layout.kind != "cuda_linear":
            raise RuntimeError("MatrixMan/CUDA: BatchNorm requires a CUDA-backed MatrixManTensor")
        if input_tensor.dtype != torch.float32 or len(input_tensor.shape) != 4:
            raise NotImplementedError("MatrixMan/CUDA: BatchNorm supports float32 NCHW tensors only")
        n, channels, height, width = (int(value) for value in input_tensor.shape)

        def parameter(value, name, default):
            if value is None:
                return torch.full((channels,), default, dtype=torch.float32)
            if (
                not isinstance(value, torch.Tensor)
                or value.device.type != "cpu"
                or value.dtype != torch.float32
                or not value.is_contiguous()
                or tuple(value.shape) != (channels,)
            ):
                raise NotImplementedError(
                    f"MatrixMan/CUDA: BatchNorm {name} must be contiguous CPU float32 shape [{channels}]"
                )
            return value.detach()

        running_mean = parameter(running_mean, "running_mean", 0.0)
        running_var = parameter(running_var, "running_var", 1.0)
        weight = parameter(weight, "weight", 1.0)
        bias = parameter(bias, "bias", 0.0)
        if float(eps) < 0:
            raise ValueError("MatrixMan/CUDA: BatchNorm epsilon must be non-negative")

        parameter_pointers = []
        try:
            for value in (running_mean, running_var, weight, bias):
                parameter_pointers.append(self.execution.to_device(value.numpy()))
        except Exception:
            for pointer in parameter_pointers:
                self.execution.free(pointer)
            raise
        output_pointer = None
        try:
            output_pointer = self.execution.allocate(n * channels * height * width * 4)
            self.execution.batch_norm(
                input_tensor._owner.pointer,
                parameter_pointers[0],
                parameter_pointers[1],
                parameter_pointers[2],
                parameter_pointers[3],
                output_pointer,
                n * channels * height * width,
                channels,
                height * width,
                float(eps),
            )
            strides = (channels * height * width, height * width, width, 1)
            return CudaTensorOwner(
                self.execution,
                output_pointer,
                (n, channels, height, width),
                strides,
            )
        except Exception:
            if output_pointer is not None:
                self.execution.free(output_pointer)
            raise
        finally:
            for pointer in parameter_pointers:
                self.execution.free(pointer)

    def silu(self, input_tensor, inplace=False):
        """Execute float32 SiLU, optionally mutating the existing CUDA storage."""
        if not hasattr(input_tensor, "_owner") or input_tensor._owner.layout.kind != "cuda_linear":
            raise RuntimeError("MatrixMan/CUDA: SiLU requires a CUDA-backed MatrixManTensor")
        if input_tensor.dtype != torch.float32:
            raise NotImplementedError("MatrixMan/CUDA: SiLU supports float32 tensors only")
        shape = tuple(int(value) for value in input_tensor.shape)
        expected_strides = []
        stride = 1
        for dimension in reversed(shape):
            expected_strides.insert(0, stride)
            stride *= dimension
        if tuple(input_tensor._owner.strides) != tuple(expected_strides):
            raise NotImplementedError("MatrixMan/CUDA: SiLU requires contiguous tensors")
        count = _numel(shape)
        output_pointer = input_tensor._owner.pointer if inplace else None
        if output_pointer is None:
            output_pointer = self.execution.allocate(count * 4)
        try:
            self.execution.silu(input_tensor._owner.pointer, output_pointer, count)
            if inplace:
                return None
            return CudaTensorOwner(
                self.execution,
                output_pointer,
                shape,
                tuple(expected_strides),
            )
        except Exception:
            if not inplace:
                self.execution.free(output_pointer)
            raise

    def split(self, input_tensor, split_size, dimension):
        """Copy contiguous float32 tensor chunks into independent CUDA buffers."""
        if not hasattr(input_tensor, "_owner") or input_tensor._owner.layout.kind != "cuda_linear":
            raise RuntimeError("MatrixMan/CUDA: split requires a CUDA-backed MatrixManTensor")
        if input_tensor.dtype != torch.float32:
            raise NotImplementedError("MatrixMan/CUDA: split supports float32 tensors only")
        shape = tuple(int(value) for value in input_tensor.shape)
        rank = len(shape)
        if rank < 1 or rank > 4:
            raise NotImplementedError("MatrixMan/CUDA: split supports tensor ranks 1 through 4")
        expected_strides = []
        stride = 1
        for size in reversed(shape):
            expected_strides.insert(0, stride)
            stride *= size
        if tuple(input_tensor._owner.strides) != tuple(expected_strides):
            raise NotImplementedError("MatrixMan/CUDA: split requires contiguous tensors")
        split_size = int(split_size)
        if split_size <= 0:
            raise ValueError("MatrixMan/CUDA: split size must be positive")
        dimension = int(dimension)
        if dimension < 0:
            dimension += rank
        if dimension < 0 or dimension >= rank:
            raise ValueError("MatrixMan/CUDA: split dimension is out of range")

        split_axis_size = shape[dimension]
        chunks = [
            min(split_size, split_axis_size - start)
            for start in range(0, split_axis_size, split_size)
        ]
        padded_input = (1,) * (4 - rank) + shape
        padded_dimension = dimension + 4 - rank
        owners = []
        offset = 0
        try:
            for chunk_size in chunks:
                output_shape = shape[:dimension] + (chunk_size,) + shape[dimension + 1:]
                padded_output = (1,) * (4 - rank) + output_shape
                output_pointer = self.execution.allocate(_numel(output_shape) * 4)
                try:
                    self.execution.split_copy(
                        input_tensor._owner.pointer,
                        output_pointer,
                        _numel(output_shape),
                        padded_dimension,
                        offset,
                        padded_input,
                        padded_output,
                    )
                except Exception:
                    self.execution.free(output_pointer)
                    raise
                owners.append(
                    CudaTensorOwner(
                        self.execution,
                        output_pointer,
                        output_shape,
                        tuple(_numel(output_shape[index + 1:]) for index in range(rank)),
                    )
                )
                offset += chunk_size
        except Exception:
            for owner in owners:
                owner.release()
            raise
        return owners

    def add(self, left, right, alpha=1):
        """Execute contiguous float32 tensor addition through matrix_add."""
        for name, tensor in (("left", left), ("right", right)):
            if not hasattr(tensor, "_owner") or tensor._owner.layout.kind != "cuda_linear":
                raise RuntimeError(f"MatrixMan/CUDA: add {name} must be CUDA-backed")
            if tensor.dtype != torch.float32:
                raise NotImplementedError("MatrixMan/CUDA: add supports float32 tensors only")
            if not isinstance(tensor.shape, torch.Size):
                raise RuntimeError("MatrixMan/CUDA: add received invalid tensor metadata")
        if tuple(left.shape) != tuple(right.shape):
            raise NotImplementedError("MatrixMan/CUDA: add broadcasting is not implemented; shapes must match")
        shape = tuple(int(value) for value in left.shape)
        left_strides = tuple(int(value) for value in left._owner.strides)
        right_strides = tuple(int(value) for value in right._owner.strides)
        expected_strides = []
        stride = 1
        for size in reversed(shape):
            expected_strides.insert(0, stride)
            stride *= size
        if left_strides != tuple(expected_strides) or right_strides != tuple(expected_strides):
            raise NotImplementedError("MatrixMan/CUDA: add requires contiguous tensors")
        try:
            alpha = float(alpha)
        except (TypeError, ValueError) as exc:
            raise NotImplementedError("MatrixMan/CUDA: add alpha must be a scalar") from exc
        if alpha != 1.0:
            raise NotImplementedError("MatrixMan/CUDA: add currently supports alpha=1 only")

        output_pointer = self.execution.allocate(_numel(shape) * 4)
        try:
            self.execution.add(
                left._owner.pointer,
                right._owner.pointer,
                output_pointer,
                _numel(shape),
            )
            return CudaTensorOwner(
                self.execution,
                output_pointer,
                shape,
                tuple(expected_strides),
            )
        except Exception:
            self.execution.free(output_pointer)
            raise

    def cat(self, tensors, dimension):
        """Concatenate contiguous float32 tensors into one CUDA allocation."""
        tensors = tuple(tensors)
        if not tensors:
            raise ValueError("MatrixMan/CUDA: cat requires at least one tensor")
        first_shape = tuple(int(value) for value in tensors[0].shape)
        rank = len(first_shape)
        if rank < 1 or rank > 4:
            raise NotImplementedError("MatrixMan/CUDA: cat supports tensor ranks 1 through 4")
        dimension = int(dimension)
        if dimension < 0:
            dimension += rank
        if dimension < 0 or dimension >= rank:
            raise ValueError("MatrixMan/CUDA: cat dimension is out of range")
        expected_strides = []
        stride = 1
        for size in reversed(first_shape):
            expected_strides.insert(0, stride)
            stride *= size

        axis_total = 0
        for index, tensor in enumerate(tensors):
            if (
                not hasattr(tensor, "_owner")
                or tensor._owner.layout.kind != "cuda_linear"
                or tensor.dtype != torch.float32
            ):
                raise RuntimeError(
                    f"MatrixMan/CUDA: cat input {index} must be a CUDA-backed float32 tensor"
                )
            shape = tuple(int(value) for value in tensor.shape)
            if len(shape) != rank:
                raise NotImplementedError("MatrixMan/CUDA: cat inputs must have matching ranks")
            if tuple(tensor._owner.strides) != tuple(
                _numel(shape[index + 1:]) for index in range(rank)
            ):
                raise NotImplementedError("MatrixMan/CUDA: cat requires contiguous tensors")
            if any(shape[axis] != first_shape[axis] for axis in range(rank) if axis != dimension):
                raise ValueError("MatrixMan/CUDA: cat non-concatenated dimensions must match")
            if tensor._owner.execution is not self.execution:
                raise RuntimeError("MatrixMan/CUDA: cat inputs must share one CUDA execution context")
            axis_total += shape[dimension]

        output_shape = first_shape[:dimension] + (axis_total,) + first_shape[dimension + 1:]
        output_strides = tuple(
            _numel(output_shape[index + 1:]) for index in range(rank)
        )
        padded_output = (1,) * (4 - rank) + output_shape
        output_pointer = self.execution.allocate(_numel(output_shape) * 4)
        offset = 0
        try:
            for tensor in tensors:
                shape = tuple(int(value) for value in tensor.shape)
                padded_input = (1,) * (4 - rank) + shape
                self.execution.cat_copy(
                    tensor._owner.pointer,
                    output_pointer,
                    _numel(output_shape),
                    dimension + 4 - rank,
                    offset,
                    padded_input,
                    padded_output,
                )
                offset += shape[dimension]
            return CudaTensorOwner(
                self.execution,
                output_pointer,
                output_shape,
                output_strides,
            )
        except Exception:
            self.execution.free(output_pointer)
            raise

    def upsample_nearest2d(self, input_tensor, output_size, scale_h=None, scale_w=None):
        """Resize contiguous float32 NCHW storage with CUDA nearest-neighbor sampling."""
        if not hasattr(input_tensor, "_owner") or input_tensor._owner.layout.kind != "cuda_linear":
            raise RuntimeError("MatrixMan/CUDA: upsample requires a CUDA-backed MatrixManTensor")
        if input_tensor.dtype != torch.float32 or len(input_tensor.shape) != 4:
            raise NotImplementedError("MatrixMan/CUDA: upsample_nearest2d supports float32 NCHW tensors only")
        n, channels, input_height, input_width = (int(value) for value in input_tensor.shape)
        input_strides = tuple(int(value) for value in input_tensor._owner.strides)
        expected_input_strides = (
            channels * input_height * input_width,
            input_height * input_width,
            input_width,
            1,
        )
        if input_strides != expected_input_strides:
            raise NotImplementedError("MatrixMan/CUDA: upsample requires contiguous tensors")

        if output_size is not None:
            output_size = tuple(int(value) for value in output_size)
            if len(output_size) != 2:
                raise ValueError("MatrixMan/CUDA: upsample output_size must contain Hout and Wout")
            output_height, output_width = output_size
        else:
            if scale_h is None or scale_w is None:
                raise ValueError("MatrixMan/CUDA: upsample requires output_size or scale factors")
            scale_h, scale_w = float(scale_h), float(scale_w)
            if scale_h <= 0 or scale_w <= 0:
                raise ValueError("MatrixMan/CUDA: upsample scale factors must be positive")
            output_height = int(input_height * scale_h)
            output_width = int(input_width * scale_w)
        if output_height <= 0 or output_width <= 0:
            raise ValueError("MatrixMan/CUDA: upsample output dimensions must be positive")

        output_shape = (n, channels, output_height, output_width)
        output_pointer = self.execution.allocate(_numel(output_shape) * 4)
        try:
            self.execution.upsample_nearest2d(
                input_tensor._owner.pointer,
                output_pointer,
                _numel(output_shape),
                channels,
                input_height,
                input_width,
                output_height,
                output_width,
            )
            output_strides = (
                channels * output_height * output_width,
                output_height * output_width,
                output_width,
                1,
            )
            return CudaTensorOwner(
                self.execution,
                output_pointer,
                output_shape,
                output_strides,
            )
        except Exception:
            self.execution.free(output_pointer)
            raise

    def close(self) -> None:
        self.execution.close()

    @classmethod
    def probe(cls) -> bool:
        try:
            detect_device()
        except Exception:
            return False
        return True
