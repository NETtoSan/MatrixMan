"""MatrixMan CUDA backend façade."""

from dataclasses import dataclass
import os

import torch

from ...backend import Backend
from ...tensor import contiguous_strides
from . import factories
from . import profiling
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
    profiling.count_activation("uploads", array.nbytes)
    return CudaTensorOwner(execution, pointer, tuple(data.shape), tuple(data.stride()))


class CudaBackend(Backend):
    """Backend interface adapter around the reusable CUDA execution runtime."""

    name = "cuda"

    def __init__(self):
        factories.install_privateuse1_factory_kernels()
        self.execution = CudaExecutionBackend()
        # Raw parameter pointers are owned by this backend for its lifetime;
        # activation/output owners remain independently managed by tensors.
        self._parameter_cache = {}

    def device_info(self) -> dict[str, str]:
        return {
            "backend": "CUDA",
            "device": self.execution.info["name"],
            "compute_capability": self.execution.info["compute_capability"],
            "memory_mib": self.execution.info["memory_mib"],
        }

    def synchronize(self):
        self.execution.synchronize()

    def matmul(self, left, right):
        """Execute 2D matrix multiplication through the existing CUDA kernel."""
        for name, value in (("left", left), ("right", right)):
            if not hasattr(value, "_owner") or value._owner.layout.kind != "cuda_linear":
                raise RuntimeError(f"MatrixMan/CUDA: mm requires a CUDA-backed {name} tensor")
            if value.dtype != torch.float32 or len(value.shape) != 2:
                raise NotImplementedError("MatrixMan/CUDA: mm supports 2D float32 tensors only")
            if tuple(int(item) for item in value._logical_strides) != contiguous_strides(tuple(value.shape)):
                raise NotImplementedError("MatrixMan/CUDA: mm requires contiguous logical inputs")
            if int(value._storage_offset) != 0:
                raise NotImplementedError("MatrixMan/CUDA: mm does not support nonzero storage offsets")
        m, k = (int(item) for item in left.shape)
        right_k, n = (int(item) for item in right.shape)
        if k != right_k:
            raise RuntimeError(
                f"MatrixMan/CUDA: mm shapes cannot be multiplied ({m}x{k} and {right_k}x{n})"
            )
        output_pointer = self.execution.allocate(m * n * 4)
        try:
            self.execution.matmul(
                left._owner.pointer,
                right._owner.pointer,
                output_pointer,
                m,
                k,
                n,
            )
            return CudaTensorOwner(
                self.execution,
                output_pointer,
                (m, n),
                contiguous_strides((m, n)),
            )
        except Exception:
            self.execution.free(output_pointer)
            raise

    @staticmethod
    def _parameter_cache_key(value: torch.Tensor):
        try:
            storage_pointer = int(value.untyped_storage().data_ptr())
        except AttributeError:
            storage_pointer = int(value.storage().data_ptr())
        return (
            storage_pointer,
            int(value.storage_offset()),
            tuple(int(item) for item in value.shape),
            tuple(int(item) for item in value.stride()),
            str(value.dtype),
            value.device.type,
        )

    def _cached_parameter(self, value: torch.Tensor):
        """Return a persistent device pointer and whether it was uploaded."""
        key = self._parameter_cache_key(value)
        version = int(value._version)
        entry = self._parameter_cache.get(key)
        if entry is not None and entry[1] == version:
            profiling.parameter_cache_event("hits")
            return entry[2], False

        if entry is not None:
            self.execution.synchronize()
            self.execution.free(entry[2])
            profiling.parameter_cache_adjust("retained_allocations", -1)
            profiling.parameter_cache_adjust("retained_bytes", -entry[3])

        pointer = self.execution.to_device(value.detach().numpy(), category="parameter")
        nbytes = int(value.numel() * value.element_size())
        # Keep the source tensor alive while its storage identity is cached.
        # This prevents a later CPU allocation from reusing the same data_ptr
        # and accidentally matching a stale cache entry.
        self._parameter_cache[key] = (value, version, pointer, nbytes)
        profiling.parameter_cache_event("misses")
        profiling.parameter_cache_adjust("retained_allocations", 1)
        profiling.parameter_cache_adjust("retained_bytes", nbytes)
        return pointer, True

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
        disable_specialized = os.environ.get(
            "MATRIXMAN_CUDA_DISABLE_SPECIALIZED_CONV", ""
        ).strip().lower() not in {"", "0", "false", "no", "off"}
        conv3x3_variant = os.environ.get(
            "MATRIXMAN_CUDA_CONV3X3_VARIANT", "plane"
        ).strip().lower()
        specialized_3x3_plane_legacy = (
            not disable_specialized and
            conv3x3_variant == "plane_legacy" and
            c == 64 and k == 64 and r == 3 and s == 3
            and stride_h == 1 and stride_w == 1
            and pad_h == 1 and pad_w == 1
            and dilation_h == 1 and dilation_w == 1 and groups == 1
        )
        specialized_1x1 = (
            not disable_specialized and
            n == 1 and c == 64 and k == 64 and r == 1 and s == 1
            and stride_h == 1 and stride_w == 1
            and pad_h == 0 and pad_w == 0
            and dilation_h == 1 and dilation_w == 1 and groups == 1
        )
        specialized_1x1_cin16 = (
            not disable_specialized and
            input_tensor.is_contiguous()
            and n == 1 and c == 16 and k >= 8 and r == 1 and s == 1
            and stride_h == 1 and stride_w == 1
            and pad_h == 0 and pad_w == 0
            and dilation_h == 1 and dilation_w == 1 and groups == 1
        )
        specialized_1x1_cin24 = (
            not disable_specialized and
            input_tensor.is_contiguous()
            and n == 1 and c == 24 and r == 1 and s == 1
            and stride_h == 1 and stride_w == 1
            and pad_h == 0 and pad_w == 0
            and dilation_h == 1 and dilation_w == 1 and groups == 1
        )
        specialized_1x1_cin36 = (
            not disable_specialized and
            input_tensor.is_contiguous()
            and n == 1 and c == 36 and r == 1 and s == 1
            and stride_h == 1 and stride_w == 1
            and pad_h == 0 and pad_w == 0
            and dilation_h == 1 and dilation_w == 1 and groups == 1
        )
        specialized_1x1_cin48 = (
            not disable_specialized and
            input_tensor.is_contiguous()
            and n == 1 and c == 48 and r == 1 and s == 1
            and stride_h == 1 and stride_w == 1
            and pad_h == 0 and pad_w == 0
            and dilation_h == 1 and dilation_w == 1 and groups == 1
        )
        specialized_1x1_cin72 = (
            not disable_specialized and
            input_tensor.is_contiguous()
            and n == 1 and c == 72 and r == 1 and s == 1
            and stride_h == 1 and stride_w == 1
            and pad_h == 0 and pad_w == 0
            and dilation_h == 1 and dilation_w == 1 and groups == 1
        )
        specialized_3x3_spatial = (
            not disable_specialized and
            conv3x3_variant == "spatial" and
            c == 64 and k == 64 and r == 3 and s == 3
            and stride_h == 1 and stride_w == 1
            and pad_h == 1 and pad_w == 1
            and dilation_h == 1 and dilation_w == 1 and groups == 1
        )
        specialized_3x3_plane = (
            not disable_specialized and
            conv3x3_variant in {"", "plane"} and
            c == 64 and k == 64 and r == 3 and s == 3
            and stride_h == 1 and stride_w == 1
            and pad_h == 1 and pad_w == 1
            and dilation_h == 1 and dilation_w == 1 and groups == 1
        )
        specialized_3x3_c8_c64_plane = (
            not disable_specialized and
            n == 1 and c == 8 and k == 64 and h == 80 and w == 80
            and r == 3 and s == 3
            and stride_h == 1 and stride_w == 1
            and pad_h == 1 and pad_w == 1
            and dilation_h == 1 and dilation_w == 1 and groups == 1
        )
        specialized_3x3_small_c8 = (
            not disable_specialized and
            input_tensor.is_contiguous() and
            c == 8 and k == 8
            and r == 3 and s == 3
            and stride_h == 1 and stride_w == 1
            and pad_h == 1 and pad_w == 1
            and dilation_h == 1 and dilation_w == 1 and groups == 1
        )
        specialized_3x3_small_c10 = (
            not disable_specialized and input_tensor.is_contiguous()
            and c == 10 and k == 10 and r == 3 and s == 3
            and stride_h == 1 and stride_w == 1
            and pad_h == 1 and pad_w == 1
            and dilation_h == 1 and dilation_w == 1 and groups == 1
        )
        specialized_3x3_small_c12 = (
            not disable_specialized and input_tensor.is_contiguous()
            and c == 12 and k == 12 and r == 3 and s == 3
            and stride_h == 1 and stride_w == 1
            and pad_h == 1 and pad_w == 1
            and dilation_h == 1 and dilation_w == 1 and groups == 1
        )
        specialized_3x3_small_c24 = (
            not disable_specialized and input_tensor.is_contiguous()
            and c == 24 and k == 24 and r == 3 and s == 3
            and stride_h == 1 and stride_w == 1
            and pad_h == 1 and pad_w == 1
            and dilation_h == 1 and dilation_w == 1 and groups == 1
        )
        specialized_3x3_c24_c64_plane = (
            not disable_specialized and
            n == 1 and c == 24 and k == 64 and h == 40 and w == 40
            and r == 3 and s == 3
            and stride_h == 1 and stride_w == 1
            and pad_h == 1 and pad_w == 1
            and dilation_h == 1 and dilation_w == 1 and groups == 1
        )
        specialized_3x3_c48_c64_plane = (
            not disable_specialized and
            n == 1 and c == 48 and k == 64 and h == 20 and w == 20
            and r == 3 and s == 3
            and stride_h == 1 and stride_w == 1
            and pad_h == 1 and pad_w == 1
            and dilation_h == 1 and dilation_w == 1 and groups == 1
        )

        weight_pointer, weight_uploaded = self._cached_parameter(weight)
        if weight_uploaded:
            profiling.count_conv2d("weight_uploads", weight.numel() * weight.element_size())
        bias_pointer = None
        output_pointer = None
        try:
            if bias is not None:
                bias_pointer, bias_uploaded = self._cached_parameter(bias)
                if bias_uploaded:
                    profiling.count_conv2d("bias_uploads", bias.numel() * bias.element_size())
            else:
                bias_pointer = type(input_tensor._owner.pointer)()
            output_pointer = self.execution.allocate(n * k * out_h * out_w * 4)
            profiling.count_conv2d("output_allocations", n * k * out_h * out_w * 4)
            self.execution.convolution(
                input_tensor._owner.pointer, weight_pointer, bias_pointer, output_pointer,
                n, c, h, w, k, r, s, out_h, out_w,
                stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups,
                specialized=specialized_3x3_plane_legacy,
                specialized_1x1=specialized_1x1,
                specialized_1x1_cin16=specialized_1x1_cin16,
                specialized_1x1_cin24=specialized_1x1_cin24,
                specialized_1x1_cin36=specialized_1x1_cin36,
                specialized_1x1_cin48=specialized_1x1_cin48,
                specialized_1x1_cin72=specialized_1x1_cin72,
                specialized_3x3_spatial=specialized_3x3_spatial,
                specialized_3x3_plane=specialized_3x3_plane,
                specialized_3x3_c8_c64_plane=specialized_3x3_c8_c64_plane,
                specialized_3x3_small_c8=specialized_3x3_small_c8,
                specialized_3x3_small_c10=specialized_3x3_small_c10,
                specialized_3x3_small_c12=specialized_3x3_small_c12,
                specialized_3x3_small_c24=specialized_3x3_small_c24,
                specialized_3x3_c24_c64_plane=specialized_3x3_c24_c64_plane,
                specialized_3x3_c48_c64_plane=specialized_3x3_c48_c64_plane,
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
        profiling.count_batch_norm("invocations")
        for value in (running_mean, running_var, weight, bias):
            pointer, uploaded = self._cached_parameter(value)
            parameter_pointers.append(pointer)
            if uploaded:
                profiling.count_batch_norm("parameter_uploads", value.numel() * value.element_size())
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

    def sigmoid(self, input_tensor):
        """Execute float32 sigmoid on a readable CUDA logical layout."""
        if not hasattr(input_tensor, "_owner") or input_tensor._owner.layout.kind != "cuda_linear":
            raise RuntimeError("MatrixMan/CUDA: sigmoid requires a CUDA-backed MatrixManTensor")
        if input_tensor.dtype != torch.float32:
            raise NotImplementedError("MatrixMan/CUDA: sigmoid supports float32 tensors only")
        shape = tuple(int(value) for value in input_tensor.shape)
        if not 1 <= len(shape) <= 4:
            raise NotImplementedError("MatrixMan/CUDA: sigmoid supports tensor ranks 1 through 4")
        strides = tuple(int(value) for value in input_tensor._logical_strides)
        if any(value < 0 for value in strides):
            raise NotImplementedError("MatrixMan/CUDA: sigmoid does not support negative logical strides")
        if input_tensor._owner.execution is not self.execution:
            raise RuntimeError("MatrixMan/CUDA: sigmoid tensor uses a different CUDA execution context")
        padding = 4 - len(shape)
        output_pointer = self.execution.allocate(_numel(shape) * 4)
        try:
            self.execution.sigmoid(
                input_tensor._owner.pointer,
                output_pointer,
                _numel(shape),
                (1,) * padding + shape,
                (0,) * padding + strides,
                int(input_tensor._storage_offset),
            )
            return CudaTensorOwner(self.execution, output_pointer, shape, contiguous_strides(shape))
        except Exception:
            self.execution.free(output_pointer)
            raise

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
        input_strides = tuple(int(value) for value in input_tensor._logical_strides)
        if len(input_strides) != rank or any(value < 0 for value in input_strides):
            raise NotImplementedError("MatrixMan/CUDA: split requires non-negative logical strides")
        if input_tensor._owner.execution is not self.execution:
            raise RuntimeError("MatrixMan/CUDA: split tensor uses a different CUDA execution context")
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
        padded_input_strides = (0,) * (4 - rank) + input_strides
        padded_dimension = dimension + 4 - rank
        owners = []
        offset = 0
        try:
            for chunk_size in chunks:
                output_shape = shape[:dimension] + (chunk_size,) + shape[dimension + 1:]
                padded_output = (1,) * (4 - rank) + output_shape
                output_pointer = self.execution.allocate(_numel(output_shape) * 4)
                try:
                    input_pointer = type(input_tensor._owner.pointer)(
                        input_tensor._owner.pointer.value
                        + int(input_tensor._storage_offset) * 4
                    )
                    self.execution.split_copy(
                        input_pointer,
                        output_pointer,
                        _numel(output_shape),
                        padded_dimension,
                        offset,
                        padded_input,
                        padded_input_strides,
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
        """Execute float32 tensor or scalar addition on CUDA."""
        left_is_tensor = hasattr(left, "_owner")
        right_is_tensor = hasattr(right, "_owner")
        if left_is_tensor and right_is_tensor:
            tensor = left
            other = right
        elif left_is_tensor:
            tensor = left
            other = right
        elif right_is_tensor:
            tensor = right
            other = left
        else:
            raise RuntimeError("MatrixMan/CUDA: add requires a MatrixMan tensor operand")
        if tensor._owner.layout.kind != "cuda_linear":
            raise RuntimeError("MatrixMan/CUDA: add tensor must be CUDA-backed")
        if tensor.dtype != torch.float32:
            raise NotImplementedError("MatrixMan/CUDA: add supports float32 tensors only")
        if not isinstance(tensor.shape, torch.Size):
            raise RuntimeError("MatrixMan/CUDA: add received invalid tensor metadata")
        shape = tuple(int(value) for value in tensor.shape)
        expected_strides = []
        stride = 1
        for size in reversed(shape):
            expected_strides.insert(0, stride)
            stride *= size
        try:
            alpha = float(alpha)
        except (TypeError, ValueError) as exc:
            raise NotImplementedError("MatrixMan/CUDA: add alpha must be a scalar") from exc

        output_pointer = self.execution.allocate(_numel(shape) * 4)
        try:
            if left_is_tensor and right_is_tensor:
                if right._owner.layout.kind != "cuda_linear" or right.dtype != torch.float32:
                    raise NotImplementedError("MatrixMan/CUDA: add supports CUDA float32 tensors only")
                if tuple(left.shape) != tuple(right.shape):
                    raise NotImplementedError("MatrixMan/CUDA: add broadcasting is not implemented; shapes must match")
                if not 1 <= len(shape) <= 4:
                    raise NotImplementedError("MatrixMan/CUDA: add supports tensor ranks 1 through 4")
                left_strides = tuple(int(value) for value in left._logical_strides)
                right_strides = tuple(int(value) for value in right._logical_strides)
                if left._owner.execution is not self.execution or right._owner.execution is not self.execution:
                    raise RuntimeError("MatrixMan/CUDA: add tensors use different CUDA execution contexts")
                if any(stride < 0 for stride in left_strides + right_strides):
                    raise NotImplementedError("MatrixMan/CUDA: add does not support negative logical strides")
                padding = 4 - len(shape)
                self.execution.add(
                    left._owner.pointer,
                    right._owner.pointer,
                    output_pointer,
                    _numel(shape),
                    (1,) * padding + shape,
                    (0,) * padding + left_strides,
                    (0,) * padding + right_strides,
                    int(left._storage_offset),
                    int(right._storage_offset),
                    alpha,
                )
            else:
                tensor_strides = tuple(int(value) for value in tensor._logical_strides)
                if tensor_strides != tuple(expected_strides):
                    raise NotImplementedError("MatrixMan/CUDA: scalar add requires contiguous tensors")
                if not isinstance(other, (int, float)):
                    raise NotImplementedError("MatrixMan/CUDA: scalar add requires a numeric scalar")
                if not left_is_tensor and alpha != 1.0:
                    raise NotImplementedError("MatrixMan/CUDA: scalar-left add alpha is unsupported")
                self.execution.add_scalar(
                    tensor._owner.pointer,
                    float(other) * alpha if left_is_tensor else float(other),
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

    def mul(self, left, right):
        """Multiply float32 CUDA tensors with aligned singleton broadcasting."""
        if not hasattr(left, "_owner") or not hasattr(right, "_owner"):
            raise RuntimeError("MatrixMan/CUDA: mul requires two CUDA-backed MatrixMan tensors")
        for name, tensor in (("left", left), ("right", right)):
            if tensor._owner.layout.kind != "cuda_linear":
                raise RuntimeError(f"MatrixMan/CUDA: mul {name} must be CUDA-backed")
            if tensor.dtype != torch.float32:
                raise NotImplementedError("MatrixMan/CUDA: mul supports float32 tensors only")
            if tensor._owner.execution is not self.execution:
                raise RuntimeError("MatrixMan/CUDA: mul tensors use different CUDA execution contexts")
            if any(int(value) < 0 for value in tensor._logical_strides):
                raise NotImplementedError("MatrixMan/CUDA: mul does not support negative logical strides")
        left_shape = tuple(int(value) for value in left.shape)
        right_shape = tuple(int(value) for value in right.shape)
        if not 1 <= len(left_shape) <= 4 or not 1 <= len(right_shape) <= 4:
            raise NotImplementedError("MatrixMan/CUDA: mul supports tensor ranks 1 through 4")
        rank = max(len(left_shape), len(right_shape))
        left_shape = (1,) * (rank - len(left_shape)) + left_shape
        right_shape = (1,) * (rank - len(right_shape)) + right_shape
        left_strides = (0,) * (rank - len(left._logical_strides)) + tuple(int(value) for value in left._logical_strides)
        right_strides = (0,) * (rank - len(right._logical_strides)) + tuple(int(value) for value in right._logical_strides)
        output_shape = []
        for axis, (left_size, right_size) in enumerate(zip(left_shape, right_shape)):
            if left_size == right_size:
                output_shape.append(left_size)
            elif left_size == 1:
                output_shape.append(right_size)
                left_strides = left_strides[:axis] + (0,) + left_strides[axis + 1:]
            elif right_size == 1:
                output_shape.append(left_size)
                right_strides = right_strides[:axis] + (0,) + right_strides[axis + 1:]
            else:
                raise NotImplementedError("MatrixMan/CUDA: mul shapes are not broadcast-compatible")
        padding = 4 - rank
        padded_shape = (1,) * padding + tuple(output_shape)
        padded_left_strides = (0,) * padding + left_strides
        padded_right_strides = (0,) * padding + right_strides
        output_shape = tuple(output_shape)
        output_pointer = self.execution.allocate(_numel(output_shape) * 4)
        try:
            self.execution.mul_elementwise(
                left._owner.pointer,
                right._owner.pointer,
                output_pointer,
                _numel(output_shape),
                padded_shape,
                padded_left_strides,
                padded_right_strides,
                int(left._storage_offset),
                int(right._storage_offset),
            )
            return CudaTensorOwner(self.execution, output_pointer, output_shape, contiguous_strides(output_shape))
        except Exception:
            self.execution.free(output_pointer)
            raise

    def div(self, input_tensor, divisor):
        """Divide a readable float32 CUDA tensor by a scalar on CUDA."""
        if not hasattr(input_tensor, "_owner") or input_tensor._owner.layout.kind != "cuda_linear":
            raise RuntimeError("MatrixMan/CUDA: div input must be CUDA-backed")
        if input_tensor.dtype != torch.float32:
            raise NotImplementedError("MatrixMan/CUDA: div supports float32 tensors only")
        try:
            divisor = float(divisor)
        except (TypeError, ValueError) as exc:
            raise NotImplementedError("MatrixMan/CUDA: div requires a scalar divisor") from exc
        shape = tuple(int(value) for value in input_tensor.shape)
        if not 1 <= len(shape) <= 4:
            raise NotImplementedError("MatrixMan/CUDA: div supports tensor ranks 1 through 4")
        strides = tuple(int(value) for value in input_tensor._logical_strides)
        if any(value < 0 for value in strides):
            raise NotImplementedError("MatrixMan/CUDA: div does not support negative logical strides")
        if input_tensor._owner.execution is not self.execution:
            raise RuntimeError("MatrixMan/CUDA: div tensor uses a different CUDA execution context")
        padding = 4 - len(shape)
        output_pointer = self.execution.allocate(_numel(shape) * 4)
        try:
            self.execution.div_scalar(
                input_tensor._owner.pointer,
                output_pointer,
                _numel(shape),
                (1,) * padding + shape,
                (0,) * padding + strides,
                int(input_tensor._storage_offset),
                divisor,
            )
            return CudaTensorOwner(
                self.execution,
                output_pointer,
                shape,
                contiguous_strides(shape),
            )
        except Exception:
            self.execution.free(output_pointer)
            raise

    def sub(self, left, right, alpha=1):
        """Subtract matching float32 CUDA tensors using logical view strides."""
        if not hasattr(left, "_owner") or not hasattr(right, "_owner"):
            raise RuntimeError("MatrixMan/CUDA: sub requires two CUDA-backed MatrixMan tensors")
        for name, tensor in (("left", left), ("right", right)):
            if tensor._owner.layout.kind != "cuda_linear":
                raise RuntimeError(f"MatrixMan/CUDA: sub {name} must be CUDA-backed")
            if tensor.dtype != torch.float32:
                raise NotImplementedError("MatrixMan/CUDA: sub supports float32 tensors only")
            if tensor._owner.execution is not self.execution:
                raise RuntimeError("MatrixMan/CUDA: sub tensors use different CUDA execution contexts")
            if any(int(stride) < 0 for stride in tensor._logical_strides):
                raise NotImplementedError("MatrixMan/CUDA: sub does not support negative logical strides")
        if tuple(left.shape) != tuple(right.shape):
            raise NotImplementedError("MatrixMan/CUDA: sub requires matching tensor shapes")
        try:
            alpha = float(alpha)
        except (TypeError, ValueError) as exc:
            raise NotImplementedError("MatrixMan/CUDA: sub alpha must be a scalar") from exc

        shape = tuple(int(value) for value in left.shape)
        if not 1 <= len(shape) <= 4:
            raise NotImplementedError("MatrixMan/CUDA: sub supports tensor ranks 1 through 4")
        padding = 4 - len(shape)
        padded_shape = (1,) * padding + shape
        padded_left_strides = (0,) * padding + tuple(int(value) for value in left._logical_strides)
        padded_right_strides = (0,) * padding + tuple(int(value) for value in right._logical_strides)
        output_pointer = self.execution.allocate(_numel(shape) * 4)
        try:
            self.execution.sub(
                left._owner.pointer,
                right._owner.pointer,
                output_pointer,
                _numel(shape),
                padded_shape,
                padded_left_strides,
                padded_right_strides,
                int(left._storage_offset),
                int(right._storage_offset),
                alpha,
            )
            return CudaTensorOwner(
                self.execution,
                output_pointer,
                shape,
                tuple(_numel(shape[index + 1:]) for index in range(len(shape))),
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
            if tuple(tensor._logical_strides) != tuple(
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
                input_pointer = type(tensor._owner.pointer)(
                    tensor._owner.pointer.value + int(tensor._storage_offset) * 4
                )
                self.execution.cat_copy(
                    input_pointer,
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

    def stack(self, tensors, dimension):
        """Stack matching float32 tensors through a stride-aware CUDA copy."""
        tensors = tuple(tensors)
        if not tensors:
            raise ValueError("MatrixMan/CUDA: stack requires at least one tensor")
        first_shape = tuple(int(value) for value in tensors[0].shape)
        rank = len(first_shape)
        if rank < 1 or rank > 4:
            raise NotImplementedError("MatrixMan/CUDA: stack supports tensor ranks 1 through 4")
        dimension = int(dimension)
        if dimension < 0:
            dimension += rank + 1
        if dimension < 0 or dimension > rank:
            raise ValueError("MatrixMan/CUDA: stack dimension is out of range")
        for index, tensor in enumerate(tensors):
            if (
                not hasattr(tensor, "_owner")
                or tensor._owner.layout.kind != "cuda_linear"
                or tensor.dtype != torch.float32
            ):
                raise RuntimeError(
                    f"MatrixMan/CUDA: stack input {index} must be a CUDA-backed float32 tensor"
                )
            if tuple(int(value) for value in tensor.shape) != first_shape:
                raise ValueError("MatrixMan/CUDA: stack inputs must have matching shapes")
            if tensor._owner.execution is not self.execution:
                raise RuntimeError("MatrixMan/CUDA: stack inputs must share one CUDA execution context")

        output_shape = first_shape[:dimension] + (len(tensors),) + first_shape[dimension:]
        output_strides = tuple(_numel(output_shape[index + 1:]) for index in range(rank + 1))
        output_pointer = self.execution.allocate(_numel(output_shape) * 4)
        suffix = _numel(first_shape[dimension:])
        padded_shape = (1,) * (4 - rank) + first_shape
        try:
            for stack_index, tensor in enumerate(tensors):
                padded_strides = (0,) * (4 - rank) + tuple(
                    int(value) for value in tensor._logical_strides
                )
                input_pointer = type(tensor._owner.pointer)(
                    tensor._owner.pointer.value + int(tensor._storage_offset) * 4
                )
                self.execution.stack_copy(
                    input_pointer,
                    output_pointer,
                    _numel(first_shape),
                    suffix,
                    len(tensors),
                    stack_index,
                    padded_shape,
                    padded_strides,
                )
            return CudaTensorOwner(self.execution, output_pointer, output_shape, output_strides)
        except Exception:
            self.execution.free(output_pointer)
            raise

    def fill(self, tensor, value):
        """Fill a CUDA-backed float32 tensor in place using logical strides."""
        if not hasattr(tensor, "_owner") or tensor._owner.layout.kind != "cuda_linear":
            raise RuntimeError("MatrixMan/CUDA: fill requires a CUDA-backed MatrixManTensor")
        if tensor.dtype != torch.float32:
            raise NotImplementedError("MatrixMan/CUDA: fill supports float32 tensors only")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise NotImplementedError("MatrixMan/CUDA: fill value must be a scalar") from exc
        shape = tuple(int(item) for item in tensor.shape)
        strides = tuple(int(item) for item in tensor._logical_strides)
        if len(shape) != len(strides) or len(shape) < 1 or len(shape) > 4:
            raise NotImplementedError("MatrixMan/CUDA: fill supports ranks 1 through 4")
        if any(stride < 0 for stride in strides):
            raise NotImplementedError("MatrixMan/CUDA: fill does not support negative strides")
        if any(size > 1 and stride == 0 for size, stride in zip(shape, strides)):
            raise NotImplementedError("MatrixMan/CUDA: fill rejects overlapping zero-stride views")
        if tensor._owner.execution is not self.execution:
            raise RuntimeError("MatrixMan/CUDA: fill tensor uses a different CUDA execution context")
        if not _numel(shape):
            return tensor
        padded_shape = (1,) * (4 - len(shape)) + shape
        padded_strides = (0,) * (4 - len(strides)) + strides
        pointer = type(tensor._owner.pointer)(
            tensor._owner.pointer.value + int(tensor._storage_offset) * 4
        )
        self.execution.fill(pointer, value, _numel(shape), padded_shape, padded_strides)
        return tensor

    def softmax(self, tensor, dimension, half_to_float=False):
        """Execute stable float32 softmax while honoring logical input strides."""
        if not hasattr(tensor, "_owner") or tensor._owner.layout.kind != "cuda_linear":
            raise RuntimeError("MatrixMan/CUDA: softmax requires a CUDA-backed MatrixManTensor")
        if tensor.dtype != torch.float32:
            raise NotImplementedError("MatrixMan/CUDA: softmax supports float32 tensors only")
        shape = tuple(int(item) for item in tensor.shape)
        strides = tuple(int(item) for item in tensor._logical_strides)
        rank = len(shape)
        if rank < 1 or rank > 4:
            raise NotImplementedError("MatrixMan/CUDA: softmax supports ranks 1 through 4")
        dimension = int(dimension)
        if dimension < 0:
            dimension += rank
        if dimension < 0 or dimension >= rank:
            raise IndexError("MatrixMan/CUDA: softmax dimension is out of range")
        if len(strides) != rank or any(stride < 0 for stride in strides):
            raise NotImplementedError("MatrixMan/CUDA: softmax requires non-negative logical strides")
        if tensor._owner.execution is not self.execution:
            raise RuntimeError("MatrixMan/CUDA: softmax tensor uses a different CUDA execution context")
        if shape[dimension] <= 0:
            raise NotImplementedError("MatrixMan/CUDA: softmax does not support empty dimensions")
        if not _numel(shape):
            raise NotImplementedError("MatrixMan/CUDA: softmax does not support empty tensors")
        output_strides = tuple(_numel(shape[index + 1:]) for index in range(rank))
        outer_strides = []
        for index in range(rank):
            product = 1
            for following in range(index + 1, rank):
                if following != dimension:
                    product *= shape[following]
            outer_strides.append(product)
        padded_shape = (1,) * (4 - rank) + shape
        padded_strides = (0,) * (4 - rank) + strides
        padded_output_strides = (0,) * (4 - rank) + output_strides
        padded_outer_strides = (0,) * (4 - rank) + tuple(outer_strides)
        output_shape = shape
        output_pointer = self.execution.allocate(_numel(output_shape) * 4)
        input_pointer = type(tensor._owner.pointer)(
            tensor._owner.pointer.value + int(tensor._storage_offset) * 4
        )
        try:
            self.execution.softmax(
                input_pointer,
                output_pointer,
                _numel(shape) // shape[dimension],
                shape[dimension],
                dimension + 4 - rank,
                padded_strides[dimension + 4 - rank],
                padded_output_strides[dimension + 4 - rank],
                padded_shape,
                padded_strides,
                padded_output_strides,
                padded_outer_strides,
            )
            return CudaTensorOwner(self.execution, output_pointer, output_shape, output_strides)
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
        self.execution.synchronize("shutdown")
        for _, (_, _, pointer, _) in list(self._parameter_cache.items()):
            self.execution.free(pointer)
        self._parameter_cache.clear()
        self.execution.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @classmethod
    def probe(cls) -> bool:
        try:
            detect_device()
        except Exception:
            return False
        return True
