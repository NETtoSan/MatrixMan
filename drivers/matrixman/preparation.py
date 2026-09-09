"""Small, explicit model-preparation experiments for MatrixMan."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
import time
from weakref import WeakKeyDictionary

import torch
from torch import nn

from .config import config


_PREPARATION_MARKER = "_matrixman_preparation_kind"
_AUTO_PREPARE_CACHE: WeakKeyDictionary[nn.Module, dict[str, tuple[tuple, nn.Module]]] = WeakKeyDictionary()


def _model_signature(model: nn.Module) -> tuple:
    """Capture cheap topology/state facts relevant to folded parameters."""
    modules = []
    for name, module in model.named_modules(remove_duplicate=False):
        scalar_state = []
        for key, value in module.__dict__.items():
            if key.startswith("_") or key in {"training", "shape", "anchors", "strides"}:
                continue
            if value is None or isinstance(value, (bool, int, float, str)):
                scalar_state.append((key, value))
            elif isinstance(value, tuple) and all(
                item is None or isinstance(item, (bool, int, float, str)) for item in value
            ):
                scalar_state.append((key, value))
        modules.append((name, type(module).__module__, type(module).__qualname__, module.training, tuple(sorted(scalar_state))))
    parameters = []
    for name, parameter in model.named_parameters(remove_duplicate=False):
        parameters.append((
            name, id(parameter), int(parameter._version), int(parameter.data_ptr()),
            tuple(int(item) for item in parameter.shape), tuple(int(item) for item in parameter.stride()),
            str(parameter.dtype), str(parameter.device), bool(parameter.requires_grad),
        ))
    buffers = []
    for name, buffer in model.named_buffers(remove_duplicate=False):
        buffers.append((
            name, id(buffer), int(buffer._version), int(buffer.data_ptr()),
            tuple(int(item) for item in buffer.shape), str(buffer.dtype), str(buffer.device),
        ))
    return tuple(modules), tuple(parameters), tuple(buffers)


def _auto_prepare_log(message: str) -> None:
    if config.debug or config.trace:
        print(f"[MatrixMan] auto-prepare: {message}")


def _lazy_prepared_model(model: nn.Module, backend_name: str) -> nn.Module:
    signature = _model_signature(model)
    entries = _AUTO_PREPARE_CACHE.setdefault(model, {})
    entry = entries.get(backend_name)
    if entry is not None and entry[0] == signature:
        _auto_prepare_log("cache hit")
        return entry[1]
    _auto_prepare_log(f"preparing {backend_name} model")
    started = time.perf_counter()
    prepared = prepare(model, backend=backend_name, inplace=False)
    setattr(prepared, _PREPARATION_MARKER, "lazy")
    entries[backend_name] = (signature, prepared)
    if config.debug or config.trace:
        _auto_prepare_log(f"prepared in {(time.perf_counter() - started) * 1000.0:.1f} ms")
    return prepared


class LazyPreparedModel(nn.Module):
    """Narrow model-level boundary for safe lazy MatrixMan preparation."""

    def __init__(self, model: nn.Module):
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("matrixman.auto_prepare requires a torch.nn.Module")
        self.model = model

    def forward(self, *args, **kwargs):
        from .backend import get_backend
        from .tensor import MatrixManTensor

        input_tensor = args[0] if args else kwargs.get("x")
        if not isinstance(input_tensor, MatrixManTensor):
            return self.model(*args, **kwargs)
        backend = get_backend()
        if backend.name not in {"cuda", "opengl"}:
            _auto_prepare_log("skipped, backend does not support preparation")
            return self.model(*args, **kwargs)
        if getattr(self.model, _PREPARATION_MARKER, None):
            return self.model(*args, **kwargs)
        if config.disableAutoPrepare:
            _auto_prepare_log("disabled")
            return self.model(*args, **kwargs)
        if self.model.training:
            _auto_prepare_log("skipped, training mode")
            return self.model(*args, **kwargs)
        if torch.is_grad_enabled():
            _auto_prepare_log("skipped, autograd is enabled")
            return self.model(*args, **kwargs)
        prepared = _lazy_prepared_model(self.model, backend.name)
        return prepared(*args, **kwargs)


def auto_prepare(model: nn.Module) -> LazyPreparedModel:
    """Return a model wrapper that lazily prepares eligible CUDA inference."""
    if isinstance(model, LazyPreparedModel):
        return model
    return LazyPreparedModel(model)


class PreparedCudaConvSiLU(nn.Module):
    """Inference-only Conv/SiLU module using direct MatrixMan CUDA calls."""

    def __init__(self, conv: nn.Module, activation: nn.Module, source: nn.Module | None = None):
        super().__init__()
        self.conv = conv
        self.bn = nn.Identity()
        self.act = activation
        if source is not None:
            # Ultralytics' graph runner stores routing metadata (notably `f`
            # and `i`) directly on each layer rather than in its state dict.
            for key, value in source.__dict__.items():
                if not key.startswith("_") and key not in {"training"}:
                    setattr(self, key, value)

    @classmethod
    def from_module(cls, module: nn.Module) -> "PreparedCudaConvSiLU":
        return cls(module.conv, module.act, source=module)

    def forward(self, input_tensor):
        from .backend import get_backend
        from .tensor import MatrixManTensor

        if not isinstance(input_tensor, MatrixManTensor):
            raise RuntimeError("PreparedCudaConvSiLU requires a MatrixMan CUDA tensor")
        backend = get_backend()
        if backend.name != "cuda":
            raise RuntimeError("PreparedCudaConvSiLU requires the CUDA backend")
        owner = backend.convolution_silu(
            input_tensor,
            self.conv.weight,
            self.conv.bias,
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
            self.conv.groups,
        )
        return type(input_tensor)._from_owner(owner, owner.shape, logical_strides=owner.strides)


class PreparedCudaDFL(nn.Module):
    """Inference-only DFL chain using direct CUDA softmax and Conv primitives."""

    def __init__(self, conv: nn.Conv2d, c1: int, source: nn.Module | None = None):
        super().__init__()
        self.conv = conv
        self.c1 = int(c1)
        if source is not None:
            for key, value in source.__dict__.items():
                if not key.startswith("_") and key not in {"training"}:
                    setattr(self, key, value)

    @classmethod
    def from_module(cls, module: nn.Module) -> "PreparedCudaDFL":
        return cls(module.conv, module.c1, source=module)

    def forward(self, input_tensor):
        from .backend import get_backend
        from .tensor import MatrixManTensor, contiguous_strides

        if not isinstance(input_tensor, MatrixManTensor):
            raise RuntimeError("PreparedCudaDFL requires a MatrixMan CUDA tensor")
        backend = get_backend()
        if backend.name != "cuda":
            raise RuntimeError("PreparedCudaDFL requires the CUDA backend")
        batch, channels, anchors = (int(value) for value in input_tensor.shape)
        if channels != 4 * self.c1:
            raise RuntimeError(
                f"PreparedCudaDFL expected {4 * self.c1} channels, got {channels}"
            )
        reshaped_shape = (batch, 4, self.c1, anchors)
        reshaped = type(input_tensor)._from_owner(
            input_tensor._owner,
            reshaped_shape,
            storage_offset=input_tensor._storage_offset,
            logical_strides=contiguous_strides(reshaped_shape),
        )
        transposed_shape = (batch, self.c1, 4, anchors)
        transposed_strides = (
            reshaped._logical_strides[0],
            reshaped._logical_strides[2],
            reshaped._logical_strides[1],
            reshaped._logical_strides[3],
        )
        transposed = type(input_tensor)._from_owner(
            input_tensor._owner,
            transposed_shape,
            storage_offset=input_tensor._storage_offset,
            logical_strides=transposed_strides,
        )
        softmax_owner = backend.softmax(transposed, 1)
        softmax_tensor = type(input_tensor)._from_owner(
            softmax_owner, softmax_owner.shape, logical_strides=softmax_owner.strides
        )
        try:
            convolution_owner = backend.convolution(
                softmax_tensor,
                self.conv.weight,
                None,
                self.conv.stride,
                self.conv.padding,
                self.conv.dilation,
                self.conv.groups,
            )
        finally:
            softmax_owner.release()
        output_shape = (batch, 4, anchors)
        return type(input_tensor)._from_owner(
            convolution_owner,
            output_shape,
            logical_strides=contiguous_strides(output_shape),
        )


class PreparedOpenGLConvSiLU(nn.Module):
    """Inference-only Conv/SiLU module using direct OpenGL primitives."""

    def __init__(self, conv: nn.Module, activation: nn.Module, source: nn.Module | None = None):
        super().__init__()
        self.conv = conv
        self.bn = nn.Identity()
        self.act = activation
        if source is not None:
            for key, value in source.__dict__.items():
                if not key.startswith("_") and key not in {"training"}:
                    setattr(self, key, value)

    @classmethod
    def from_module(cls, module: nn.Module) -> "PreparedOpenGLConvSiLU":
        return cls(module.conv, module.act, source=module)

    def forward(self, input_tensor):
        from .backend import get_backend
        from .backends.opengl import convolution
        from .backends.opengl.ops import activation
        from .tensor import MatrixManTensor

        if not isinstance(input_tensor, MatrixManTensor):
            raise RuntimeError("PreparedOpenGLConvSiLU requires a MatrixMan OpenGL tensor")
        if get_backend().name != "opengl":
            raise RuntimeError("PreparedOpenGLConvSiLU requires the OpenGL backend")
        output = convolution.execute_prepared((
            input_tensor, self.conv.weight, self.conv.bias,
            self.conv.stride, self.conv.padding, self.conv.dilation,
            False, (0, 0), self.conv.groups,
        ))
        return activation._render_silu_inplace((output,))


@dataclass
class PreparationReport:
    """Summary of one opt-in inference-only preparation pass."""

    backend: str = "cuda"

    conv_bn_pairs_found: int = 0
    folded: int = 0
    skipped: int = 0
    original_batch_norm_modules: int = 0
    remaining_batch_norm_modules: int = 0
    conv_silu_candidates: int = 0
    conv_silu_prepared: int = 0
    conv_silu_skipped: int = 0
    detection_head_candidates: int = 0
    detection_head_prepared: int = 0
    detection_head_skipped: int = 0
    skip_reasons: Counter[str] = field(default_factory=Counter)

    def print(self) -> None:
        print(f"MatrixMan {self.backend.upper()} preparation (experimental, inference-only)")
        print(f"  Conv+BN pairs found: {self.conv_bn_pairs_found}")
        print(f"  folded: {self.folded}")
        print(f"  skipped: {self.skipped}")
        print(f"  original BatchNorm modules: {self.original_batch_norm_modules}")
        print(f"  remaining BatchNorm modules: {self.remaining_batch_norm_modules}")
        print(f"  Conv+SiLU candidates: {self.conv_silu_candidates}")
        print(f"  Conv+SiLU prepared: {self.conv_silu_prepared}")
        print(f"  Conv+SiLU skipped: {self.conv_silu_skipped}")
        print(f"  detection-head DFL candidates: {self.detection_head_candidates}")
        print(f"  detection-head DFL prepared: {self.detection_head_prepared}")
        print(f"  detection-head DFL skipped: {self.detection_head_skipped}")
        if self.skip_reasons:
            print("  skip reasons:")
            for reason, count in sorted(self.skip_reasons.items()):
                print(f"    {reason}: {count}")


def _module_alias_counts(model: nn.Module) -> Counter[int]:
    counts: Counter[int] = Counter()
    for _name, module in model.named_modules(remove_duplicate=False):
        counts[id(module)] += 1
    return counts


def _fold_pair(module: nn.Module, report: PreparationReport, aliases: Counter[int]) -> None:
    conv = getattr(module, "conv", None)
    bn = getattr(module, "bn", None)
    if conv is None and bn is None:
        return
    if not isinstance(conv, nn.Conv2d) or not isinstance(bn, nn.BatchNorm2d):
        return
    report.conv_bn_pairs_found += 1
    reason = None
    if aliases[id(module)] > 1 or aliases[id(conv)] > 1 or aliases[id(bn)] > 1:
        reason = "shared_module_alias"
    elif module.__class__.__name__ not in {"Conv", "DWConv"}:
        reason = "unsupported_conv_module_type"
    elif any(hasattr(module, name) for name in ("cv1", "cv2", "conv1", "conv2")):
        reason = "multiple_conv_producers"
    elif conv.weight.dtype != torch.float32 or bn.running_mean is None or bn.running_var is None:
        reason = "unsupported_dtype_or_missing_running_stats"
    elif conv.weight.device.type != "cpu" or bn.running_mean.device.type != "cpu":
        reason = "non_cpu_parameter_device"
    elif conv.out_channels != bn.num_features:
        reason = "channel_count_mismatch"
    elif bn.training:
        reason = "batch_norm_training"
    if reason is not None:
        report.skipped += 1
        report.skip_reasons[reason] += 1
        return

    with torch.no_grad():
        conv_bias = torch.zeros_like(bn.running_mean) if conv.bias is None else conv.bias.detach()
        bn_weight = torch.ones_like(bn.running_mean) if bn.weight is None else bn.weight.detach()
        bn_bias = torch.zeros_like(bn.running_mean) if bn.bias is None else bn.bias.detach()
        scale = bn_weight / torch.sqrt(bn.running_var.detach() + bn.eps)
        folded_weight = conv.weight.detach() * scale.reshape(-1, 1, 1, 1)
        folded_bias = (conv_bias - bn.running_mean.detach()) * scale + bn_bias
        conv.weight = nn.Parameter(folded_weight, requires_grad=conv.weight.requires_grad)
        conv.bias = nn.Parameter(folded_bias, requires_grad=False)
    module.bn = nn.Identity()
    report.folded += 1


def _prepare_conv_silu_children(
    parent: nn.Module,
    report: PreparationReport,
    aliases: Counter[int],
    backend: str = "cuda",
) -> None:
    """Replace safe folded Conv modules with backend-specific direct units."""
    for name, module in list(parent.named_children()):
        conv = getattr(module, "conv", None)
        bn = getattr(module, "bn", None)
        activation = getattr(module, "act", None)
        if conv is None or bn is None or activation is None:
            _prepare_conv_silu_children(module, report, aliases, backend)
            continue
        if not isinstance(conv, nn.Conv2d) or not isinstance(activation, nn.SiLU):
            _prepare_conv_silu_children(module, report, aliases, backend)
            continue
        report.conv_silu_candidates += 1
        reason = None
        if not isinstance(bn, nn.Identity):
            reason = "batch_norm_not_folded"
        # Ultralytics commonly shares one stateless nn.SiLU instance across
        # blocks. Duplicating that activation object does not change behavior;
        # sharing the producer module or its parameters would.
        elif aliases[id(module)] > 1 or aliases[id(conv)] > 1:
            reason = "shared_module_alias"
        elif module.__class__.__name__ not in {"Conv", "DWConv"}:
            reason = "unsupported_conv_module_type"
        elif conv.weight.dtype != torch.float32 or conv.weight.device.type != "cpu":
            reason = "unsupported_dtype_or_parameter_device"
        elif conv.bias is None or conv.bias.dtype != torch.float32 or not conv.bias.is_contiguous():
            reason = "missing_or_unsupported_bias"
        elif not conv.weight.is_contiguous():
            reason = "noncontiguous_weight"
        elif activation.training or conv.training:
            reason = "module_training"
        elif not bool(getattr(activation, "inplace", False)):
            reason = "non_inplace_activation"
        if reason is not None:
            report.conv_silu_skipped += 1
            report.skip_reasons[f"conv_silu:{reason}"] += 1
            _prepare_conv_silu_children(module, report, aliases, backend)
            continue
        prepared_type = PreparedCudaConvSiLU if backend == "cuda" else PreparedOpenGLConvSiLU
        setattr(parent, name, prepared_type.from_module(module))
        report.conv_silu_prepared += 1


def _prepare_detection_head_children(
    parent: nn.Module,
    report: PreparationReport,
    aliases: Counter[int],
) -> None:
    """Replace only the exact Ultralytics DFL topology."""
    for name, module in list(parent.named_children()):
        if module.__class__.__name__ != "DFL":
            _prepare_detection_head_children(module, report, aliases)
            continue
        report.detection_head_candidates += 1
        conv = getattr(module, "conv", None)
        reason = None
        if not isinstance(conv, nn.Conv2d) or int(getattr(module, "c1", 0)) <= 0:
            reason = "unsupported_dfl_structure"
        elif aliases[id(module)] > 1 or aliases[id(conv)] > 1:
            reason = "shared_module_alias"
        elif module.training or conv.training:
            reason = "module_training"
        elif conv.bias is not None:
            reason = "unexpected_bias"
        elif conv.weight.dtype != torch.float32 or conv.weight.device.type != "cpu":
            reason = "unsupported_dtype_or_parameter_device"
        elif not conv.weight.is_contiguous():
            reason = "noncontiguous_weight"
        elif tuple(conv.kernel_size) != (1, 1) or tuple(conv.stride) != (1, 1):
            reason = "unsupported_dfl_conv_geometry"
        elif tuple(conv.padding) != (0, 0) or tuple(conv.dilation) != (1, 1) or conv.groups != 1:
            reason = "unsupported_dfl_conv_geometry"
        elif conv.in_channels != int(module.c1) or conv.out_channels != 1:
            reason = "dfl_channel_mismatch"
        if reason is not None:
            report.detection_head_skipped += 1
            report.skip_reasons[f"detection_head:{reason}"] += 1
            continue
        setattr(parent, name, PreparedCudaDFL.from_module(module))
        report.detection_head_prepared += 1


def _prepare_model(model: nn.Module, *, backend: str, inplace: bool, diagnostics: bool):
    """Prepare an eval model for CUDA by folding safe direct Conv+BN pairs.

    Preparation is explicit, experimental, CUDA-only, and may use CPU tensor
    arithmetic once before inference. Unsupported pairs remain unchanged.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("matrixman.prepare requires a torch.nn.Module")
    if model.training:
        raise ValueError("matrixman.prepare_cuda requires model.eval(); training BatchNorm is not foldable")
    if getattr(model, _PREPARATION_MARKER, None):
        return model
    target = model if inplace else deepcopy(model)
    report = PreparationReport(
        backend=backend,
        original_batch_norm_modules=sum(isinstance(module, nn.BatchNorm2d) for module in target.modules())
    )
    aliases = _module_alias_counts(target)
    for _name, module in target.named_modules(remove_duplicate=False):
        _fold_pair(module, report, aliases)
    _prepare_conv_silu_children(target, report, aliases, backend)
    if backend == "cuda":
        _prepare_detection_head_children(target, report, aliases)
    else:
        dfl_count = sum(module.__class__.__name__ == "DFL" for module in target.modules())
        report.detection_head_candidates += dfl_count
        report.detection_head_skipped += dfl_count
        if dfl_count:
            report.skip_reasons["detection_head:direct_openGL_chain_unavailable"] += dfl_count
    report.remaining_batch_norm_modules = sum(
        isinstance(module, nn.BatchNorm2d) for module in target.modules()
    )
    if diagnostics:
        report.print()
    target._matrixman_preparation_report = report
    target._matrixman_cuda_preparation_report = report
    setattr(target, _PREPARATION_MARKER, "explicit" if inplace else "explicit-copy")
    return target


def prepare_cuda(model: nn.Module, *, inplace: bool = False, diagnostics: bool = False):
    """Prepare an eval model for CUDA."""
    return _prepare_model(model, backend="cuda", inplace=inplace, diagnostics=diagnostics)


def prepare_opengl(model: nn.Module, *, inplace: bool = False, diagnostics: bool = False):
    """Prepare an eval model for OpenGL using shared safe transforms."""
    return _prepare_model(model, backend="opengl", inplace=inplace, diagnostics=diagnostics)


def prepare(model: nn.Module, *, backend: str = "cuda", inplace: bool = False, diagnostics: bool = False):
    """Explicit experimental inference preparation for CUDA or OpenGL."""
    backend_name = str(backend).strip().lower()
    if backend_name not in {"cuda", "opengl"}:
        raise ValueError("matrixman.prepare supports backend='cuda' or backend='opengl'")
    return _prepare_model(model, backend=backend_name, inplace=inplace, diagnostics=diagnostics)
