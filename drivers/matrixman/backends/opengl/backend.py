"""Public OpenGL backend entrypoint and compatibility exports."""

from ...backend import Backend
from . import diagnostics, factories, gpumatrix as gm, profiling, runtime, tensor as tensor_module
from . import metadata, operation_context, resources
from .ops import matmul, postprocessing
from ...tensor import (
    Gm45Tensor,
    MatrixManTensor,
    PRIVATEUSE_DEVICE,
    install_tensor_method,
    is_gm45_tensor,
    is_matrixman_tensor,
)


class OpenGLBackend(Backend):
    name = "opengl"

    def __init__(self):
        # Factory registration is selection-time work.  Importing this module
        # must not install OpenGL PrivateUse1 implementations for CUDA.
        factories.install_privateuse1_factory_kernels()

    def device_info(self) -> dict[str, str]:
        return device_info()

    @classmethod
    def probe(cls) -> bool:
        try:
            runtime.init()
        except Exception:
            return False
        return runtime.is_active()

    def matmul(self, a, b):
        return matmul.render_matmul(a, b)

    def synchronize(self):
        gm.glFinish()


def device_info() -> dict[str, str]:
    """Return device and shading-language information from the active context."""

    def text(value) -> str:
        if value is None:
            return "unavailable"
        if isinstance(value, bytes):
            return value.decode("utf-8", "replace")
        if isinstance(value, str):
            return value
        return str(value)

    runtime.init()
    values = {
        "backend": "OpenGL",
        "vendor": gm.glGetString(0x1F00),
        "renderer": gm.glGetString(0x1F01),
        "opengl": gm.glGetString(0x1F02),
        "glsl": gm.glGetString(0x8B8C),
    }
    renderer_text = text(values["renderer"])
    vendor_text = text(values["vendor"])
    policy = classify_renderer(vendor_text, renderer_text)
    return {
        key: text(value)
        for key, value in values.items()
    } | {"device_policy": policy}


def classify_renderer(vendor: str, renderer: str) -> str:
    """Describe renderer preference without affecting GL availability."""
    lower_renderer = renderer.lower()
    software = any(name in lower_renderer for name in ("llvmpipe", "softpipe", "swrast"))
    if software:
        policy = "software renderer; hardware GPU preferred when available"
    elif "nvidia" in vendor.lower() or "nvidia" in renderer.lower():
        policy = "NVIDIA fallback; alternate renderer selection unavailable via SDL"
    else:
        policy = "preferred non-NVIDIA renderer"
    return policy


def init() -> None:
    runtime.init()


def shutdown() -> None:
    runtime.shutdown()


def set_trace(enabled: bool = True) -> None:
    diagnostics.set_trace(enabled)


def debug_enabled() -> bool:
    return diagnostics.debug_enabled()


def profile_enabled() -> bool:
    return profiling.is_enabled()


def profile_report() -> None:
    profiling.report()


def profile_reset() -> None:
    profiling.reset()


def reset_unsupported_report() -> None:
    diagnostics.reset_unsupported_report()


def unsupported_report() -> dict[str, dict]:
    return diagnostics.unsupported_report()


tensor = tensor_module.tensor
randn = tensor_module.randn
to_gm45 = tensor_module.tensor


def gpu_postprocess_detection(tensor):
    """Experimental GPU-side fixed-size detection reduction."""
    return postprocessing.reduce_detection_output(tensor)

# Narrow compatibility service surface used by convolution.py.  These are
# direct aliases to the extracted owners, not a second implementation path.
_runtime_required = runtime.runtime_required
_trace = diagnostics.trace
_kernel_log = diagnostics.kernel_log
_shape_text = diagnostics.shape_text
_require_contiguous_logical = metadata.require_contiguous_logical
_new_empty_packed_texture = operation_context.output_texture
_read_texture = tensor_module.readback_tensor
_TextureOwner = tensor_module._TextureOwner
_profile_enabled = profiling.enabled
_profile_counters = profiling.counters
_profile_conv = profiling.conv


install_tensor_method()
