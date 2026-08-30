"""Public OpenGL backend entrypoint and compatibility exports."""

from ...backend import Backend
from . import diagnostics, factories, gpumatrix as gm, profiling, runtime, tensor as tensor_module
from . import metadata, operation_context, resources
from .ops import matmul


class OpenGLBackend(Backend):
    name = "opengl"

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


Gm45Tensor = tensor_module.Gm45Tensor
PRIVATEUSE_DEVICE = tensor_module.PRIVATEUSE_DEVICE


def init() -> None:
    runtime.init()


def shutdown() -> None:
    runtime.shutdown()


def set_trace(enabled: bool = True) -> None:
    diagnostics.set_trace(enabled)


def debug_enabled() -> bool:
    return diagnostics.debug_enabled()


def profile_enabled() -> bool:
    return profiling.enabled


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
to_gm45 = tensor_module.to_gm45
is_gm45_tensor = tensor_module.is_gm45_tensor
install_tensor_method = tensor_module.install_tensor_method

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


_tensor_module = __import__(__name__.rsplit(".", 1)[0] + ".tensor", fromlist=["tensor"])
_tensor_module.install_tensor_method()
factories.install_privateuse1_factory_kernels()
