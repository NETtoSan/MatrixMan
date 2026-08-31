"""Backward-compatible MatrixMan frontend façade."""

import warnings

from .backend import get_backend
from .tensor import (
    MatrixManTensor,
    install_tensor_method as _install_tensor_method,
    is_matrixman_tensor as _is_matrixman_tensor,
)


def _opengl_implementation():
    """Load the OpenGL façade only when an OpenGL operation is requested."""
    return __import__(
        "drivers.matrixman.backends.opengl.backend",
        fromlist=["backend"],
    )


def _require_opengl_frontend(operation: str):
    backend = get_backend()
    if backend.name != "opengl":
        raise NotImplementedError(f"MatrixMan/{backend.name.upper()}: {operation} not implemented")
    return _opengl_implementation()


def init() -> None:
    if get_backend().name == "cuda":
        return
    _require_opengl_frontend("initialization").init()


def shutdown() -> None:
    backend = get_backend()
    if backend.name == "cuda":
        close = getattr(backend, "close", None)
        if close is not None:
            close()
        return
    _require_opengl_frontend("shutdown").shutdown()


def tensor(data, *, device=None):
    backend = get_backend()
    if backend.name == "cuda":
        import torch

        from .backends.cuda.backend import upload_tensor

        if device is not None and str(device) not in {"matrixman", "matrixman:0", "privateuseone", "privateuseone:0"}:
            raise RuntimeError("MatrixMan/CUDA: tensor only creates tensors on the MatrixMan device")
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        owner = upload_tensor(data, backend.execution)
        return MatrixManTensor._from_owner(
            owner, owner.shape, logical_strides=owner.strides
        )
    return _require_opengl_frontend("tensor upload").tensor(data, device=device)


def to_device(data):
    return tensor(data)


def to_gm45(data):
    """Deprecated compatibility alias for :func:`to_device`."""
    warnings.warn(
        "matrixman.to_gm45() is deprecated; use matrixman.to_device() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return to_device(data)


def randn(*shape: int, seed: int | None = None):
    if get_backend().name == "cuda":
        import torch

        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)
        if len(shape) == 1:
            shape = (shape[0], shape[0])
        return tensor(torch.randn(shape, dtype=torch.float32, generator=generator))
    return _require_opengl_frontend("random tensor creation").randn(*shape, seed=seed)


def is_matrixman_tensor(value) -> bool:
    return _is_matrixman_tensor(value)


def is_gm45_tensor(value) -> bool:
    """Deprecated compatibility alias for :func:`is_matrixman_tensor`."""
    warnings.warn(
        "is_gm45_tensor() is deprecated; use is_matrixman_tensor() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return _is_matrixman_tensor(value)


def install_tensor_method() -> None:
    """Install the backend-neutral historical ``Tensor.gm45`` method."""
    _install_tensor_method()


def set_trace(enabled: bool = True) -> None:
    _require_opengl_frontend("set_trace").set_trace(enabled)


def debug_enabled() -> bool:
    backend = get_backend()
    return _require_opengl_frontend("debug inspection").debug_enabled() if backend.name == "opengl" else False


def profile_enabled() -> bool:
    backend = get_backend()
    return _require_opengl_frontend("profile inspection").profile_enabled() if backend.name == "opengl" else False


def profile_report() -> None:
    _require_opengl_frontend("profile reporting").profile_report()


def profile_reset() -> None:
    _require_opengl_frontend("profile reset").profile_reset()


def reset_unsupported_report() -> None:
    _require_opengl_frontend("unsupported-operation reporting").reset_unsupported_report()


def unsupported_report() -> dict:
    backend = get_backend()
    return _require_opengl_frontend("unsupported-operation reporting").unsupported_report() if backend.name == "opengl" else {}


def gpu_postprocess_detection(value):
    return _require_opengl_frontend("GPU detection postprocessing").gpu_postprocess_detection(value)
