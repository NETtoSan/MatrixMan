"""Backend capability probing and explicit MatrixMan backend selection."""

from __future__ import annotations

from dataclasses import dataclass, field
import os

from .backend import Backend, set_backend
from .backends.cuda.backend import CudaBackend
from .backends.cuda.gpumatrix import detect_device
from .privateuse import register_privateuse1_backend


@dataclass(frozen=True)
class BackendCapability:
    """Small, shared capability record used by probing and selection."""

    name: str
    available: bool
    enabled: bool
    implemented: bool
    backend: type[Backend] | None = None
    device: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


def probe_capabilities(requested: str = "") -> dict[str, BackendCapability]:
    """Probe all known backends once and return the ordered capability registry."""
    try:
        _, cuda_info = detect_device()
    except Exception:
        cuda_info = None

    # OpenGL probing currently creates its hidden context.  Do not import or
    # probe it on the CUDA path.  It is still probed when needed for fallback
    # selection or when explicitly requested.
    opengl_available = False
    opengl_backend = None
    if requested == "opengl" or cuda_info is None:
        try:
            from .backends.opengl.backend import OpenGLBackend

            opengl_available = OpenGLBackend.probe()
            opengl_backend = OpenGLBackend
        except Exception:
            opengl_available = False

    return {
        "cuda": BackendCapability(
            name="cuda",
            available=cuda_info is not None,
            enabled=True,
            implemented=True,
            backend=CudaBackend,
            device=cuda_info["name"] if cuda_info else None,
            metadata={"compute_capability": cuda_info["compute_capability"]}
            if cuda_info
            else {},
        ),
        "opengl": BackendCapability(
            name="opengl",
            available=opengl_available,
            enabled=True,
            implemented=True,
            backend=opengl_backend,
        ),
        "opencl": BackendCapability(
            name="opencl", available=False, enabled=False, implemented=False
        ),
    }


def _status(capability: BackendCapability) -> str:
    if capability.name == "opencl" and not capability.implemented:
        return "not implemented"
    return "available" if capability.available else "unavailable"


def _print_probe(capabilities: dict[str, BackendCapability]) -> None:
    print("MatrixMan probe:")
    for name, capability in capabilities.items():
        status = _status(capability)
        suffix = " [disabled]" if capability.available and not capability.enabled else ""
        print(f"  {name_label(name)}: {status}{suffix}")
        if capability.device:
            print(f"    device: {capability.device}")
        if capability.metadata.get("compute_capability"):
            print(f"    compute capability: {capability.metadata['compute_capability']}")


def name_label(name: str) -> str:
    return {"opencl": "OpenCL", "opengl": "OpenGL", "cuda": "CUDA"}[name]


def select_backend() -> Backend:
    """Probe known backends, report the registry, and select an enabled backend."""
    requested = os.environ.get("MATRIXMAN_BACKEND", "").strip().lower()
    capabilities = probe_capabilities(requested)
    _print_probe(capabilities)

    if requested and requested not in {"cuda", "opengl", "opencl"}:
        raise RuntimeError(
            f"Unknown MatrixMan backend {requested!r}; available names: cuda, opengl"
        )

    if requested:
        capability = capabilities[requested]
        if not capability.available or not capability.enabled or not capability.implemented:
            if not capability.implemented:
                raise RuntimeError(
                    "MatrixMan backend 'opencl' was requested, but OpenCL is not implemented"
                )
            if not capability.enabled:
                raise RuntimeError(f"MatrixMan backend '{requested}' is disabled")
            raise RuntimeError(f"MatrixMan backend '{requested}' is unavailable")
        selected = capability
    else:
        selected = next(
            (
                capability
                for capability in capabilities.values()
                if capability.enabled and capability.implemented and capability.available
            ),
            None,
        )
        if selected is None:
            raise RuntimeError("No usable MatrixMan backend found")

    if selected.backend is None:
        raise RuntimeError(f"MatrixMan backend '{selected.name}' has no implementation")
    register_privateuse1_backend()
    print(f"MatrixMan selected: {name_label(selected.name)}")
    return set_backend(selected.backend())
