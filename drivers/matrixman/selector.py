"""Backend capability probing and explicit MatrixMan backend selection."""

from __future__ import annotations

from dataclasses import dataclass, field

from .backend import Backend, active_backend, set_backend
from .backends.cuda.backend import CudaBackend
from .backends.cuda.gpumatrix import detect_device
from .config import config, profiling_enabled
from .privateuse import register_privateuse1_backend


@dataclass(frozen=True)
class BackendCapability:
    """Small, shared capability record used by probing and selection."""

    name: str
    available: bool
    enabled: bool
    implemented: bool
    probed: bool = True
    probe_reason: str | None = None
    backend: type[Backend] | None = None
    device: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


def probe_capabilities(requested: str = "") -> dict[str, BackendCapability]:
    """Probe all known backends once and return the ordered capability registry."""
    cuda_probe_reason = None
    try:
        _, cuda_info = detect_device()
    except Exception as exc:
        cuda_info = None
        cuda_probe_reason = str(exc) or exc.__class__.__name__

    # OpenGL probing currently creates its hidden context.  Do not import or
    # probe it on the CUDA path.  It is still probed when needed for fallback
    # selection or when explicitly requested.
    opengl_available = False
    opengl_backend = None
    opengl_info = {}
    opengl_probed = False
    opengl_probe_reason = None
    if requested == "opengl" or cuda_info is None:
        opengl_probed = True
        try:
            from .backends.opengl.backend import OpenGLBackend

            opengl_available = OpenGLBackend.probe()
            opengl_backend = OpenGLBackend
        except Exception as exc:
            opengl_available = False
            opengl_probe_reason = str(exc) or exc.__class__.__name__
        if opengl_available:
            # Telemetry/classification must not turn a valid context into an
            # unavailable backend. The context probe is the availability test.
            try:
                opengl_info = OpenGLBackend().device_info()
            except Exception as exc:
                opengl_info = {
                    "renderer": "unknown",
                    "vendor": "unknown",
                    "device_policy": f"renderer metadata unavailable: {exc}",
                }

    return {
        "cuda": BackendCapability(
            name="cuda",
            available=cuda_info is not None,
            enabled=True,
            implemented=True,
            probed=True,
            backend=CudaBackend,
            device=cuda_info["name"] if cuda_info else None,
            metadata={
                "compute_capability": cuda_info["compute_capability"],
                "driver_library": cuda_info["driver_library"],
            }
            if cuda_info
            else {},
            probe_reason=cuda_probe_reason,
        ),
        "opengl": BackendCapability(
            name="opengl",
            available=opengl_available,
            enabled=True,
            implemented=True,
            probed=opengl_probed,
            probe_reason=(
                "CUDA selected before OpenGL probe"
                if not opengl_probed
                else opengl_probe_reason
            ),
            backend=opengl_backend,
            device=opengl_info.get("renderer"),
            metadata={
                "vendor": opengl_info.get("vendor", ""),
                "device_policy": opengl_info.get("device_policy", ""),
                "gpu_preference": opengl_info.get("gpu_preference", ""),
                "gpu_preference_honored": opengl_info.get("gpu_preference_honored", ""),
                "gpu_preference_reason": opengl_info.get("gpu_preference_reason", ""),
            },
        ),
    }


def _status(capability: BackendCapability) -> str:
    if not capability.probed:
        return "skipped"
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
        if capability.metadata.get("driver_library"):
            print(f"    CUDA driver library: {capability.metadata['driver_library']}")
        if capability.metadata.get("vendor"):
            print(f"    vendor: {capability.metadata['vendor']}")
        if capability.metadata.get("device_policy"):
            print(f"    device policy: {capability.metadata['device_policy']}")
        if capability.metadata.get("gpu_preference"):
            print(f"    GPU preference: {capability.metadata['gpu_preference']}")
        if capability.metadata.get("gpu_preference_honored"):
            print(f"    GPU preference honored: {capability.metadata['gpu_preference_honored']}")
        if capability.metadata.get("gpu_preference_reason"):
            print(f"    GPU preference reason: {capability.metadata['gpu_preference_reason']}")
        if capability.probe_reason:
            print(f"    reason: {capability.probe_reason}")


def name_label(name: str) -> str:
    return {"opengl": "OpenGL", "cuda": "CUDA"}[name]


def select_backend() -> Backend:
    """Probe known backends, report the registry, and select an enabled backend."""
    requested = config.backend
    # ``auto`` is the public spelling for the pre-refactor unset state. It is
    # not a backend implementation and must therefore enter normal probing.
    if requested == "auto":
        requested = ""
    capabilities = probe_capabilities(requested)
    _print_probe(capabilities)

    if requested and requested not in {"cuda", "opengl"}:
        raise RuntimeError(
            f"Unknown MatrixMan backend {requested!r}; available names: cuda, opengl"
        )

    if requested:
        capability = capabilities[requested]
        if not capability.available or not capability.enabled or not capability.implemented:
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
    current = active_backend()
    if current is not None:
        if current.name != selected.name:
            raise RuntimeError(
                f"MatrixMan backend is already initialized as {current.name.upper()}; "
                "backend preference must be set before first MatrixMan device use"
            )
        return current
    register_privateuse1_backend()
    print(f"MatrixMan selected: {name_label(selected.name)}")
    backend = set_backend(selected.backend())
    profiler = __import__(
        f"drivers.matrixman.backends.{selected.name}.profiling",
        fromlist=["profiling"],
    )
    profiler.set_enabled(
        profiling_enabled(legacy_cuda=selected.name == "cuda")
    )
    return backend
