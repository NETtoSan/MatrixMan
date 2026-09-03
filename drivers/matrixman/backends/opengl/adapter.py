"""Best-effort OpenGL adapter preference before SDL context creation."""

from __future__ import annotations

import os


def preference_name(use_dgpu: bool) -> str:
    return "discrete" if use_dgpu else "integrated"


def request_preference(use_dgpu: bool) -> dict[str, str]:
    """Apply only platform mechanisms that exist; never overwrite user env."""
    preference = preference_name(use_dgpu)
    result = {
        "gpu_preference": preference,
        "gpu_preference_honored": "unknown",
        "gpu_preference_reason": "SDL does not expose OpenGL adapter selection",
    }

    if os.name == "nt":
        # The SDL2/WGL path used by MatrixMan has no adapter enumeration or
        # explicit adapter index. DXGI's GPU-preference API only orders DXGI
        # enumeration and cannot steer SDL_GL_CreateContext. Optimus and
        # PowerXpress exports must be exported by the executable itself; a
        # ctypes-loaded Python module cannot add them to python.exe.
        result["gpu_preference_reason"] = (
            "SDL/WGL exposes no programmatic OpenGL adapter selector; use Windows Graphics Settings "
            "or a vendor application profile"
        )
        return result

    # PRIME/DRI_PRIME is an existing user/system mechanism on Linux. Respect
    # it rather than changing the environment behind the user's back.
    if os.environ.get("DRI_PRIME"):
        result["gpu_preference_honored"] = "requested-via-DRI_PRIME"
        result["gpu_preference_reason"] = "using existing DRI_PRIME adapter selection"
    else:
        result["gpu_preference_reason"] = (
            "no internal Linux OpenGL adapter selector; preserve system default"
        )
    return result


def classify_renderer(vendor: str, renderer: str) -> str:
    """Classify only well-known renderer strings; unknown is safer than guessing."""
    text = f"{vendor} {renderer}".lower()
    if any(name in text for name in ("llvmpipe", "softpipe", "swrast")):
        return "software"
    if any(name in text for name in ("gt 7", "geforce", "quadro", "radeon rx", "arc a")):
        return "discrete"
    if any(name in text for name in ("intel", "uhd graphics", "iris", "radeon graphics")):
        return "integrated"
    return "unknown"


def finalize_preference(result: dict[str, str], vendor: str, renderer: str) -> dict[str, str]:
    kind = classify_renderer(vendor, renderer)
    requested = result["gpu_preference"]
    if kind == requested:
        result["gpu_preference_honored"] = "yes"
        result["gpu_preference_reason"] = "active renderer matches requested preference"
    elif kind != "unknown":
        result["gpu_preference_honored"] = "no"
        result["gpu_preference_reason"] = f"active renderer classified as {kind}"
    return result
