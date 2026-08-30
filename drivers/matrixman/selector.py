"""Explicit MatrixMan backend probing and selection."""

from __future__ import annotations

import os

from .backend import Backend, set_backend
from .backends.opengl.backend import OpenGLBackend


def _probe_results() -> list[tuple[str, str, type[Backend] | None]]:
    results = [("opencl", "not implemented", None)]
    try:
        available = OpenGLBackend.probe()
    except Exception:
        available = False
    results.append(("opengl", "available" if available else "unavailable", OpenGLBackend))
    return results


def select_backend() -> Backend:
    """Probe the known backends, select OpenGL when usable, and log once."""
    requested = os.environ.get("MATRIXMAN_BACKEND", "").strip().lower()
    if requested and requested not in {"opencl", "opengl"}:
        raise RuntimeError(
            f"Unknown MatrixMan backend {requested!r}; available names: opengl"
        )

    results = _probe_results()
    labels = {"opencl": "OpenCL", "opengl": "OpenGL"}
    summary = ", ".join(f"{labels[name]}={status}" for name, status, _ in results)
    print(f"MatrixMan probe: {summary}")

    statuses = {name: status for name, status, _ in results}
    if requested:
        if statuses[requested] != "available":
            if statuses[requested] == "not implemented":
                raise RuntimeError(
                    "MatrixMan backend 'opencl' was requested, but OpenCL is not implemented"
                )
            raise RuntimeError(f"MatrixMan backend '{requested}' is unavailable")
        selected_name = requested
    else:
        selected_name = next(
            (name for name, status, _ in results if status == "available"), None
        )
        if selected_name is None:
            raise RuntimeError("No usable MatrixMan backend found")

    selected_type = next(cls for name, _, cls in results if name == selected_name)
    print(f"MatrixMan selected: {labels[selected_name]}")
    return set_backend(selected_type())
