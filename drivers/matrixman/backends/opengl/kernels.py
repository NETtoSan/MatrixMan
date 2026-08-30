"""Shared OpenGL kernel/program infrastructure."""

from __future__ import annotations

from . import gpumatrix as gm
from . import gpu_stress
from . import diagnostics
from . import runtime


def glsl_float(value: float) -> str:
    """Format a GLSL float literal exactly as the backend historically did."""
    text = f"{float(value):.10g}"
    if "." not in text and "e" not in text and "E" not in text:
        text += ".0"
    return text


def _trace(message: str) -> None:
    diagnostics.trace(message)


def program(kind: str, n: int) -> tuple[int, int, int]:
    """Return the cached add/matmul program and sampler locations."""
    rt = runtime.runtime_required()
    if kind == "add":
        programs = rt.add_programs
        uniforms = rt.add_uniforms
        source = gpu_stress.shader_source(gpu_stress.ADD_SHADER, n)
    elif kind == "matmul":
        programs = rt.matmul_programs
        uniforms = rt.matmul_uniforms
        source = gpu_stress.shader_source(gpu_stress.MUL_SHADER, n)
    else:
        raise AssertionError(kind)
    if n not in programs:
        _trace(f"gm45.compile -> {kind} GLSL fragment shader for {n}x{n}")
        program_id = gm.make_program(source)
        programs[n] = program_id
        uniforms[n] = (
            gm.glGetUniformLocation(program_id, b"left_tex"),
            gm.glGetUniformLocation(program_id, b"right_tex"),
        )
    left_loc, right_loc = uniforms[n]
    return programs[n], left_loc, right_loc
