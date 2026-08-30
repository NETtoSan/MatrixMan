#!/usr/bin/env python3
"""Address-only packed RGBA shader diagnostics; no convolution is performed."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from drivers import matrixman as gm45
from drivers.matrixman import gpumatrix as gl
from drivers.matrixman.backends.opengl import operation_context, runtime, storage


def _probe_shader(numel: int, source_width: int, source_height: int, output_width: int, mode: str) -> bytes:
    if mode == "input":
        probe = "return read_packed(linear_index, INPUT_TEX_W, INPUT_TEX_H);"
    elif mode == "weight":
        probe = "return read_packed(linear_index, INPUT_TEX_W, INPUT_TEX_H);"
    elif mode == "output":
        probe = "return float(linear_index);"
    else:
        raise ValueError(mode)
    source = f"""
#version 120
uniform sampler2D input_tex;

float pick_component(vec4 value, int component)
{{
    if (component == 0) return value.r;
    if (component == 1) return value.g;
    if (component == 2) return value.b;
    return value.a;
}}

float read_packed(int linear_index, int tex_width, int tex_height)
{{
    int texel_index = linear_index / 4;
    int component = linear_index - texel_index * 4;
    int x = texel_index - (texel_index / tex_width) * tex_width;
    int y = texel_index / tex_width;
    vec2 uv = (vec2(float(x), float(y)) + vec2(0.5, 0.5)) /
              vec2(float(tex_width), float(tex_height));
    return pick_component(texture2D(input_tex, uv), component);
}}

float probe(int linear_index)
{{
    if (linear_index >= NUMEL) return 0.0;
    {probe}
}}

void main()
{{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * OUT_TEX_W + tex_x) * 4;
    gl_FragColor = vec4(probe(base), probe(base + 1), probe(base + 2), probe(base + 3));
}}
"""
    replacements = {
        "NUMEL": numel,
        "INPUT_TEX_W": source_width,
        "INPUT_TEX_H": source_height,
        "OUT_TEX_W": output_width,
    }
    for name, value in replacements.items():
        source = source.replace(name, str(value))
    return source.encode("ascii")


def _run_probe(tensor: gm45.Gm45Tensor, mode: str, shape: tuple[int, ...]) -> torch.Tensor:
    out_owner = operation_context.output_texture(shape)
    rt = runtime.runtime_required()
    program = gl.make_program(_probe_shader(
        storage.numel(shape), tensor._owner.layout.texture_width,
        tensor._owner.layout.texture_height, out_owner.layout.texture_width, mode,
    ))
    try:
        gl.glViewport(0, 0, out_owner.layout.texture_width, out_owner.layout.texture_height)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, rt.fbo.value)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, out_owner.texture, 0)
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("address diagnostic framebuffer incomplete")
        gl.glUseProgram(program)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tensor._owner.texture)
        location = gl.glGetUniformLocation(program, b"input_tex")
        gl.glUniform1i(location, 0)
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(-1.0, -1.0); gl.glVertex2f(1.0, -1.0)
        gl.glVertex2f(1.0, 1.0); gl.glVertex2f(-1.0, 1.0)
        gl.glEnd()
        error = gl.glGetError()
        if error:
            raise RuntimeError(f"OpenGL address probe error: 0x{error:04x}")
        result = gm45.Gm45Tensor._from_owner(out_owner, shape).cpu()
    finally:
        gl.glDeleteProgram(program)
    return result


def _packed_position(index: int, texture_width: int) -> tuple[int, int, int]:
    texel = index // 4
    return texel, texel % texture_width, texel // texture_width, index % 4


def _coords(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    coords = []
    for size in reversed(shape):
        coords.append(index % size)
        index //= size
    return tuple(reversed(coords))


def _report_mismatch(label: str, expected: torch.Tensor, actual: torch.Tensor, shape: tuple[int, ...], width: int) -> None:
    mismatch = (expected != actual) & ~(torch.isnan(expected) & torch.isnan(actual))
    if not mismatch.any():
        print(f"  {label}: PASS (all packed logical values match exactly)")
        return
    index = int(mismatch.flatten().nonzero()[0])
    expected_value = float(expected.flatten()[index])
    actual_value = float(actual.flatten()[index])
    expected_texel, expected_x, expected_y, expected_component = _packed_position(index, width)
    actual_index = int(round(actual_value))
    actual_texel, actual_x, actual_y, actual_component = _packed_position(actual_index, width)
    print(f"  {label}: FIRST MISMATCH logical_index={index} coords={_coords(index, shape)}")
    print(f"    expected value={expected_value} texel={expected_texel} xy=({expected_x},{expected_y}) component={expected_component}")
    print(f"    actual value={actual_value} interpreted_source_index={actual_index} texel={actual_texel} xy=({actual_x},{actual_y}) component={actual_component}")


def input_case(shape: tuple[int, ...]) -> None:
    count = math.prod(shape)
    cpu = torch.arange(count, dtype=torch.float32).reshape(shape)
    tensor = gm45.to_gm45(cpu)
    actual = _run_probe(tensor, "input", shape)
    print(f"INPUT shape={list(shape)} texture={tensor._owner.texture} atlas={tensor._owner.layout.texture_width}x{tensor._owner.layout.texture_height}")
    _report_mismatch("input lookup", cpu, actual, shape, tensor._owner.layout.texture_width)


def weight_case(shape: tuple[int, ...]) -> None:
    oc, ic, kh, kw = shape
    cpu = torch.empty(shape, dtype=torch.float32)
    for o in range(oc):
        for i in range(ic):
            for y in range(kh):
                for x in range(kw):
                    cpu[o, i, y, x] = o * 1000 + i * 10 + y * 2 + x
    flat = cpu.reshape(-1)
    tensor = gm45.to_gm45(flat)
    actual = _run_probe(tensor, "weight", flat.shape)
    print(f"WEIGHT logical shape={list(shape)} packed buffer={list(flat.shape)} texture={tensor._owner.texture} atlas={tensor._owner.layout.texture_width}x{tensor._owner.layout.texture_height}")
    _report_mismatch("weight lookup", flat, actual, shape, tensor._owner.layout.texture_width)


def output_case(shape: tuple[int, ...]) -> None:
    count = math.prod(shape)
    source = gm45.to_gm45(torch.zeros(shape, dtype=torch.float32))
    actual = _run_probe(source, "output", shape)
    expected = torch.arange(count, dtype=torch.float32).reshape(shape)
    print(f"OUTPUT shape={list(shape)} atlas={source._owner.layout.texture_width}x{source._owner.layout.texture_height}")
    _report_mismatch("output mapping", expected, actual, shape, source._owner.layout.texture_width)


def main() -> int:
    gm45.set_trace(False)
    print("GM45 packed-address diagnostic (no convolution)")
    for shape in ((1, 8, 32, 32), (1, 16, 32, 32), (1, 64, 64, 64), (1, 64, 128, 128), (1, 64, 160, 160)):
        input_case(shape)
    for shape in ((8, 8, 3, 3), (16, 16, 3, 3), (64, 64, 3, 3)):
        weight_case(shape)
    for shape in ((1, 8, 32, 32), (1, 64, 64, 64), (1, 64, 128, 128), (1, 64, 160, 160)):
        output_case(shape)
    gm45.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
