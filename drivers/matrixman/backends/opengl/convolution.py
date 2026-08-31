"""OpenGL/GLSL convolution implementation.

The backend module owns the tensor wrapper and the shared OpenGL runtime.  This
module owns only convolution execution and obtains those shared objects lazily
to avoid creating a second runtime or an import cycle.
"""

from __future__ import annotations

import ctypes
import math
import os
import time

import numpy as np
import torch

from . import gpumatrix as gm, profiling
from . import resources as _resources
from .storage import StorageLayout, packed_atlas_size


# Conservative GM45-validated default; larger physical draws may be unstable.
CONV_PHYSICAL_TILE_LIMIT = 256
_tile_diagnostic_snapshots: list[dict] = []
_last_tile_geometry: list[dict] = []
_last_tile_output_texture: int | None = None
_GL_SCISSOR_TEST = 0x0C11
gm.gl.glScissor.restype = None
gm.gl.glScissor.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
gm.gl.glEnable.restype = None
gm.gl.glEnable.argtypes = [ctypes.c_uint]
gm.gl.glDisable.restype = None
gm.gl.glDisable.argtypes = [ctypes.c_uint]
gm.gl.glCopyTexSubImage2D.restype = None
gm.gl.glCopyTexSubImage2D.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]


def _backend():
    # Import only when a kernel is executed: the MatrixMan backend façade
    # imports this module on demand.
    from . import backend as gm45_backend
    return gm45_backend


def _conv_shader_source(params: tuple) -> bytes:
    (
        in_c, in_h, in_w, out_c, out_h, out_w, kernel_h, kernel_w,
        stride_h, stride_w, pad_h, pad_w, has_bias, groups, input_offset,
        input_tex_w, input_tex_h, weight_tex_w, weight_tex_h, bias_tex_w,
        out_tex_w,
    ) = params
    bias_expr = "read_bias(oc)" if has_bias else "0.0"
    source = f"""
#version 120
uniform sampler2D input_tex;
uniform sampler2D weight_tex;
uniform sampler2D bias_tex;

float pick_component(vec4 value, int component)
{{
    if (component == 0) return value.r;
    if (component == 1) return value.g;
    if (component == 2) return value.b;
    return value.a;
}}

float read_input(int ic, int iy, int ix)
{{
    int linear_index = INPUT_OFFSET + ((ic * IN_H) + iy) * IN_W + ix;
    int texel = linear_index / 4;
    int component = linear_index - texel * 4;
    int x = texel - (texel / INPUT_TEX_W) * INPUT_TEX_W;
    int y = texel / INPUT_TEX_W;
    vec2 uv = (vec2(float(x), float(y)) + vec2(0.5, 0.5)) / vec2(float(INPUT_TEX_W), float(INPUT_TEX_H));
    return pick_component(texture2D(input_tex, uv), component);
}}

float read_weight(int oc, int ic, int ky, int kx)
{{
    int linear_index = (((oc * IN_C_PER_GROUP + ic) * K_H + ky) * K_W) + kx;
    int texel = linear_index / 4;
    int component = linear_index - texel * 4;
    int x = texel - (texel / WEIGHT_TEX_W) * WEIGHT_TEX_W;
    int y = texel / WEIGHT_TEX_W;
    vec2 uv = (vec2(float(x), float(y)) + vec2(0.5, 0.5)) / vec2(float(WEIGHT_TEX_W), float(WEIGHT_TEX_H));
    return pick_component(texture2D(weight_tex, uv), component);
}}

float read_bias(int oc)
{{
    int linear_index = oc;
    int texel = linear_index / 4;
    int component = linear_index - texel * 4;
    int x = texel - (texel / BIAS_TEX_W) * BIAS_TEX_W;
    int y = texel / BIAS_TEX_W;
    vec2 uv = (vec2(float(x), float(y)) + vec2(0.5, 0.5)) / vec2(float(BIAS_TEX_W), float(BIAS_TEX_H));
    return pick_component(texture2D(bias_tex, uv), component);
}}

float compute_output(int out_index)
{{
    if (out_index >= OUT_NUMEL) return 0.0;
    int ox = out_index - (out_index / OUT_W) * OUT_W;
    int tmp0 = out_index / OUT_W;
    int oy = tmp0 - (tmp0 / OUT_H) * OUT_H;
    int oc = tmp0 / OUT_H;
    float acc = {bias_expr};

    int group = oc / OUT_C_PER_GROUP;
    int input_channel_start = group * IN_C_PER_GROUP;
    for (int ic = 0; ic < IN_C_PER_GROUP; ++ic) {{
        for (int ky = 0; ky < K_H; ++ky) {{
            int iy = oy * STRIDE_H + ky - PAD_H;
            if (iy >= 0 && iy < IN_H) {{
                for (int kx = 0; kx < K_W; ++kx) {{
                    int ix = ox * STRIDE_W + kx - PAD_W;
                    if (ix >= 0 && ix < IN_W) {{
                        acc += read_input(input_channel_start + ic, iy, ix) * read_weight(oc, ic, ky, kx);
                    }}
                }}
            }}
        }}
    }}
    return acc;
}}

void main()
{{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * OUT_TEX_W + tex_x) * 4;
    gl_FragColor = vec4(
        compute_output(base),
        compute_output(base + 1),
        compute_output(base + 2),
        compute_output(base + 3)
    );
}}
"""
    replacements = {
        "IN_C": in_c, "IN_H": in_h, "IN_W": in_w,
        "OUT_C": out_c, "OUT_H": out_h, "OUT_W": out_w,
        "OUT_NUMEL": out_c * out_h * out_w, "GROUPS": groups,
        "IN_C_PER_GROUP": in_c // groups, "OUT_C_PER_GROUP": out_c // groups,
        "K_H": kernel_h, "K_W": kernel_w,
        "STRIDE_H": stride_h, "STRIDE_W": stride_w,
        "PAD_H": pad_h, "PAD_W": pad_w, "INPUT_OFFSET": input_offset,
        "INPUT_TEX_W": input_tex_w, "INPUT_TEX_H": input_tex_h,
        "WEIGHT_TEX_W": weight_tex_w, "WEIGHT_TEX_H": weight_tex_h,
        "BIAS_TEX_W": bias_tex_w,
        "BIAS_TEX_H": max(1, packed_atlas_size(max(out_c, 1))[1]),
        "OUT_TEX_W": out_tex_w,
    }
    for name, value in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        source = source.replace(name, str(value))
    return source.encode("ascii")


def _conv_program(params: tuple) -> tuple[int, int, int, int]:
    b = _backend()
    rt = b._runtime_required()
    if params not in rt.conv_programs:
        b._trace(f"gm45.compile -> convolution GLSL fragment shader params={params}")
        program = gm.make_program(_conv_shader_source(params))
        rt.conv_programs[params] = program
        rt.conv_uniforms[params] = (
            gm.glGetUniformLocation(program, b"input_tex"),
            gm.glGetUniformLocation(program, b"weight_tex"),
            gm.glGetUniformLocation(program, b"bias_tex"),
        )
    input_loc, weight_loc, bias_loc = rt.conv_uniforms[params]
    return rt.conv_programs[params], input_loc, weight_loc, bias_loc


def _spatial_reuse_enabled() -> bool:
    value = os.environ.get("MATRIXMAN_CONV_SPATIAL_REUSE", "0")
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _conv_spatial_reuse_supported(input_tensor, out_owner, params, tile_limit: int) -> bool:
    return (
        input_tensor._storage_offset == 0
        and int(params[6]) == 3 and int(params[7]) == 3
        and tuple(params[8:10]) == (1, 1)
        and tuple(params[10:12]) == (1, 1)
        and int(params[13]) == 1
        and int(params[5]) % 4 == 0
        and out_owner.layout.texture_width <= tile_limit
        and out_owner.layout.texture_height <= tile_limit
        and out_owner.layout.numel == out_owner.layout.texture_width * out_owner.layout.texture_height * 4
    )


def _conv_spatial_shader_source(params: tuple) -> bytes:
    (
        in_c, in_h, in_w, out_c, out_h, out_w, _kernel_h, _kernel_w,
        _stride_h, _stride_w, _pad_h, _pad_w, has_bias, _groups, input_offset,
        input_tex_w, input_tex_h, weight_tex_w, weight_tex_h, bias_tex_w,
        out_tex_w,
    ) = params
    bias_expr = "read_bias(oc)" if has_bias else "0.0"
    source = f"""
#version 120
uniform sampler2D input_tex;
uniform sampler2D weight_tex;
uniform sampler2D bias_tex;

float pick_component(vec4 value, int component)
{{
    if (component == 0) return value.r;
    if (component == 1) return value.g;
    if (component == 2) return value.b;
    return value.a;
}}

vec4 read_texel(sampler2D tex, int texel, int tex_width, int tex_height)
{{
    int x = texel - (texel / tex_width) * tex_width;
    int y = texel / tex_width;
    vec2 uv = (vec2(float(x), float(y)) + vec2(0.5, 0.5)) /
              vec2(float(tex_width), float(tex_height));
    return texture2D(tex, uv);
}}

float span_value(vec4 first, vec4 second, vec4 third, int first_component, int offset)
{{
    int component = first_component + offset;
    if (component < 4) return pick_component(first, component);
    if (component < 8) return pick_component(second, component - 4);
    return pick_component(third, component - 8);
}}

float read_scalar(sampler2D tex, int linear_index, int tex_width, int tex_height)
{{
    int texel = linear_index / 4;
    int component = linear_index - texel * 4;
    return pick_component(read_texel(tex, texel, tex_width, tex_height), component);
}}

float compute_output_scalar(int out_index)
{{
    if (out_index >= OUT_NUMEL) return 0.0;
    int ox = out_index - (out_index / OUT_W) * OUT_W;
    int tmp0 = out_index / OUT_W;
    int oy = tmp0 - (tmp0 / OUT_H) * OUT_H;
    int oc = tmp0 / OUT_H;
    float acc = {bias_expr};
    for (int ic = 0; ic < IN_C; ++ic) {{
        for (int ky = 0; ky < 3; ++ky) {{
            int iy = oy + ky - 1;
            if (iy >= 0 && iy < IN_H) {{
                for (int kx = 0; kx < 3; ++kx) {{
                    int ix = ox + kx - 1;
                    if (ix >= 0 && ix < IN_W) {{
                        int input_index = INPUT_OFFSET + ((ic * IN_H + iy) * IN_W) + ix;
                        int weight_index = (((oc * IN_C + ic) * 3 + ky) * 3) + kx;
                        acc += read_scalar(input_tex, input_index, INPUT_TEX_W, INPUT_TEX_H) *
                               read_scalar(weight_tex, weight_index, WEIGHT_TEX_W, WEIGHT_TEX_H);
                    }}
                }}
            }}
        }}
    }}
    return acc;
}}

float read_bias(int oc)
{{
    int texel = oc / 4;
    int component = oc - texel * 4;
    return pick_component(read_texel(bias_tex, texel, BIAS_TEX_W, BIAS_TEX_H), component);
}}

vec4 compute_outputs(int out_index)
{{
    int ox = out_index - (out_index / OUT_W) * OUT_W;
    int tmp0 = out_index / OUT_W;
    int oy = tmp0 - (tmp0 / OUT_H) * OUT_H;
    int oc = tmp0 / OUT_H;
    // The shared window is only valid when all four output lanes stay in one
    // logical row and its requested origin is non-negative.  In particular,
    // do not clamp ox-1 and retain the old lane offsets: the x=0 group needs
    // the baseline's explicit out-of-bounds-zero semantics.
    if (ox < 1 || ox + 3 >= OUT_W) {{
        return vec4(
            compute_output_scalar(out_index),
            compute_output_scalar(out_index + 1),
            compute_output_scalar(out_index + 2),
            compute_output_scalar(out_index + 3)
        );
    }}
    vec4 acc = vec4({bias_expr});
    int first_x = ox - 1;
    for (int ic = 0; ic < IN_C; ++ic) {{
        for (int ky = 0; ky < 3; ++ky) {{
            int iy = oy + ky - 1;
            if (iy >= 0 && iy < IN_H) {{
                int input_base = INPUT_OFFSET + ((ic * IN_H + iy) * IN_W) + first_x;
                int input_texel = input_base / 4;
                int input_component = input_base - input_texel * 4;
                vec4 input_first = read_texel(input_tex, input_texel, INPUT_TEX_W, INPUT_TEX_H);
                vec4 input_second = read_texel(input_tex, input_texel + 1, INPUT_TEX_W, INPUT_TEX_H);
                vec4 input_third = read_texel(input_tex, input_texel + 2, INPUT_TEX_W, INPUT_TEX_H);
                int weight_index = (((oc * IN_C + ic) * 3 + ky) * 3);
                int weight_texel = weight_index / 4;
                int weight_component = weight_index - weight_texel * 4;
                vec4 weight_first = read_texel(weight_tex, weight_texel, WEIGHT_TEX_W, WEIGHT_TEX_H);
                vec4 weight_second = read_texel(weight_tex, weight_texel + 1, WEIGHT_TEX_W, WEIGHT_TEX_H);
                float w0 = span_value(weight_first, weight_second, weight_second, weight_component, 0);
                float w1 = span_value(weight_first, weight_second, weight_second, weight_component, 1);
                float w2 = span_value(weight_first, weight_second, weight_second, weight_component, 2);
                float v0 = span_value(input_first, input_second, input_third, input_component, 0);
                float v1 = span_value(input_first, input_second, input_third, input_component, 1);
                float v2 = span_value(input_first, input_second, input_third, input_component, 2);
                float v3 = span_value(input_first, input_second, input_third, input_component, 3);
                float v4 = span_value(input_first, input_second, input_third, input_component, 4);
                float v5 = span_value(input_first, input_second, input_third, input_component, 5);
                if (ox >= 1) acc.x += v0 * w0;
                if (ox >= 0 && ox + 0 < IN_W) acc.x += v1 * w1;
                if (ox + 1 < IN_W) acc.x += v2 * w2;
                if (ox >= 0 && ox + 0 < IN_W) acc.y += v1 * w0;
                if (ox + 1 < IN_W) acc.y += v2 * w1;
                if (ox + 2 < IN_W) acc.y += v3 * w2;
                if (ox + 1 < IN_W) acc.z += v2 * w0;
                if (ox + 2 < IN_W) acc.z += v3 * w1;
                if (ox + 3 < IN_W) acc.z += v4 * w2;
                if (ox + 2 < IN_W) acc.w += v3 * w0;
                if (ox + 3 < IN_W) acc.w += v4 * w1;
                if (ox + 4 < IN_W) acc.w += v5 * w2;
            }}
        }}
    }}
    return acc;
}}

void main()
{{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * OUT_TEX_W + tex_x) * 4;
    gl_FragColor = compute_outputs(base);
}}
"""
    replacements = {
        "IN_C": in_c, "IN_H": in_h, "IN_W": in_w,
        "OUT_NUMEL": out_c * out_h * out_w, "OUT_H": out_h, "OUT_W": out_w, "OUT_TEX_W": out_tex_w,
        "INPUT_OFFSET": input_offset, "INPUT_TEX_W": input_tex_w, "INPUT_TEX_H": input_tex_h,
        "WEIGHT_TEX_W": weight_tex_w, "WEIGHT_TEX_H": weight_tex_h,
        "BIAS_TEX_W": bias_tex_w, "BIAS_TEX_H": max(1, packed_atlas_size(max(out_c, 1))[1]),
    }
    for name, value in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        source = source.replace(name, str(value))
    return source.encode("ascii")


def _conv_spatial_program(params: tuple) -> tuple[int, int, int, int]:
    b = _backend()
    rt = b._runtime_required()
    if params not in rt.conv_spatial_programs:
        b._trace(f"gm45.compile -> spatial-reuse convolution GLSL shader params={params}")
        program = gm.make_program(_conv_spatial_shader_source(params))
        rt.conv_spatial_programs[params] = program
        rt.conv_spatial_uniforms[params] = (
            gm.glGetUniformLocation(program, b"input_tex"),
            gm.glGetUniformLocation(program, b"weight_tex"),
            gm.glGetUniformLocation(program, b"bias_tex"),
        )
    return rt.conv_spatial_programs[params], *rt.conv_spatial_uniforms[params]


def _render_convolution_spatial(input_tensor, out_owner, weight_owner, bias_owner, params):
    b = _backend()
    b._kernel_log(f"Conv2D RGBA spatial reuse {b._shape_text(input_tensor.shape)} -> {b._shape_text((1, params[3], params[4], params[5]))}")
    program, input_loc, weight_loc, bias_loc = _conv_spatial_program(params)
    rt = b._runtime_required()
    b._trace(
        "gm45.kernel -> spatial-neighbor-reuse convolution shader:\n"
        f"  input texture #{input_tensor._owner.texture} shape={list(input_tensor.shape)}\n"
        f"  weight texture #{weight_owner.texture} shape=[{params[3]},{params[0]},{params[6]},{params[7]}]\n"
        f"  -> output texture #{out_owner.texture} shape={[1, params[3], params[4], params[5]]}"
    )
    gm.glViewport(0, 0, out_owner.layout.texture_width, out_owner.layout.texture_height)
    gm.glBindFramebuffer(gm.GL_FRAMEBUFFER, rt.fbo.value)
    gm.glFramebufferTexture2D(gm.GL_FRAMEBUFFER, gm.GL_COLOR_ATTACHMENT0, gm.GL_TEXTURE_2D, out_owner.texture, 0)
    if gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER) != gm.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError("gm45 spatial-reuse convolution framebuffer incomplete")
    gm.glUseProgram(program)
    gm.glActiveTexture(gm.GL_TEXTURE0); gm.glBindTexture(gm.GL_TEXTURE_2D, input_tensor._owner.texture); gm.glUniform1i(input_loc, 0)
    gm.glActiveTexture(gm.GL_TEXTURE1); gm.glBindTexture(gm.GL_TEXTURE_2D, weight_owner.texture); gm.glUniform1i(weight_loc, 1)
    gm.glActiveTexture(gm.GL_TEXTURE2); gm.glBindTexture(gm.GL_TEXTURE_2D, bias_owner.texture); gm.glUniform1i(bias_loc, 2)
    with profiling.gpu_timer("Conv2D spatial reuse"):
        gm.glBegin(gm.GL_QUADS)
        gm.glVertex2f(-1.0, -1.0); gm.glVertex2f(1.0, -1.0)
        gm.glVertex2f(1.0, 1.0); gm.glVertex2f(-1.0, 1.0); gm.glEnd()
    if (err := gm.glGetError()):
        raise RuntimeError(f"gm45 spatial-reuse convolution OpenGL error: 0x{err:04x}")
    return b.MatrixManTensor._from_owner(out_owner, (1, params[3], params[4], params[5]))


def _conv_tile_shader_source(params: tuple, tile_x: int, tile_y: int) -> bytes:
    source = _conv_shader_source(params).decode("ascii")
    out_tex_w = params[-1]
    old = f"int base = (tex_y * {out_tex_w} + tex_x) * 4;"
    new = f"int base = ((tex_y + {tile_y}) * {out_tex_w} + tex_x + {tile_x}) * 4;"
    if old not in source:
        raise RuntimeError("gm45 tiled convolution could not locate output address expression")
    return source.replace(old, new).encode("ascii")


def _program_key(fragment_source: bytes) -> tuple[bytes, bytes]:
    """Identify a linked program by the exact shader sources it uses."""
    return gm.VERTEX_SHADER, fragment_source


def _conv_tile_program(fragment_source: bytes) -> tuple[int, int, int, int]:
    b = _backend()
    rt = b._runtime_required()
    key = _program_key(fragment_source)
    if key not in rt.conv_tile_programs:
        b._trace("gm45.compile -> tiled convolution GLSL fragment shader")
        program = gm.make_program(fragment_source)
        rt.conv_tile_programs[key] = program
        rt.conv_tile_uniforms[key] = (
            gm.glGetUniformLocation(program, b"input_tex"),
            gm.glGetUniformLocation(program, b"weight_tex"),
            gm.glGetUniformLocation(program, b"bias_tex"),
        )
    return (rt.conv_tile_programs[key], *rt.conv_tile_uniforms[key])


def _tile_copy_program(fragment_source: bytes) -> tuple[int, int]:
    b = _backend()
    rt = b._runtime_required()
    key = _program_key(fragment_source)
    if key not in rt.tile_copy_programs:
        b._trace("gm45.compile -> tiled convolution copy GLSL fragment shader")
        program = gm.make_program(fragment_source)
        rt.tile_copy_programs[key] = program
        rt.tile_copy_uniforms[key] = gm.glGetUniformLocation(program, b"tile_tex")
    return rt.tile_copy_programs[key], rt.tile_copy_uniforms[key]


def _tile_copy_shader_source(tile_width: int, tile_height: int, origin_x: int, origin_y: int) -> bytes:
    return f"""
#version 120
uniform sampler2D tile_tex;
void main() {{
    int x = int(floor(gl_FragCoord.x)) - {origin_x};
    int y = int(floor(gl_FragCoord.y)) - {origin_y};
    vec2 uv = (vec2(float(x), float(y)) + vec2(0.5, 0.5)) /
              vec2(float({tile_width}), float({tile_height}));
    gl_FragColor = texture2D(tile_tex, uv);
}}
""".encode("ascii")


def _new_physical_packed_owner(width: int, height: int):
    b = _backend()
    texture = _resources.acquire_scratch_texture(width, height)
    return b._TextureOwner(texture, StorageLayout("packed_rgba", width, height, width * height * 4))


def _as_pair(value, name: str) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise RuntimeError(f"gm45 convolution expects {name} as int or pair")


def _tile_sync_mode() -> str:
    mode = os.environ.get("MATRIXMAN_TILE_SYNC", "per_tile").strip().lower() or "per_tile"
    if mode not in {"per_tile", "end", "flush", "none"}:
        raise RuntimeError(
            "MATRIXMAN_TILE_SYNC must be one of: per_tile, end, flush, none"
        )
    return mode


def _skip_pre_consolidation_sync() -> bool:
    value = os.environ.get("MATRIXMAN_SKIP_PRE_CONSOLIDATION_SYNC", "0")
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _tile_limit() -> int:
    """Return the experimental limit without changing the safe default."""
    b = _backend()
    raw = os.environ.get("MATRIXMAN_TILE_LIMIT")
    if raw is None or not raw.strip():
        return int(getattr(b, "CONV_PHYSICAL_TILE_LIMIT", CONV_PHYSICAL_TILE_LIMIT))
    try:
        limit = int(raw)
    except ValueError as exc:
        raise RuntimeError("MATRIXMAN_TILE_LIMIT must be a positive integer") from exc
    if limit <= 0:
        raise RuntimeError("MATRIXMAN_TILE_LIMIT must be a positive integer")
    return limit


def _tile_limits() -> tuple[int, int]:
    limit = _tile_limit()
    # Independent dimensions are intentionally diagnostic-only.  Normal
    # execution always retains the square production limit.
    if os.environ.get("MATRIXMAN_DIAGNOSTIC_RECT_TILES") != "1":
        return limit, limit
    width_raw = os.environ.get("MATRIXMAN_DIAG_TILE_WIDTH")
    height_raw = os.environ.get("MATRIXMAN_DIAG_TILE_HEIGHT")
    if width_raw is None and height_raw is None:
        return limit, limit
    try:
        width = int(width_raw if width_raw is not None else limit)
        height = int(height_raw if height_raw is not None else limit)
    except ValueError as exc:
        raise RuntimeError(
            "MATRIXMAN_DIAG_TILE_WIDTH and MATRIXMAN_DIAG_TILE_HEIGHT must be positive integers"
        ) from exc
    if width <= 0 or height <= 0:
        raise RuntimeError(
            "MATRIXMAN_DIAG_TILE_WIDTH and MATRIXMAN_DIAG_TILE_HEIGHT must be positive integers"
        )
    return width, height


def _tile_grid_order(tiles_x: int, tiles_y: int) -> list[tuple[int, int]]:
    normal = [(tile_x, tile_y) for tile_y in range(tiles_y) for tile_x in range(tiles_x)]
    if os.environ.get("MATRIXMAN_DIAGNOSTIC_RECT_TILES") != "1":
        return normal
    order = os.environ.get("MATRIXMAN_DIAG_TILE_ORDER", "normal").strip().lower() or "normal"
    if order == "normal":
        return normal
    if order == "reverse":
        return list(reversed(normal))
    column = [(tile_x, tile_y) for tile_x in range(tiles_x) for tile_y in range(tiles_y)]
    if order == "column":
        return column
    if order == "reverse_column":
        return list(reversed(column))
    raise RuntimeError(
        "MATRIXMAN_DIAG_TILE_ORDER must be one of: normal, reverse, column, reverse_column"
    )


def _conv_gpu_metadata(input_tensor, out_owner, params, tiled: bool, physical_tile_count: int) -> dict:
    in_c, _in_h, _in_w, out_c, out_h, out_w = (int(value) for value in params[:6])
    kernel = (int(params[6]), int(params[7]))
    groups = int(params[13])
    return {
        "input_shape": tuple(int(value) for value in input_tensor.shape),
        "weight_shape": (out_c, in_c // groups, kernel[0], kernel[1]),
        "output_shape": (1, out_c, out_h, out_w),
        "kernel": kernel,
        "stride": (int(params[8]), int(params[9])),
        "padding": (int(params[10]), int(params[11])),
        "dilation": (1, 1),
        "groups": groups,
        "logical_output_elements": out_c * out_h * out_w,
        "atlas": (int(out_owner.layout.texture_width), int(out_owner.layout.texture_height)),
        "macs_per_output": (in_c // groups) * kernel[0] * kernel[1],
        "texture_samples_per_output": (in_c // groups) * kernel[0] * kernel[1],
        "tiled": bool(tiled),
        "physical_tile_count": int(physical_tile_count),
    }


def _render_convolution_tiled(input_tensor, out_owner, weight_owner, bias_owner, params):
    global _last_tile_output_texture
    b = _backend()
    sync_mode = _tile_sync_mode()
    full_w, full_h = out_owner.layout.texture_width, out_owner.layout.texture_height
    # Keep the compatibility probe's existing mutable backend limit visible.
    width_limit, height_limit = _tile_limits()
    tiles_x = math.ceil(full_w / width_limit)
    tiles_y = math.ceil(full_h / height_limit)
    tile_order = _tile_grid_order(tiles_x, tiles_y)
    gpu_metadata = _conv_gpu_metadata(input_tensor, out_owner, params, True, tiles_x * tiles_y)
    if b._profile_enabled:
        b._profile_counters["tiled_conv_calls"] += 1
        b._profile_counters["tiled_conv_tiles"] += tiles_x * tiles_y
        b._profile_counters["tiled_conv_max_tile_width"] = max(b._profile_counters["tiled_conv_max_tile_width"], width_limit)
        b._profile_counters["tiled_conv_max_tile_height"] = max(b._profile_counters["tiled_conv_max_tile_height"], height_limit)
    b._kernel_log(f"Tiled Conv2D {b._shape_text(input_tensor.shape)} -> {b._shape_text((1, params[3], params[4], params[5]))} tiles={tiles_x}x{tiles_y}")
    b._trace("gm45.conv -> tiled dispatch\n" f"  logical atlas: {full_w}x{full_h}\n" f"  tile limit: {width_limit}x{height_limit}\n" f"  physical tiles: {tiles_x}x{tiles_y} = {tiles_x * tiles_y}")
    rt = b._runtime_required()
    _tile_diagnostic_snapshots.clear()
    _last_tile_geometry.clear()
    _last_tile_output_texture = out_owner.texture
    tile_owners = []
    try:
        tile_render_started = time.perf_counter()
        finish_before_tiles = b._profile_counters["glFinish_seconds"]
        flush_before_tiles = b._profile_counters["glFlush_seconds"]
        for render_sequence_index, (tile_x, tile_y) in enumerate(tile_order):
            origin_y = tile_y * height_limit
            tile_h = min(height_limit, full_h - origin_y)
            origin_x = tile_x * width_limit
            tile_w = min(width_limit, full_w - origin_x)
            _last_tile_geometry.append({
                "render_sequence_index": render_sequence_index,
                "grid": (tile_x, tile_y),
                "origin": (origin_x, origin_y),
                "width": tile_w,
                "height": tile_h,
                "logical_region": (origin_x, origin_y, tile_w, tile_h),
                "texture_size": (tile_w, tile_h),
            })
            tile = _new_physical_packed_owner(tile_w, tile_h)
            tile_owners.append(tile)
            _last_tile_geometry[-1]["texture"] = tile.texture
            if b._profile_enabled:
                b._profile_counters["tiled_draw_calls"] += 1
            program, input_loc, weight_loc, bias_loc = _conv_tile_program(
                _conv_tile_shader_source(params, origin_x, origin_y)
            )
            gm.glViewport(0, 0, tile_w, tile_h)
            gm.glBindFramebuffer(gm.GL_FRAMEBUFFER, rt.fbo.value)
            gm.glFramebufferTexture2D(gm.GL_FRAMEBUFFER, gm.GL_COLOR_ATTACHMENT0, gm.GL_TEXTURE_2D, tile.texture, 0)
            if gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER) != gm.GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError("gm45 tiled convolution framebuffer incomplete")
            gm.glUseProgram(program)
            for unit, texture, uniform_location in ((gm.GL_TEXTURE0, input_tensor._owner.texture, input_loc), (gm.GL_TEXTURE1, weight_owner.texture, weight_loc), (gm.GL_TEXTURE2, bias_owner.texture, bias_loc)):
                gm.glActiveTexture(unit)
                gm.glBindTexture(gm.GL_TEXTURE_2D, texture)
                gm.glUniform1i(uniform_location, unit - gm.GL_TEXTURE0)
            with profiling.gpu_timer("Conv2D", gpu_metadata):
                gm.glBegin(gm.GL_QUADS)
                gm.glVertex2f(-1.0, -1.0); gm.glVertex2f(1.0, -1.0)
                gm.glVertex2f(1.0, 1.0); gm.glVertex2f(-1.0, 1.0); gm.glEnd()
            if (err := gm.glGetError()):
                raise RuntimeError(f"gm45 tiled convolution OpenGL error: 0x{err:04x}")
            if sync_mode == "per_tile":
                gm.glFinish()
            elif sync_mode == "flush":
                gm.glFlush()
        # Diagnostic readback is deliberately delayed until every production
        # tile render has completed. It cannot alter inter-tile scheduling.
        if os.environ.get("MATRIXMAN_DIAGNOSTIC_TILES") == "1":
            for index, tile in enumerate(tile_owners):
                geometry = _last_tile_geometry[index]
                diagnostic = b._read_texture(
                    tile, (1, 1, geometry["height"], geometry["width"] * 4)
                )
                _tile_diagnostic_snapshots.append({
                    "tile_index": index,
                    "grid": geometry["grid"],
                    "origin_x": geometry["origin"][0],
                    "origin_y": geometry["origin"][1],
                    "width": geometry["width"],
                    "height": geometry["height"],
                    "texture": tile.texture,
                    "data": diagnostic,
                })
        if b._profile_enabled:
            b._profile_conv["tile_render"] += time.perf_counter() - tile_render_started
            b._profile_conv["sync"] += (
                b._profile_counters["glFinish_seconds"] - finish_before_tiles
                + b._profile_counters["glFlush_seconds"] - flush_before_tiles
            )
        # This is the ordering point before consolidation.  It is retained for
        # the default/end/none modes.  In flush mode, avoid immediately
        # following the final glFlush with a glFinish as requested by the
        # experiment; the existing post-consolidation barrier remains.
        if sync_mode != "flush":
            if _skip_pre_consolidation_sync():
                if b._profile_enabled:
                    b._profile_counters["pre_consolidation_sync_skips"] += 1
                b._trace("gm45 tiled convolution -> skipped experimental pre-consolidation glFinish")
            else:
                if b._profile_enabled:
                    b._profile_counters["pre_consolidation_sync_calls"] += 1
                gm.glFinish()
        tiles_by_grid = {
            geometry["grid"]: tile
            for geometry, tile in zip(_last_tile_geometry, tile_owners)
        }
        consolidation_started = time.perf_counter()
        _consolidate_tiles(tile_owners, _last_tile_geometry, out_owner, full_w, full_h, width_limit, height_limit, rt)
        gm.glFinish()
        if b._profile_enabled:
            b._profile_conv["consolidation"] += time.perf_counter() - consolidation_started
        return b.MatrixManTensor._from_owner(out_owner, tuple(int(v) for v in (1, params[3], params[4], params[5])))
    finally:
        for tile in tile_owners:
            _resources.release_scratch_texture(tile)


def _consolidate_tiles(tile_owners, geometries, out_owner, full_w, full_h, width_limit, height_limit, rt):
    """Run the existing GPU tile-copy pass for production and diagnostics."""
    b = _backend()
    tiles_x = math.ceil(full_w / width_limit)
    tiles_y = math.ceil(full_h / height_limit)
    tiles_by_grid = {geometry["grid"]: tile for geometry, tile in zip(geometries, tile_owners)}
    for tile_y in range(tiles_y):
        origin_y = tile_y * height_limit
        tile_h = min(height_limit, full_h - origin_y)
        for tile_x in range(tiles_x):
            origin_x = tile_x * width_limit
            tile_w = min(width_limit, full_w - origin_x)
            tile = tiles_by_grid[(tile_x, tile_y)]
            program, tile_loc = _tile_copy_program(
                _tile_copy_shader_source(tile_w, tile_h, origin_x, origin_y)
            )
            if b._profile_enabled:
                b._profile_counters["consolidation_draw_calls"] += 1
            gm.glViewport(0, 0, full_w, full_h)
            gm.gl.glEnable(_GL_SCISSOR_TEST)
            gm.gl.glScissor(origin_x, origin_y, tile_w, tile_h)
            gm.glBindFramebuffer(gm.GL_FRAMEBUFFER, rt.fbo.value)
            gm.glFramebufferTexture2D(gm.GL_FRAMEBUFFER, gm.GL_COLOR_ATTACHMENT0, gm.GL_TEXTURE_2D, out_owner.texture, 0)
            if gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER) != gm.GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError("gm45 tiled convolution output framebuffer incomplete")
            gm.glUseProgram(program)
            gm.glActiveTexture(gm.GL_TEXTURE0)
            gm.glBindTexture(gm.GL_TEXTURE_2D, tile.texture)
            gm.glUniform1i(tile_loc, 0)
            with profiling.gpu_timer("consolidation"):
                gm.glBegin(gm.GL_QUADS)
                gm.glVertex2f(-1.0, -1.0); gm.glVertex2f(1.0, -1.0)
                gm.glVertex2f(1.0, 1.0); gm.glVertex2f(-1.0, 1.0); gm.glEnd()
            gm.gl.glDisable(_GL_SCISSOR_TEST)
            if (err := gm.glGetError()):
                raise RuntimeError(f"gm45 tiled convolution copy OpenGL error: 0x{err:04x}")


def execute(args):
    b = _backend()
    tile_limit = _tile_limit()
    conv_started = time.perf_counter()
    input_tensor, weight_tensor, bias_tensor = args[0], args[1], args[2]
    stride, padding, dilation = _as_pair(args[3], "stride"), _as_pair(args[4], "padding"), _as_pair(args[5], "dilation")
    transposed, output_padding, groups = bool(args[6]), _as_pair(args[7], "output_padding"), int(args[8])
    if not isinstance(input_tensor, b.MatrixManTensor):
        raise RuntimeError("gm45 convolution requires input to be a MatrixManTensor")
    if input_tensor._owner.layout.kind != "packed_rgba":
        raise RuntimeError("gm45 convolution requires packed_rgba input storage")
    b._require_contiguous_logical(input_tensor, "convolution")
    if tuple(input_tensor.shape)[0] != 1 or len(input_tensor.shape) != 4:
        raise RuntimeError("gm45 convolution supports only batch-1 NCHW 4D input")
    if not isinstance(weight_tensor, torch.Tensor) or weight_tensor.device.type != "cpu":
        raise RuntimeError("gm45 convolution currently expects CPU weight tensor for upload")
    if weight_tensor.dtype != torch.float32 or not weight_tensor.is_contiguous():
        raise RuntimeError("gm45 convolution weights must be contiguous float32")
    if bias_tensor is not None and (not isinstance(bias_tensor, torch.Tensor) or bias_tensor.device.type != "cpu" or bias_tensor.dtype != torch.float32 or not bias_tensor.is_contiguous()):
        raise RuntimeError("gm45 convolution bias must be a contiguous CPU float32 tensor or None")
    if transposed:
        raise RuntimeError("gm45 convolution does not support transposed convolution")
    if output_padding != (0, 0):
        raise RuntimeError("gm45 convolution requires output_padding=(0,0)")
    if dilation != (1, 1):
        raise RuntimeError("gm45 convolution currently supports dilation=(1,1) only")
    _, in_c, in_h, in_w = (int(v) for v in input_tensor.shape)
    out_c, weight_in_c, kernel_h, kernel_w = (int(v) for v in weight_tensor.shape)
    if groups < 1 or in_c % groups != 0 or out_c % groups != 0:
        raise RuntimeError("gm45 grouped convolution requires positive groups dividing Cin and Cout")
    input_channels_per_group = in_c // groups
    grouped = groups > 1 and weight_in_c == input_channels_per_group
    if groups != 1 and not grouped:
        raise RuntimeError("gm45 grouped convolution weight shape does not match Cin/groups")
    if groups == 1 and weight_in_c != in_c:
        raise RuntimeError("gm45 convolution input channels do not match weight channels")
    if groups > 1 and (kernel_h, kernel_w) != (3, 3):
        raise RuntimeError("gm45 grouped convolution currently supports only 3x3 kernels")
    if kernel_h not in {1, 3} or kernel_w not in {1, 3}:
        raise RuntimeError("gm45 convolution currently supports only 1x1 and 3x3 kernels")
    if stride not in {(1, 1), (2, 2)}:
        raise RuntimeError("gm45 convolution currently supports stride 1 or 2")
    if padding not in {(0, 0), (1, 1)}:
        raise RuntimeError("gm45 convolution currently supports padding 0 or 1")
    if groups > 1 and (stride, padding) not in {((2, 2), (1, 1)), ((1, 1), (1, 1))}:
        raise RuntimeError("gm45 grouped convolution supports only stride 1/2 with padding 1")
    if bias_tensor is not None and tuple(bias_tensor.shape) != (out_c,):
        raise RuntimeError("gm45 convolution bias shape must be [out_channels]")
    out_h = (in_h + 2 * padding[0] - kernel_h) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - kernel_w) // stride[1] + 1
    out_shape = (1, out_c, out_h, out_w)
    out_owner = b._new_empty_packed_texture(out_shape)
    if b._profile_enabled:
        b._profile_conv["prepare"] += time.perf_counter() - conv_started
    upload_started = time.perf_counter()
    weight_owner = _resources.cached_parameter_texture(weight_tensor, "weight")
    if bias_tensor is not None:
        bias_owner = _resources.cached_parameter_texture(bias_tensor, "bias")
    else:
        bias_owner = _resources.upload_raw_packed_array(np.zeros((1,), dtype=np.float32), "bias")
    if b._profile_enabled:
        b._profile_conv["parameter_upload"] += time.perf_counter() - upload_started
    params = (in_c, in_h, in_w, out_c, out_h, out_w, kernel_h, kernel_w, stride[0], stride[1], padding[0], padding[1], bias_tensor is not None, groups, input_tensor._storage_offset, input_tensor._owner.layout.texture_width, input_tensor._owner.layout.texture_height, weight_owner.layout.texture_width, weight_owner.layout.texture_height, bias_owner.layout.texture_width, out_owner.layout.texture_width)
    if _spatial_reuse_enabled() and _conv_spatial_reuse_supported(input_tensor, out_owner, params, tile_limit):
        return _render_convolution_spatial(input_tensor, out_owner, weight_owner, bias_owner, params)
    if out_owner.layout.texture_width > tile_limit or out_owner.layout.texture_height > tile_limit:
        return _render_convolution_tiled(input_tensor, out_owner, weight_owner, bias_owner, params)
    b._kernel_log(f"Conv2D {b._shape_text(input_tensor.shape)} -> {b._shape_text(out_shape)}")
    shader_setup_started = time.perf_counter()
    program, input_loc, weight_loc, bias_loc = _conv_program(params)
    rt = b._runtime_required()
    b._trace("gm45.kernel -> convolution shader:\n" f"  input texture #{input_tensor._owner.texture} shape={list(input_tensor.shape)}\n" f"  weight texture #{weight_owner.texture} shape={list(weight_tensor.shape)}\n" f"  bias texture #{bias_owner.texture if bias_tensor is not None else 'none'}\n" f"  -> output texture #{out_owner.texture} shape={list(out_shape)}")
    gm.glViewport(0, 0, out_owner.layout.texture_width, out_owner.layout.texture_height)
    gm.glBindFramebuffer(gm.GL_FRAMEBUFFER, rt.fbo.value)
    gm.glFramebufferTexture2D(gm.GL_FRAMEBUFFER, gm.GL_COLOR_ATTACHMENT0, gm.GL_TEXTURE_2D, out_owner.texture, 0)
    status = gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER)
    if status != gm.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(f"gm45 convolution framebuffer incomplete: 0x{status:04x}")
    gm.glUseProgram(program)
    gm.glActiveTexture(gm.GL_TEXTURE0); gm.glBindTexture(gm.GL_TEXTURE_2D, input_tensor._owner.texture); gm.glUniform1i(input_loc, 0)
    gm.glActiveTexture(gm.GL_TEXTURE1); gm.glBindTexture(gm.GL_TEXTURE_2D, weight_owner.texture); gm.glUniform1i(weight_loc, 1)
    gm.glActiveTexture(gm.GL_TEXTURE2); gm.glBindTexture(gm.GL_TEXTURE_2D, bias_owner.texture); gm.glUniform1i(bias_loc, 2)
    if b._profile_enabled:
        b._profile_conv["shader_setup"] += time.perf_counter() - shader_setup_started
    render_started = time.perf_counter()
    gpu_metadata = _conv_gpu_metadata(input_tensor, out_owner, params, False, 0)
    with profiling.gpu_timer("Conv2D", gpu_metadata):
        gm.glBegin(gm.GL_QUADS)
        gm.glVertex2f(-1.0, -1.0); gm.glVertex2f(1.0, -1.0); gm.glVertex2f(1.0, 1.0); gm.glVertex2f(-1.0, 1.0); gm.glEnd()
    if b._profile_enabled:
        b._profile_conv["tile_render"] += time.perf_counter() - render_started
    b._trace(f"gm45.opengl -> submitted Conv2D fullscreen quad, output texture #{out_owner.texture}")
    if (err := gm.glGetError()):
        raise RuntimeError(f"gm45 OpenGL error after convolution: 0x{err:04x}")
    return b.MatrixManTensor._from_owner(out_owner, out_shape)
