"""OpenGL/GLSL convolution implementation.

The backend module owns the tensor wrapper and the shared OpenGL runtime.  This
module owns only convolution execution and obtains those shared objects lazily
to avoid creating a second runtime or an import cycle.
"""

from __future__ import annotations

import ctypes
import math
import time

import torch

from . import gpumatrix as gm, profiling
from . import resources as _resources
from .storage import StorageLayout, packed_atlas_size
from ...config import config


# Conservative GM45-validated default; larger physical draws may be unstable.
CONV_PHYSICAL_TILE_LIMIT = 256
_tile_diagnostic_snapshots: list[dict] = []
_DEFAULT_BIAS = torch.zeros((1,), dtype=torch.float32)
_last_tile_geometry: list[dict] = []
_last_tile_output_texture: int | None = None
_last_dispatch_metadata: dict | None = None
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
        *tail,
    ) = params
    fused_silu = bool(tail[0]) if tail else False
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
    return {"acc / (1.0 + exp(-acc))" if fused_silu else "acc"};
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
    with profiling.conv_stage("program_cache_lookup"):
        cached = params in rt.conv_programs
    if not cached:
        b._trace(f"gm45.compile -> convolution GLSL fragment shader params={params}")
        with profiling.conv_stage("shader_source_generation"):
            source = _conv_shader_source(params)
        with profiling.conv_stage("shader_compile_link"):
            program = gm.make_program(source)
        rt.conv_programs[params] = program
        with profiling.conv_stage("uniform_location_setup"):
            rt.conv_uniforms[params] = (
                gm.glGetUniformLocation(program, b"input_tex"),
                gm.glGetUniformLocation(program, b"weight_tex"),
                gm.glGetUniformLocation(program, b"bias_tex"),
            )
    with profiling.conv_stage("program_retrieval"):
        input_loc, weight_loc, bias_loc = rt.conv_uniforms[params]
        return rt.conv_programs[params], input_loc, weight_loc, bias_loc


def _spatial_reuse_enabled() -> bool:
    return bool(config.convSpatialReuse)


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
        *_tail,
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
    with profiling.conv_stage("program_cache_lookup"):
        cached = params in rt.conv_spatial_programs
    if not cached:
        b._trace(f"gm45.compile -> spatial-reuse convolution GLSL shader params={params}")
        with profiling.conv_stage("shader_source_generation"):
            source = _conv_spatial_shader_source(params)
        with profiling.conv_stage("shader_compile_link"):
            program = gm.make_program(source)
        rt.conv_spatial_programs[params] = program
        with profiling.conv_stage("uniform_location_setup"):
            rt.conv_spatial_uniforms[params] = (
                gm.glGetUniformLocation(program, b"input_tex"),
                gm.glGetUniformLocation(program, b"weight_tex"),
                gm.glGetUniformLocation(program, b"bias_tex"),
            )
    with profiling.conv_stage("program_retrieval"):
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
    with profiling.conv_stage("viewport_state_setup"):
        gm.glViewport(0, 0, out_owner.layout.texture_width, out_owner.layout.texture_height)
    with profiling.conv_stage("fbo_output_binding"):
        gm.glBindFramebuffer(gm.GL_FRAMEBUFFER, rt.fbo.value)
        gm.glFramebufferTexture2D(gm.GL_FRAMEBUFFER, gm.GL_COLOR_ATTACHMENT0, gm.GL_TEXTURE_2D, out_owner.texture, 0)
    if gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER) != gm.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError("gm45 spatial-reuse convolution framebuffer incomplete")
    with profiling.conv_stage("glUseProgram"):
        gm.glUseProgram(program)
    with profiling.conv_stage("input_texture_binding"):
        gm.glActiveTexture(gm.GL_TEXTURE0)
        gm.glBindTexture(gm.GL_TEXTURE_2D, input_tensor._owner.texture)
    with profiling.conv_stage("uniform_setup"):
        gm.glUniform1i(input_loc, 0)
    with profiling.conv_stage("parameter_texture_binding"):
        gm.glActiveTexture(gm.GL_TEXTURE1)
        gm.glBindTexture(gm.GL_TEXTURE_2D, weight_owner.texture)
        gm.glActiveTexture(gm.GL_TEXTURE2)
        gm.glBindTexture(gm.GL_TEXTURE_2D, bias_owner.texture)
    with profiling.conv_stage("uniform_setup"):
        gm.glUniform1i(weight_loc, 1)
        gm.glUniform1i(bias_loc, 2)
    with profiling.conv_stage("draw_submission"):
        with profiling.gpu_timer("Conv2D spatial reuse"):
            gm.glBegin(gm.GL_QUADS)
            gm.glVertex2f(-1.0, -1.0); gm.glVertex2f(1.0, -1.0)
            gm.glVertex2f(1.0, 1.0); gm.glVertex2f(-1.0, 1.0); gm.glEnd()
    if (err := gm.glGetError()):
        raise RuntimeError(f"gm45 spatial-reuse convolution OpenGL error: 0x{err:04x}")
    return b.MatrixManTensor._from_owner(out_owner, (1, params[3], params[4], params[5]))


def _conv_tile_shader_source(params: tuple, tile_x: int, tile_y: int) -> bytes:
    source = _conv_shader_source(params).decode("ascii")
    out_tex_w = params[-2] if len(params) > 21 else params[-1]
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
    with profiling.conv_stage("shader_variant_key_construction"):
        key = _program_key(fragment_source)
    with profiling.conv_stage("program_cache_lookup"):
        cached = key in rt.conv_tile_programs
    if not cached:
        b._trace("gm45.compile -> tiled convolution GLSL fragment shader")
        with profiling.conv_stage("shader_compile_link"):
            program = gm.make_program(fragment_source)
        rt.conv_tile_programs[key] = program
        with profiling.conv_stage("uniform_location_setup"):
            rt.conv_tile_uniforms[key] = (
                gm.glGetUniformLocation(program, b"input_tex"),
                gm.glGetUniformLocation(program, b"weight_tex"),
                gm.glGetUniformLocation(program, b"bias_tex"),
            )
    with profiling.conv_stage("program_retrieval"):
        return (rt.conv_tile_programs[key], *rt.conv_tile_uniforms[key])


def _tile_copy_program(fragment_source: bytes) -> tuple[int, int]:
    b = _backend()
    rt = b._runtime_required()
    with profiling.conv_stage("shader_variant_key_construction"):
        key = _program_key(fragment_source)
    with profiling.conv_stage("program_cache_lookup"):
        cached = key in rt.tile_copy_programs
    if not cached:
        b._trace("gm45.compile -> tiled convolution copy GLSL fragment shader")
        with profiling.conv_stage("shader_compile_link"):
            program = gm.make_program(fragment_source)
        rt.tile_copy_programs[key] = program
        with profiling.conv_stage("uniform_location_setup"):
            rt.tile_copy_uniforms[key] = gm.glGetUniformLocation(program, b"tile_tex")
    with profiling.conv_stage("program_retrieval"):
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
    return config.tileSync


def _skip_pre_consolidation_sync() -> bool:
    return bool(config.skipPreConsolidationSync)


def _tile_limit() -> int:
    """Return the experimental limit without changing the safe default."""
    if config.tileLimit == "auto":
        return int(config.resolvedTileLimit)
    return int(config.tileLimit)


def _tile_limits() -> tuple[int, int]:
    limit = _tile_limit()
    # Independent dimensions are intentionally diagnostic-only.  Normal
    # execution always retains the square production limit.
    if not config.diagnosticRectTiles:
        return limit, limit
    width_raw = config.diagTileWidth
    height_raw = config.diagTileHeight
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
    if not config.diagnosticRectTiles:
        return normal
    order = config.diagTileOrder or "normal"
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


def diagnostic_tile_geometry() -> dict:
    """Return the most recent Conv physical dispatch geometry.

    This is intentionally diagnostic-only telemetry.  It does not influence
    dispatch policy or expose a new production execution path.
    """
    metadata = dict(_last_dispatch_metadata or {})
    metadata["tiles"] = [dict(item) for item in _last_tile_geometry]
    return metadata


def _render_convolution_tiled(input_tensor, out_owner, weight_owner, bias_owner, params):
    global _last_tile_output_texture, _last_dispatch_metadata
    b = _backend()
    sync_mode = _tile_sync_mode()
    full_w, full_h = out_owner.layout.texture_width, out_owner.layout.texture_height
    # Keep the compatibility probe's existing mutable backend limit visible.
    width_limit, height_limit = _tile_limits()
    tiles_x = math.ceil(full_w / width_limit)
    tiles_y = math.ceil(full_h / height_limit)
    tile_order = _tile_grid_order(tiles_x, tiles_y)
    gpu_metadata = _conv_gpu_metadata(input_tensor, out_owner, params, True, tiles_x * tiles_y)
    _last_dispatch_metadata = dict(gpu_metadata)
    _last_dispatch_metadata.update({"tile_limit": (width_limit, height_limit)})
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
            with profiling.conv_stage("shader_source_generation"):
                tile_source = _conv_tile_shader_source(params, origin_x, origin_y)
            program, input_loc, weight_loc, bias_loc = _conv_tile_program(tile_source)
            with profiling.conv_stage("viewport_state_setup"):
                gm.glViewport(0, 0, tile_w, tile_h)
            with profiling.conv_stage("fbo_output_binding"):
                gm.glBindFramebuffer(gm.GL_FRAMEBUFFER, rt.fbo.value)
                gm.glFramebufferTexture2D(gm.GL_FRAMEBUFFER, gm.GL_COLOR_ATTACHMENT0, gm.GL_TEXTURE_2D, tile.texture, 0)
            if gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER) != gm.GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError("gm45 tiled convolution framebuffer incomplete")
            with profiling.conv_stage("glUseProgram"):
                gm.glUseProgram(program)
            with profiling.conv_stage("input_texture_binding"):
                gm.glActiveTexture(gm.GL_TEXTURE0)
                gm.glBindTexture(gm.GL_TEXTURE_2D, input_tensor._owner.texture)
            with profiling.conv_stage("uniform_setup"):
                gm.glUniform1i(input_loc, 0)
            with profiling.conv_stage("parameter_texture_binding"):
                for unit, texture in ((gm.GL_TEXTURE1, weight_owner.texture), (gm.GL_TEXTURE2, bias_owner.texture)):
                    gm.glActiveTexture(unit)
                    gm.glBindTexture(gm.GL_TEXTURE_2D, texture)
            with profiling.conv_stage("uniform_setup"):
                gm.glUniform1i(weight_loc, 1)
                gm.glUniform1i(bias_loc, 2)
            with profiling.conv_stage("draw_submission"):
                with profiling.gpu_timer("Conv2D", gpu_metadata):
                    gm.glBegin(gm.GL_QUADS)
                    gm.glVertex2f(-1.0, -1.0); gm.glVertex2f(1.0, -1.0)
                    gm.glVertex2f(1.0, 1.0); gm.glVertex2f(-1.0, 1.0); gm.glEnd()
            if (err := gm.glGetError()):
                raise RuntimeError(f"gm45 tiled convolution OpenGL error: 0x{err:04x}")
            if sync_mode == "per_tile":
                with profiling.conv_stage("synchronization_wait"):
                    gm.glFinish()
                if b._profile_enabled:
                    b._profile_counters["tiled_per_tile_sync_calls"] += 1
            elif sync_mode == "flush":
                with profiling.conv_stage("synchronization_wait"):
                    gm.glFlush()
        # Diagnostic readback is deliberately delayed until every production
        # tile render has completed. It cannot alter inter-tile scheduling.
        if config.diagnosticTiles:
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
        # Per-tile mode already completed every producer before reaching this
        # point.  A second barrier before the copy pass is therefore redundant.
        # Keep the old barrier available for callers that explicitly disable
        # the skip, and retain end-mode's documented single barrier.
        if sync_mode != "flush":
            should_finish_before_consolidation = (
                not _skip_pre_consolidation_sync()
                and sync_mode in {"per_tile", "end"}
            )
            if should_finish_before_consolidation:
                if b._profile_enabled:
                    b._profile_counters["pre_consolidation_sync_calls"] += 1
                with profiling.conv_stage("synchronization_wait"):
                    gm.glFinish()
            elif b._profile_enabled:
                b._profile_counters["pre_consolidation_sync_skips"] += 1
            b._trace(
                "gm45 tiled convolution -> "
                f"{'kept' if should_finish_before_consolidation else 'elided'} "
                "pre-consolidation synchronization"
            )
        tiles_by_grid = {
            geometry["grid"]: tile
            for geometry, tile in zip(_last_tile_geometry, tile_owners)
        }
        consolidation_started = time.perf_counter()
        _consolidate_tiles(tile_owners, _last_tile_geometry, out_owner, full_w, full_h, width_limit, height_limit, rt)
        # The consolidation draw is ordered after the tile draws by the same
        # GL context.  Its output is either consumed by a later queued draw or
        # synchronized once by the eventual CPU readback; no glFinish is
        # required here.  This also permits the scratch tiles to return to the
        # pool without serializing the whole pipeline.
        if b._profile_enabled:
            b._profile_counters["consolidation_sync_elisions"] += 1
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
            with profiling.conv_stage("shader_source_generation"):
                tile_source = _tile_copy_shader_source(tile_w, tile_h, origin_x, origin_y)
            program, tile_loc = _tile_copy_program(tile_source)
            if b._profile_enabled:
                b._profile_counters["consolidation_draw_calls"] += 1
            with profiling.conv_stage("viewport_state_setup"):
                gm.glViewport(0, 0, full_w, full_h)
            gm.gl.glEnable(_GL_SCISSOR_TEST)
            gm.gl.glScissor(origin_x, origin_y, tile_w, tile_h)
            with profiling.conv_stage("fbo_output_binding"):
                gm.glBindFramebuffer(gm.GL_FRAMEBUFFER, rt.fbo.value)
                gm.glFramebufferTexture2D(gm.GL_FRAMEBUFFER, gm.GL_COLOR_ATTACHMENT0, gm.GL_TEXTURE_2D, out_owner.texture, 0)
            if gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER) != gm.GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError("gm45 tiled convolution output framebuffer incomplete")
            with profiling.conv_stage("glUseProgram"):
                gm.glUseProgram(program)
            with profiling.conv_stage("input_texture_binding"):
                gm.glActiveTexture(gm.GL_TEXTURE0)
                gm.glBindTexture(gm.GL_TEXTURE_2D, tile.texture)
            with profiling.conv_stage("uniform_setup"):
                gm.glUniform1i(tile_loc, 0)
            with profiling.conv_stage("draw_submission"):
                with profiling.gpu_timer("consolidation"):
                    gm.glBegin(gm.GL_QUADS)
                    gm.glVertex2f(-1.0, -1.0); gm.glVertex2f(1.0, -1.0)
                    gm.glVertex2f(1.0, 1.0); gm.glVertex2f(-1.0, 1.0); gm.glEnd()
            gm.gl.glDisable(_GL_SCISSOR_TEST)
            if (err := gm.glGetError()):
                raise RuntimeError(f"gm45 tiled convolution copy OpenGL error: 0x{err:04x}")


def _validate_convolution(args, b):
    """Validate Conv2D arguments and allocate the packed output texture."""
    input_tensor, weight_tensor, bias_tensor = args[0], args[1], args[2]
    stride = _as_pair(args[3], "stride")
    padding = _as_pair(args[4], "padding")
    dilation = _as_pair(args[5], "dilation")
    transposed = bool(args[6])
    output_padding = _as_pair(args[7], "output_padding")
    groups = int(args[8])
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
    return (input_tensor, weight_tensor, bias_tensor, stride, padding, out_shape,
            out_owner, (in_c, in_h, in_w, out_c, out_h, out_w, kernel_h, kernel_w,
                         groups))


def _upload_convolution_parameters(weight_tensor, bias_tensor, b):
    """Upload/cache CPU Conv2D parameters as packed OpenGL textures."""
    upload_started = time.perf_counter()
    weight_owner = _resources.cached_parameter_texture(weight_tensor, "weight", operation="Conv2D")
    if bias_tensor is not None:
        bias_owner = _resources.cached_parameter_texture(bias_tensor, "bias", operation="Conv2D")
    else:
        bias_owner = _resources.cached_parameter_texture(_DEFAULT_BIAS, "bias", operation="Conv2D")
    if b._profile_enabled:
        b._profile_conv["parameter_upload"] += time.perf_counter() - upload_started
    return weight_owner, bias_owner


def _convolution_params(input_tensor, out_owner, dimensions, stride, padding,
                        bias_tensor, weight_owner, bias_owner, *, fused_silu=False) -> tuple:
    """Build the immutable shader signature after resource resolution."""
    in_c, in_h, in_w, out_c, out_h, out_w, kernel_h, kernel_w, groups = dimensions
    return (
        in_c, in_h, in_w, out_c, out_h, out_w, kernel_h, kernel_w,
        stride[0], stride[1], padding[0], padding[1], bias_tensor is not None,
        groups, input_tensor._storage_offset,
        input_tensor._owner.layout.texture_width,
        input_tensor._owner.layout.texture_height,
        weight_owner.layout.texture_width, weight_owner.layout.texture_height,
        bias_owner.layout.texture_width, out_owner.layout.texture_width,
        bool(fused_silu),
    )


def _render_prepared_convolution(input_tensor, weight_tensor, bias_tensor, out_owner,
                                 weight_owner, bias_owner, dimensions, stride,
                                 padding, out_shape, b, *, fused_silu=False,
                                 allow_spatial_reuse=True):
    params = _convolution_params(
        input_tensor, out_owner, dimensions, stride, padding,
        bias_tensor, weight_owner, bias_owner, fused_silu=fused_silu,
    )
    tile_limit = _tile_limit()
    if allow_spatial_reuse and not fused_silu and _spatial_reuse_enabled() and _conv_spatial_reuse_supported(
        input_tensor, out_owner, params, tile_limit
    ):
        return _render_convolution_spatial(
            input_tensor, out_owner, weight_owner, bias_owner, params
        )
    if out_owner.layout.texture_width > tile_limit or out_owner.layout.texture_height > tile_limit:
        return _render_convolution_tiled(
            input_tensor, out_owner, weight_owner, bias_owner, params
        )
    return _render_convolution_direct(
        input_tensor, weight_tensor, bias_tensor, out_owner,
        weight_owner, bias_owner, params, out_shape, b,
    )


def _pending_descriptor(input_tensor, weight_tensor, bias_tensor, out_owner,
                        dimensions, stride, padding, out_shape) -> dict:
    return {
        "input_tensor": input_tensor,
        "weight_tensor": weight_tensor,
        "bias_tensor": bias_tensor,
        "out_owner": out_owner,
        "dimensions": dimensions,
        "stride": stride,
        "padding": padding,
        "out_shape": out_shape,
    }


def materialize_pending(tensor, *, override_parameters=None):
    """Render a deferred convolution into its already reserved output owner."""
    descriptor = getattr(tensor, "_pending_convolution", None)
    if descriptor is None:
        return tensor
    b = _backend()
    weight_tensor = descriptor["weight_tensor"]
    bias_tensor = descriptor["bias_tensor"]
    if override_parameters is None:
        override_parameters = descriptor.get("override_parameters")
    if override_parameters is None:
        render_weight, render_bias = weight_tensor, bias_tensor
    else:
        render_weight, render_bias = override_parameters
    weight_owner, bias_owner = _upload_convolution_parameters(render_weight, render_bias, b)
    result = _render_prepared_convolution(
        descriptor["input_tensor"], render_weight, render_bias,
        descriptor["out_owner"], weight_owner, bias_owner,
        descriptor["dimensions"], descriptor["stride"], descriptor["padding"],
        descriptor["out_shape"], b,
        fused_silu=bool(descriptor.get("fused_silu", False)),
    )
    tensor._owner = result._owner
    tensor._shape = result._shape
    tensor._storage_offset = result._storage_offset
    tensor._logical_strides = result._logical_strides
    del tensor._pending_convolution
    return tensor


def _parameter_version(value) -> int:
    return int(getattr(value, "_version", 0)) if value is not None else -1


def try_fuse_batch_norm(input_tensor, weight, bias, running_mean, running_var, eps):
    """Fold an immediately following inference BatchNorm into a pending Conv2D."""
    if not config.preparedExecution:
        return None
    descriptor = getattr(input_tensor, "_pending_convolution", None)
    if descriptor is None:
        return None
    conv_weight = descriptor["weight_tensor"]
    conv_bias = descriptor["bias_tensor"]
    key = (
        id(conv_weight), _parameter_version(conv_weight),
        id(conv_bias), _parameter_version(conv_bias),
        id(weight), _parameter_version(weight),
        id(bias), _parameter_version(bias),
        id(running_mean), _parameter_version(running_mean),
        id(running_var), _parameter_version(running_var), float(eps),
    )
    rt = _backend()._runtime_required()
    cached = rt.fused_parameter_cache.get(key)
    if cached is None:
        scale = weight / torch.sqrt(running_var + float(eps))
        folded_weight = conv_weight * scale.reshape(-1, 1, 1, 1)
        base_bias = conv_bias
        if base_bias is None:
            base_bias = torch.zeros_like(running_mean)
        folded_bias = (base_bias - running_mean) * scale + bias
        cached = (folded_weight.contiguous(), folded_bias.contiguous())
        rt.fused_parameter_cache[key] = cached
    descriptor["override_parameters"] = cached
    descriptor["fused_batch_norm"] = True
    if profiling.enabled:
        profiling.counters["fused_batch_norm_calls"] += 1
    return input_tensor


def try_fuse_silu(input_tensor):
    """Fuse an immediately following in-place SiLU into folded Conv+BN."""
    if not config.preparedExecution:
        return None
    descriptor = getattr(input_tensor, "_pending_convolution", None)
    if descriptor is None or not descriptor.get("fused_batch_norm"):
        return None
    descriptor["fused_silu"] = True
    if profiling.enabled:
        profiling.counters["fused_silu_calls"] += 1
    return materialize_pending(input_tensor)


def _render_convolution_direct(input_tensor, weight_tensor, bias_tensor, out_owner, weight_owner, bias_owner, params, out_shape, b):
    """Render a non-tiled Conv2D into the final packed output texture."""
    global _last_dispatch_metadata, _last_tile_geometry
    _last_tile_geometry.clear()
    _last_dispatch_metadata = _conv_gpu_metadata(input_tensor, out_owner, params, False, 1)
    _last_dispatch_metadata.update({"tile_limit": (out_owner.layout.texture_width, out_owner.layout.texture_height)})
    _last_tile_geometry.append({
        "grid": (0, 0), "origin": (0, 0),
        "width": out_owner.layout.texture_width,
        "height": out_owner.layout.texture_height,
        "texture_size": (out_owner.layout.texture_width, out_owner.layout.texture_height),
        "texture": out_owner.texture,
    })
    b._kernel_log(f"Conv2D {b._shape_text(input_tensor.shape)} -> {b._shape_text(out_shape)}")
    program, input_loc, weight_loc, bias_loc = _conv_program(params)
    rt = b._runtime_required()
    b._trace("gm45.kernel -> convolution shader:\n" f"  input texture #{input_tensor._owner.texture} shape={list(input_tensor.shape)}\n" f"  weight texture #{weight_owner.texture} shape={list(weight_tensor.shape)}\n" f"  bias texture #{bias_owner.texture if bias_tensor is not None else 'none'}\n" f"  -> output texture #{out_owner.texture} shape={list(out_shape)}")
    with profiling.conv_stage("viewport_state_setup"):
        gm.glViewport(0, 0, out_owner.layout.texture_width, out_owner.layout.texture_height)
    with profiling.conv_stage("fbo_output_binding"):
        gm.glBindFramebuffer(gm.GL_FRAMEBUFFER, rt.fbo.value)
        gm.glFramebufferTexture2D(gm.GL_FRAMEBUFFER, gm.GL_COLOR_ATTACHMENT0, gm.GL_TEXTURE_2D, out_owner.texture, 0)
    status = gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER)
    if status != gm.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(f"gm45 convolution framebuffer incomplete: 0x{status:04x}")
    with profiling.conv_stage("glUseProgram"):
        gm.glUseProgram(program)
    with profiling.conv_stage("input_texture_binding"):
        gm.glActiveTexture(gm.GL_TEXTURE0)
        gm.glBindTexture(gm.GL_TEXTURE_2D, input_tensor._owner.texture)
    with profiling.conv_stage("uniform_setup"):
        gm.glUniform1i(input_loc, 0)
    with profiling.conv_stage("parameter_texture_binding"):
        gm.glActiveTexture(gm.GL_TEXTURE1)
        gm.glBindTexture(gm.GL_TEXTURE_2D, weight_owner.texture)
        gm.glActiveTexture(gm.GL_TEXTURE2)
        gm.glBindTexture(gm.GL_TEXTURE_2D, bias_owner.texture)
    with profiling.conv_stage("uniform_setup"):
        gm.glUniform1i(weight_loc, 1)
        gm.glUniform1i(bias_loc, 2)
    render_started = time.perf_counter()
    gpu_metadata = _conv_gpu_metadata(input_tensor, out_owner, params, False, 0)
    with profiling.gpu_timer("Conv2D", gpu_metadata):
        with profiling.conv_stage("draw_submission"):
            gm.glBegin(gm.GL_QUADS)
            gm.glVertex2f(-1.0, -1.0); gm.glVertex2f(1.0, -1.0); gm.glVertex2f(1.0, 1.0); gm.glVertex2f(-1.0, 1.0); gm.glEnd()
    if b._profile_enabled:
        b._profile_conv["tile_render"] += time.perf_counter() - render_started
    b._trace(f"gm45.opengl -> submitted Conv2D fullscreen quad, output texture #{out_owner.texture}")
    if (err := gm.glGetError()):
        raise RuntimeError(f"gm45 OpenGL error after convolution: 0x{err:04x}")
    return b.MatrixManTensor._from_owner(out_owner, out_shape)


def execute(args):
    b = _backend()
    conv_started = time.perf_counter()

    (input_tensor, weight_tensor, bias_tensor,
     stride, padding, out_shape, out_owner, dimensions) = _validate_convolution(args, b)

    if b._profile_enabled:
        b._profile_conv["prepare"] += time.perf_counter() - conv_started

    if config.preparedExecution:
        # Reserve the output now so shape/alias semantics remain visible, but
        # delay the draw until the consumer is known.  BatchNorm can then fold
        # into this convolution; every other consumer materializes normally.
        output = b.MatrixManTensor._from_owner(out_owner, out_shape)
        if profiling.enabled:
            profiling.counters["prepared_convolution_deferrals"] += 1
        output._pending_convolution = _pending_descriptor(
            input_tensor, weight_tensor, bias_tensor, out_owner,
            dimensions, stride, padding, out_shape,
        )
        return output

    weight_owner, bias_owner = _upload_convolution_parameters(weight_tensor, bias_tensor, b)
    return _render_prepared_convolution(
        input_tensor, weight_tensor, bias_tensor, out_owner,
        weight_owner, bias_owner, dimensions, stride, padding, out_shape, b,
    )


def execute_prepared(args):
    """Execute a prepared Conv2D without deferred prepared-execution routing."""
    b = _backend()
    (input_tensor, weight_tensor, bias_tensor,
     stride, padding, out_shape, out_owner, dimensions) = _validate_convolution(args, b)
    weight_owner, bias_owner = _upload_convolution_parameters(weight_tensor, bias_tensor, b)
    return _render_prepared_convolution(
        input_tensor, weight_tensor, bias_tensor, out_owner,
        weight_owner, bias_owner, dimensions, stride, padding, out_shape, b,
        allow_spatial_reuse=False,
    )
