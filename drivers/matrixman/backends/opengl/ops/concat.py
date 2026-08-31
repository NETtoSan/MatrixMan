"""OpenGL concat, stack, and fill operations."""

from __future__ import annotations

import torch

from .. import diagnostics, gpumatrix as gm, operation_context, profiling
from ..kernels import glsl_float as _glsl_float
from ..storage import contiguous_strides, numel
from ....tensor import MatrixManTensor

def _stack_program(params: tuple) -> tuple[int, tuple[int, ...]]:
    rt = operation_context.gl_runtime()
    if params not in rt.stack_programs:
        diagnostics.trace(f"gm45.compile -> stack GLSL fragment shader params={params}")
        program = gm.make_program(_stack_shader_source(params))
        rt.stack_programs[params] = program
        rt.stack_uniforms[params] = (
            gm.glGetUniformLocation(program, b"input0_tex"),
            gm.glGetUniformLocation(program, b"input1_tex"),
        )
    return rt.stack_programs[params], rt.stack_uniforms[params]

def _stack_shader_source(params: tuple) -> bytes:
    rows, cols, offsets, strides0, strides1, tex_widths, tex_heights, out_tex_w = params
    source = """
#version 120
uniform sampler2D input0_tex;
uniform sampler2D input1_tex;

float pick_component(vec4 value, int component)
{
    if (component == 0) return value.r;
    if (component == 1) return value.g;
    if (component == 2) return value.b;
    return value.a;
}

float read_packed(sampler2D tex, int linear_index, int tex_width, int tex_height)
{
    int texel = linear_index / 4;
    int component = linear_index - texel * 4;
    int x = texel - (texel / tex_width) * tex_width;
    int y = texel / tex_width;
    vec2 uv = (vec2(float(x), float(y)) + vec2(0.5, 0.5)) / vec2(float(tex_width), float(tex_height));
    return pick_component(texture2D(tex, uv), component);
}

float stack_at(int out_index)
{
    if (out_index >= OUT_NUMEL) return 0.0;
    int which = out_index - (out_index / 2) * 2;
    int tmp0 = out_index / 2;
    int col = tmp0 - (tmp0 / COLS) * COLS;
    int row = tmp0 / COLS;
    if (which == 0) {
        int source_index = OFFSET0 + row * STRIDE00 + col * STRIDE01;
        return read_packed(input0_tex, source_index, TEX_W0, TEX_H0);
    }
    int source_index = OFFSET1 + row * STRIDE10 + col * STRIDE11;
    return read_packed(input1_tex, source_index, TEX_W1, TEX_H1);
}

void main()
{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * OUT_TEX_W + tex_x) * 4;
    gl_FragColor = vec4(
        stack_at(base),
        stack_at(base + 1),
        stack_at(base + 2),
        stack_at(base + 3)
    );
}
"""
    replacements = {
        "ROWS": rows,
        "COLS": cols,
        "OUT_NUMEL": rows * cols * 2,
        "OFFSET0": offsets[0],
        "OFFSET1": offsets[1],
        "STRIDE00": strides0[0],
        "STRIDE01": strides0[1],
        "STRIDE10": strides1[0],
        "STRIDE11": strides1[1],
        "TEX_W0": tex_widths[0],
        "TEX_W1": tex_widths[1],
        "TEX_H0": tex_heights[0],
        "TEX_H1": tex_heights[1],
        "OUT_TEX_W": out_tex_w,
    }
    for name, value in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        source = source.replace(name, str(value))
    return source.encode("ascii")

def _fill_program(params: tuple) -> int:
    rt = operation_context.gl_runtime()
    if params not in rt.fill_programs:
        diagnostics.trace(f"gm45.compile -> fill GLSL fragment shader params={params}")
        program = gm.make_program(_fill_shader_source(params))
        rt.fill_programs[params] = program
        rt.fill_uniforms[params] = ()
    return rt.fill_programs[params]

def _fill_shader_source(params: tuple) -> bytes:
    numel, out_tex_w, scalar = params
    source = """
#version 120

float fill_at(int linear_index)
{
    if (linear_index >= __NUMEL__) return 0.0;
    return __SCALAR__;
}

void main()
{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * __OUT_TEX_W__ + tex_x) * 4;
    gl_FragColor = vec4(
        fill_at(base),
        fill_at(base + 1),
        fill_at(base + 2),
        fill_at(base + 3)
    );
}
"""
    replacements = {
        "__NUMEL__": int(numel),
        "__OUT_TEX_W__": int(out_tex_w),
        "__SCALAR__": _glsl_float(scalar),
    }
    for name, value in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        source = source.replace(name, str(value))
    return source.encode("ascii")

def _cat_program(params: tuple) -> tuple[int, tuple[int, ...]]:
    rt = operation_context.gl_runtime()
    if params not in rt.cat_programs:
        diagnostics.trace(f"gm45.compile -> cat GLSL fragment shader params={params}")
        program = gm.make_program(_cat_shader_source(params))
        rt.cat_programs[params] = program
        rt.cat_uniforms[params] = tuple(
            gm.glGetUniformLocation(program, f"input{index}_tex".encode("ascii"))
            for index in range(params[0])
        )
    return rt.cat_programs[params], rt.cat_uniforms[params]

def _cat_dim0_2d_program(params: tuple) -> tuple[int, tuple[int, ...]]:
    rt = operation_context.gl_runtime()
    if params not in rt.cat_dim0_2d_programs:
        diagnostics.trace(f"gm45.compile -> 2D dim-0 cat GLSL fragment shader params={params}")
        program = gm.make_program(_cat_dim0_2d_shader_source(params))
        rt.cat_dim0_2d_programs[params] = program
        rt.cat_dim0_2d_uniforms[params] = tuple(
            gm.glGetUniformLocation(program, f"input{index}_tex".encode("ascii"))
            for index in range(params[0])
        )
    return rt.cat_dim0_2d_programs[params], rt.cat_dim0_2d_uniforms[params]

def _cat_dim0_2d_shader_source(params: tuple) -> bytes:
    num_inputs, rows, cols, offsets, tex_widths, tex_heights, out_tex_w = params
    total_rows = sum(rows)
    uniforms = "\n".join(f"uniform sampler2D input{index}_tex;" for index in range(num_inputs))

    branches = []
    row_start = 0
    for index, row_count in enumerate(rows):
        condition = "if" if index == 0 else "else if"
        branches.append(
            f"""
    {condition} (row < {row_start + row_count}) {{
        int local_row = row - {row_start};
        int source_index = {offsets[index]} + local_row * COLS + col;
        return read_packed(input{index}_tex, source_index, {tex_widths[index]}, {tex_heights[index]});
    }}"""
        )
        row_start += row_count

    source = f"""
#version 120
{uniforms}

float pick_component(vec4 value, int component)
{{
    if (component == 0) return value.r;
    if (component == 1) return value.g;
    if (component == 2) return value.b;
    return value.a;
}}

float read_packed(sampler2D tex, int linear_index, int tex_width, int tex_height)
{{
    int texel = linear_index / 4;
    int component = linear_index - texel * 4;
    int x = texel - (texel / tex_width) * tex_width;
    int y = texel / tex_width;
    vec2 uv = (vec2(float(x), float(y)) + vec2(0.5, 0.5)) / vec2(float(tex_width), float(tex_height));
    return pick_component(texture2D(tex, uv), component);
}}

float cat_at(int out_index)
{{
    if (out_index >= OUT_NUMEL) return 0.0;
    int col = out_index - (out_index / COLS) * COLS;
    int row = out_index / COLS;
{''.join(branches)}
    return 0.0;
}}

void main()
{{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * OUT_TEX_W + tex_x) * 4;
    gl_FragColor = vec4(
        cat_at(base),
        cat_at(base + 1),
        cat_at(base + 2),
        cat_at(base + 3)
    );
}}
"""
    replacements = {
        "OUT_NUMEL": total_rows * cols,
        "OUT_TEX_W": out_tex_w,
        "COLS": cols,
    }
    for name, value in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        source = source.replace(name, str(value))
    return source.encode("ascii")

def _cat_shader_source(params: tuple) -> bytes:
    num_inputs, channels, height, width, offsets, tex_widths, tex_heights, out_tex_w = params
    total_channels = sum(channels)
    uniforms = "\n".join(f"uniform sampler2D input{index}_tex;" for index in range(num_inputs))

    branches = []
    channel_start = 0
    for index, channel_count in enumerate(channels):
        condition = "if" if index == 0 else "else if"
        branches.append(
            f"""
    {condition} (oc < {channel_start + channel_count}) {{
        int local_c = oc - {channel_start};
        int source_index = {offsets[index]} + ((local_c * H) + oy) * W + ox;
        return read_packed(input{index}_tex, source_index, {tex_widths[index]}, {tex_heights[index]});
    }}"""
        )
        channel_start += channel_count

    source = f"""
#version 120
{uniforms}

float pick_component(vec4 value, int component)
{{
    if (component == 0) return value.r;
    if (component == 1) return value.g;
    if (component == 2) return value.b;
    return value.a;
}}

float read_packed(sampler2D tex, int linear_index, int tex_width, int tex_height)
{{
    int texel = linear_index / 4;
    int component = linear_index - texel * 4;
    int x = texel - (texel / tex_width) * tex_width;
    int y = texel / tex_width;
    vec2 uv = (vec2(float(x), float(y)) + vec2(0.5, 0.5)) / vec2(float(tex_width), float(tex_height));
    return pick_component(texture2D(tex, uv), component);
}}

float cat_at(int out_index)
{{
    if (out_index >= OUT_NUMEL) return 0.0;
    int ox = out_index - (out_index / W) * W;
    int tmp0 = out_index / W;
    int oy = tmp0 - (tmp0 / H) * H;
    int oc = tmp0 / H;
{''.join(branches)}
    return 0.0;
}}

void main()
{{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * OUT_TEX_W + tex_x) * 4;
    gl_FragColor = vec4(
        cat_at(base),
        cat_at(base + 1),
        cat_at(base + 2),
        cat_at(base + 3)
    );
}}
"""
    replacements = {
        "OUT_NUMEL": total_channels * height * width,
        "OUT_TEX_W": out_tex_w,
        "H": height,
        "W": width,
    }
    for name, value in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        source = source.replace(name, str(value))
    return source.encode("ascii")

def _cat_lastdim_program(params: tuple) -> tuple[int, tuple[int, ...]]:
    rt = operation_context.gl_runtime()
    if params not in rt.cat_lastdim_programs:
        diagnostics.trace(f"gm45.compile -> 3D last-dim cat GLSL fragment shader params={params}")
        program = gm.make_program(_cat_lastdim_shader_source(params))
        rt.cat_lastdim_programs[params] = program
        rt.cat_lastdim_uniforms[params] = tuple(
            gm.glGetUniformLocation(program, f"input{index}_tex".encode("ascii"))
            for index in range(params[0])
        )
    return rt.cat_lastdim_programs[params], rt.cat_lastdim_uniforms[params]

def _cat_lastdim_shader_source(params: tuple) -> bytes:
    num_inputs, rows, widths, offsets, tex_widths, tex_heights, out_tex_w = params
    total_width = sum(widths)
    uniforms = "\n".join(f"uniform sampler2D input{index}_tex;" for index in range(num_inputs))

    branches = []
    pos_start = 0
    for index, width in enumerate(widths):
        condition = "if" if index == 0 else "else if"
        branches.append(
            f"""
    {condition} (pos < {pos_start + width}) {{
        int local_pos = pos - {pos_start};
        int source_index = {offsets[index]} + row * {width} + local_pos;
        return read_packed(input{index}_tex, source_index, {tex_widths[index]}, {tex_heights[index]});
    }}"""
        )
        pos_start += width

    source = f"""
#version 120
{uniforms}

float pick_component(vec4 value, int component)
{{
    if (component == 0) return value.r;
    if (component == 1) return value.g;
    if (component == 2) return value.b;
    return value.a;
}}

float read_packed(sampler2D tex, int linear_index, int tex_width, int tex_height)
{{
    int texel = linear_index / 4;
    int component = linear_index - texel * 4;
    int x = texel - (texel / tex_width) * tex_width;
    int y = texel / tex_width;
    vec2 uv = (vec2(float(x), float(y)) + vec2(0.5, 0.5)) / vec2(float(tex_width), float(tex_height));
    return pick_component(texture2D(tex, uv), component);
}}

float cat_at(int out_index)
{{
    if (out_index >= OUT_NUMEL) return 0.0;
    int pos = out_index - (out_index / OUT_WIDTH) * OUT_WIDTH;
    int row = out_index / OUT_WIDTH;
{''.join(branches)}
    return 0.0;
}}

void main()
{{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * OUT_TEX_W + tex_x) * 4;
    gl_FragColor = vec4(
        cat_at(base),
        cat_at(base + 1),
        cat_at(base + 2),
        cat_at(base + 3)
    );
}}
"""
    replacements = {
        "OUT_NUMEL": rows * total_width,
        "OUT_WIDTH": total_width,
        "OUT_TEX_W": out_tex_w,
    }
    for name, value in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        source = source.replace(name, str(value))
    return source.encode("ascii")

def _cat_dim1_3d_program(params: tuple) -> tuple[int, int, int]:
    rt = operation_context.gl_runtime()
    if params not in rt.cat_dim1_3d_programs:
        diagnostics.trace(f"gm45.compile -> 3D dim-1 cat GLSL fragment shader params={params}")
        program = gm.make_program(_cat_dim1_3d_shader_source(params))
        rt.cat_dim1_3d_programs[params] = program
        rt.cat_dim1_3d_uniforms[params] = (
            gm.glGetUniformLocation(program, b"input0_tex"),
            gm.glGetUniformLocation(program, b"input1_tex"),
        )
    left_loc, right_loc = rt.cat_dim1_3d_uniforms[params]
    return rt.cat_dim1_3d_programs[params], left_loc, right_loc

def _cat_dim1_3d_shader_source(params: tuple) -> bytes:
    channels0, channels1, anchors, offsets, tex_widths, tex_heights, out_tex_w = params
    source = """
#version 120
uniform sampler2D input0_tex;
uniform sampler2D input1_tex;

float pick_component(vec4 value, int component)
{
    if (component == 0) return value.r;
    if (component == 1) return value.g;
    if (component == 2) return value.b;
    return value.a;
}

float read_packed(sampler2D tex, int linear_index, int tex_width, int tex_height)
{
    int texel = linear_index / 4;
    int component = linear_index - texel * 4;
    int x = texel - (texel / tex_width) * tex_width;
    int y = texel / tex_width;
    vec2 uv = (vec2(float(x), float(y)) + vec2(0.5, 0.5)) /
              vec2(float(tex_width), float(tex_height));
    return pick_component(texture2D(tex, uv), component);
}

float cat_at(int out_index)
{
    if (out_index >= OUT_NUMEL) return 0.0;
    int anchor = out_index - (out_index / ANCHORS) * ANCHORS;
    int channel = out_index / ANCHORS;
    if (channel < CHANNELS0) {
        return read_packed(input0_tex, OFFSET0 + channel * ANCHORS + anchor, TEX_W0, TEX_H0);
    }
    return read_packed(input1_tex, OFFSET1 + (channel - CHANNELS0) * ANCHORS + anchor, TEX_W1, TEX_H1);
}

void main()
{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * OUT_TEX_W + tex_x) * 4;
    gl_FragColor = vec4(cat_at(base), cat_at(base + 1), cat_at(base + 2), cat_at(base + 3));
}
"""
    replacements = {
        "CHANNELS0": channels0, "CHANNELS1": channels1, "ANCHORS": anchors,
        "OFFSET0": offsets[0], "OFFSET1": offsets[1],
        "TEX_W0": tex_widths[0], "TEX_W1": tex_widths[1],
        "TEX_H0": tex_heights[0], "TEX_H1": tex_heights[1],
        "OUT_TEX_W": out_tex_w, "OUT_NUMEL": (channels0 + channels1) * anchors,
    }
    for name, value in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        source = source.replace(name, str(value))
    return source.encode("ascii")

def _render_stack(args, kwargs) -> "MatrixManTensor":
    tensors = list(args[0])
    dim = int(args[1]) if len(args) > 1 else int(kwargs.get("dim", 0))
    if not tensors:
        raise RuntimeError("gm45 stack requires at least one input tensor")
    rank = len(tensors[0].shape)
    normalized_dim = dim + rank + 1 if dim < 0 else dim
    if len(tensors) != 2:
        raise RuntimeError("gm45 stack currently supports only the two-input YOLO meshgrid case")
    if rank != 2 or normalized_dim != 2:
        raise RuntimeError("gm45 stack currently supports only 2D inputs stacked along final dim")

    shape = tuple(int(v) for v in tensors[0].shape)
    for tensor in tensors:
        if not isinstance(tensor, MatrixManTensor):
            raise RuntimeError("gm45 stack requires all inputs to be MatrixManTensor instances")
        if tensor.dtype != torch.float32:
            raise RuntimeError("gm45 stack supports only float32 inputs")
        if tensor._owner.layout.kind != "packed_rgba":
            raise RuntimeError("gm45 stack supports only packed_rgba input storage")
        if tuple(int(v) for v in tensor.shape) != shape:
            raise RuntimeError("gm45 stack requires identical input shapes")

    rows, cols = shape
    out_shape = (rows, cols, 2)
    out_owner = operation_context.output_texture(out_shape)
    params = (
        rows,
        cols,
        tuple(tensor._storage_offset for tensor in tensors),
        tensors[0]._logical_strides,
        tensors[1]._logical_strides,
        tuple(tensor._owner.layout.texture_width for tensor in tensors),
        tuple(tensor._owner.layout.texture_height for tensor in tensors),
        out_owner.layout.texture_width,
    )
    program, input_locs = _stack_program(params)
    rt = operation_context.gl_runtime()

    diagnostics.trace(
        "gm45.kernel -> stack shader:\n"
        f"  dim={dim} normalized_dim={normalized_dim}\n"
        f"  input 0: texture #{tensors[0]._owner.texture} shape={list(shape)} "
        f"offset={tensors[0]._storage_offset} strides={list(tensors[0]._logical_strides)}\n"
        f"  input 1: texture #{tensors[1]._owner.texture} shape={list(shape)} "
        f"offset={tensors[1]._storage_offset} strides={list(tensors[1]._logical_strides)}\n"
        f"  -> output texture #{out_owner.texture} shape={list(out_shape)} "
        f"strides={list(contiguous_strides(out_shape))} offset=0"
    )

    operation_context.attach_output(out_owner)
    status = gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER)
    if status != gm.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(f"gm45 stack framebuffer incomplete: 0x{status:04x}")

    gm.glUseProgram(program)
    for index, tensor in enumerate(tensors):
        gm.glActiveTexture([gm.GL_TEXTURE0, gm.GL_TEXTURE1][index])
        gm.glBindTexture(gm.GL_TEXTURE_2D, tensor._owner.texture)
        gm.glUniform1i(input_locs[index], index)

    operation_context.draw_fullscreen_quad()
    diagnostics.trace(f"gm45.opengl -> submitted stack fullscreen quad, output texture #{out_owner.texture}")

    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after stack: 0x{err:04x}")
    return MatrixManTensor._from_owner(out_owner, out_shape)

def _render_fill_scalar(args) -> "MatrixManTensor":
    tensor = args[0]
    scalar = args[1]
    if not isinstance(tensor, MatrixManTensor):
        raise RuntimeError("gm45 fill_ requires a MatrixManTensor destination")
    if tensor.dtype != torch.float32:
        raise RuntimeError("gm45 fill_ supports only float32 destinations")
    if tensor._owner.layout.kind != "packed_rgba":
        raise RuntimeError("gm45 fill_ supports only packed_rgba destination storage")
    if tensor._storage_offset != 0:
        raise RuntimeError("gm45 fill_ currently supports only storage_offset=0")
    operation_context.require_contiguous(tensor, "fill_")
    if isinstance(scalar, torch.Tensor):
        if scalar.device.type != "cpu" or scalar.numel() != 1:
            raise RuntimeError("gm45 fill_ scalar must be a Python scalar or CPU scalar tensor")
        scalar_value = float(scalar.item())
    else:
        scalar_value = float(scalar)

    shape = tuple(int(v) for v in tensor.shape)
    if numel(shape) <= 0:
        raise RuntimeError("gm45 fill_ currently supports only non-empty tensors")

    out_owner = operation_context.output_texture(shape)
    params = (numel(shape), out_owner.layout.texture_width, scalar_value)
    program = _fill_program(params)
    rt = operation_context.gl_runtime()

    diagnostics.trace(
        "gm45.kernel -> fill_ scalar shader:\n"
        f"  destination old texture #{tensor._owner.texture} shape={list(shape)} "
        f"offset={tensor._storage_offset} strides={list(tensor._logical_strides)}\n"
        f"  scalar={scalar_value:.10g}\n"
        f"  -> replacement texture #{out_owner.texture} shape={list(shape)} offset=0\n"
        "  note: fill_ updates the wrapper owner to avoid OpenGL FBO feedback hazards"
    )

    operation_context.attach_output(out_owner)
    status = gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER)
    if status != gm.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(f"gm45 fill_ framebuffer incomplete: 0x{status:04x}")

    gm.glUseProgram(program)
    operation_context.draw_fullscreen_quad()
    diagnostics.trace(f"gm45.opengl -> submitted fill_ fullscreen quad, replacement texture #{out_owner.texture}")

    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after fill_: 0x{err:04x}")

    tensor._owner = out_owner
    tensor._shape = shape
    tensor._storage_offset = 0
    tensor._logical_strides = contiguous_strides(shape)
    return tensor

def _render_cat_dim1_3d(tensors: list["MatrixManTensor"], dim: int) -> "MatrixManTensor":
    if len(tensors) != 2 or dim != 1:
        raise RuntimeError("gm45 3D dim-1 cat currently supports exactly two inputs")
    shapes = [tuple(int(v) for v in tensor.shape) for tensor in tensors]
    if any(len(shape) != 3 or shape[0] != 1 for shape in shapes):
        raise RuntimeError("gm45 3D dim-1 cat supports only batch-1 rank-3 tensors")
    if shapes[0][2] != shapes[1][2]:
        raise RuntimeError("gm45 3D dim-1 cat requires matching final dimensions")
    for tensor in tensors:
        if not isinstance(tensor, MatrixManTensor) or tensor.dtype != torch.float32:
            raise RuntimeError("gm45 3D dim-1 cat requires float32 MatrixManTensor inputs")
        if tensor._owner.layout.kind != "packed_rgba":
            raise RuntimeError("gm45 3D dim-1 cat requires packed_rgba input storage")
        operation_context.require_contiguous(tensor, "3D dim-1 cat")
    channels0, channels1, anchors = shapes[0][1], shapes[1][1], shapes[0][2]
    out_shape = (1, channels0 + channels1, anchors)
    out_owner = operation_context.output_texture(out_shape)
    params = (
        channels0, channels1, anchors,
        tuple(tensor._storage_offset for tensor in tensors),
        tuple(tensor._owner.layout.texture_width for tensor in tensors),
        tuple(tensor._owner.layout.texture_height for tensor in tensors),
        out_owner.layout.texture_width,
    )
    program, input0_loc, input1_loc = _cat_dim1_3d_program(params)
    rt = operation_context.gl_runtime()
    diagnostics.trace(
        "gm45.kernel -> 3D dim-1 cat shader:\n"
        f"  input 0: texture #{tensors[0]._owner.texture} shape={list(shapes[0])} offset={tensors[0]._storage_offset}\n"
        f"  input 1: texture #{tensors[1]._owner.texture} shape={list(shapes[1])} offset={tensors[1]._storage_offset}\n"
        f"  -> output texture #{out_owner.texture} shape={list(out_shape)} offset=0"
    )
    operation_context.attach_output(out_owner)
    if gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER) != gm.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError("gm45 3D dim-1 cat framebuffer incomplete")
    gm.glUseProgram(program)
    for unit, tensor, uniform in [(gm.GL_TEXTURE0, tensors[0], input0_loc), (gm.GL_TEXTURE1, tensors[1], input1_loc)]:
        gm.glActiveTexture(unit)
        gm.glBindTexture(gm.GL_TEXTURE_2D, tensor._owner.texture)
        gm.glUniform1i(uniform, unit - gm.GL_TEXTURE0)
    with profiling.gpu_timer("Cat"):
        operation_context.draw_fullscreen_quad()
    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after 3D dim-1 cat: 0x{err:04x}")
    return MatrixManTensor._from_owner(out_owner, out_shape)

def _render_cat(args, kwargs) -> "MatrixManTensor":
    tensors = list(args[0])
    dim = int(args[1]) if len(args) > 1 else int(kwargs.get("dim", 0))
    if dim < 0 and tensors:
        dim += len(tensors[0].shape)
    if tensors and len(tensors[0].shape) == 2 and dim == 0:
        return _render_cat_dim0_2d(tensors, dim)
    if tensors and len(tensors[0].shape) == 3 and dim == 1:
        return _render_cat_dim1_3d(tensors, dim)
    if tensors and len(tensors[0].shape) == 3 and dim == 2:
        return _render_cat_lastdim_3d(tensors, dim)
    if not 2 <= len(tensors) <= 4:
        raise RuntimeError("gm45 cat currently supports 2 to 4 input tensors")
    if dim != 1:
        raise RuntimeError("gm45 cat currently supports only NCHW channel concatenation with dim=1")

    shapes = [tuple(int(v) for v in tensor.shape) for tensor in tensors]
    first_shape = shapes[0]
    if len(first_shape) != 4 or first_shape[0] != 1:
        raise RuntimeError("gm45 cat supports only batch-1 NCHW 4D tensors")
    _, _, height, width = first_shape

    for tensor, shape in zip(tensors, shapes):
        if not isinstance(tensor, MatrixManTensor):
            raise RuntimeError("gm45 cat requires all inputs to be MatrixManTensor instances")
        if tensor.dtype != torch.float32:
            raise RuntimeError("gm45 cat supports only float32 inputs")
        if tensor._owner.layout.kind != "packed_rgba":
            raise RuntimeError("gm45 cat currently supports only packed_rgba tensor storage")
        operation_context.require_contiguous(tensor, "cat")
        if len(shape) != 4 or shape[0] != 1:
            raise RuntimeError("gm45 cat supports only batch-1 NCHW 4D tensors")
        if shape[2] != height or shape[3] != width:
            raise RuntimeError("gm45 cat requires matching NCHW H and W dimensions")

    channels = tuple(shape[1] for shape in shapes)
    out_shape = (1, sum(channels), height, width)
    out_owner = operation_context.output_texture(out_shape)
    params = (
        len(tensors),
        channels,
        height,
        width,
        tuple(tensor._storage_offset for tensor in tensors),
        tuple(tensor._owner.layout.texture_width for tensor in tensors),
        tuple(tensor._owner.layout.texture_height for tensor in tensors),
        out_owner.layout.texture_width,
    )
    program, input_locs = _cat_program(params)
    rt = operation_context.gl_runtime()

    lines = [
        "gm45.kernel -> cat shader:",
        f"  dim={dim}",
        f"  inputs={len(tensors)}",
    ]
    for index, tensor in enumerate(tensors):
        lines.append(
            f"  input {index}: texture #{tensor._owner.texture} "
            f"shape={list(tensor.shape)} offset={tensor._storage_offset}"
        )
    lines.append(f"  -> output texture #{out_owner.texture} shape={list(out_shape)} offset=0")
    diagnostics.trace("\n".join(lines))

    operation_context.attach_output(out_owner)
    status = gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER)
    if status != gm.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(f"gm45 cat framebuffer incomplete: 0x{status:04x}")

    gm.glUseProgram(program)
    texture_units = [gm.GL_TEXTURE0, gm.GL_TEXTURE1, gm.GL_TEXTURE2, gm.GL_TEXTURE3]
    for index, tensor in enumerate(tensors):
        gm.glActiveTexture(texture_units[index])
        gm.glBindTexture(gm.GL_TEXTURE_2D, tensor._owner.texture)
        gm.glUniform1i(input_locs[index], index)

    with profiling.gpu_timer("Cat"):
        operation_context.draw_fullscreen_quad()
    diagnostics.trace(f"gm45.opengl -> submitted cat fullscreen quad, output texture #{out_owner.texture}")

    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after cat: 0x{err:04x}")
    return MatrixManTensor._from_owner(out_owner, out_shape)

def _render_cat_dim0_2d(tensors: list["MatrixManTensor"], dim: int) -> "MatrixManTensor":
    if not 2 <= len(tensors) <= 4:
        raise RuntimeError("gm45 2D dim-0 cat currently supports 2 to 4 input tensors")
    shapes = [tuple(int(v) for v in tensor.shape) for tensor in tensors]
    first_shape = shapes[0]
    if len(first_shape) != 2 or dim != 0:
        raise RuntimeError("gm45 2D cat supports only dim=0")
    _, cols = first_shape

    for tensor, shape in zip(tensors, shapes):
        if not isinstance(tensor, MatrixManTensor):
            raise RuntimeError("gm45 2D dim-0 cat requires all inputs to be MatrixManTensor instances")
        if tensor.dtype != torch.float32:
            raise RuntimeError("gm45 2D dim-0 cat supports only float32 inputs")
        if tensor._owner.layout.kind != "packed_rgba":
            raise RuntimeError("gm45 2D dim-0 cat currently supports only packed_rgba tensor storage")
        operation_context.require_contiguous(tensor, "2D dim-0 cat")
        if len(shape) != 2:
            raise RuntimeError("gm45 2D dim-0 cat supports only 2D tensors")
        if shape[1] != cols:
            raise RuntimeError("gm45 2D dim-0 cat requires matching second dimension")

    rows = tuple(shape[0] for shape in shapes)
    out_shape = (sum(rows), cols)
    out_owner = operation_context.output_texture(out_shape)
    params = (
        len(tensors),
        rows,
        cols,
        tuple(tensor._storage_offset for tensor in tensors),
        tuple(tensor._owner.layout.texture_width for tensor in tensors),
        tuple(tensor._owner.layout.texture_height for tensor in tensors),
        out_owner.layout.texture_width,
    )
    program, input_locs = _cat_dim0_2d_program(params)
    rt = operation_context.gl_runtime()

    lines = [
        "gm45.kernel -> 2D dim-0 cat shader:",
        f"  dim={dim}",
        f"  inputs={len(tensors)}",
    ]
    for index, tensor in enumerate(tensors):
        lines.append(
            f"  input {index}: texture #{tensor._owner.texture} "
            f"shape={list(tensor.shape)} offset={tensor._storage_offset} "
            f"strides={list(tensor._logical_strides)}"
        )
    lines.append(f"  -> output texture #{out_owner.texture} shape={list(out_shape)} offset=0")
    diagnostics.trace("\n".join(lines))

    operation_context.attach_output(out_owner)
    status = gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER)
    if status != gm.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(f"gm45 2D dim-0 cat framebuffer incomplete: 0x{status:04x}")

    gm.glUseProgram(program)
    texture_units = [gm.GL_TEXTURE0, gm.GL_TEXTURE1, gm.GL_TEXTURE2, gm.GL_TEXTURE3]
    for index, tensor in enumerate(tensors):
        gm.glActiveTexture(texture_units[index])
        gm.glBindTexture(gm.GL_TEXTURE_2D, tensor._owner.texture)
        gm.glUniform1i(input_locs[index], index)

    with profiling.gpu_timer("Cat"):
        operation_context.draw_fullscreen_quad()
    diagnostics.trace(f"gm45.opengl -> submitted 2D dim-0 cat fullscreen quad, output texture #{out_owner.texture}")

    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after 2D dim-0 cat: 0x{err:04x}")
    return MatrixManTensor._from_owner(out_owner, out_shape)

def _render_cat_lastdim_3d(tensors: list["MatrixManTensor"], dim: int) -> "MatrixManTensor":
    if not 2 <= len(tensors) <= 4:
        raise RuntimeError("gm45 3D last-dim cat currently supports 2 to 4 input tensors")
    shapes = [tuple(int(v) for v in tensor.shape) for tensor in tensors]
    first_shape = shapes[0]
    if len(first_shape) != 3:
        raise RuntimeError("gm45 3D last-dim cat supports only 3D tensors")
    if dim != 2:
        raise RuntimeError("gm45 3D cat supports only final logical dimension")
    batch, rows, _ = first_shape

    for tensor, shape in zip(tensors, shapes):
        if not isinstance(tensor, MatrixManTensor):
            raise RuntimeError("gm45 3D last-dim cat requires all inputs to be MatrixManTensor instances")
        if tensor.dtype != torch.float32:
            raise RuntimeError("gm45 3D last-dim cat supports only float32 inputs")
        if tensor._owner.layout.kind != "packed_rgba":
            raise RuntimeError("gm45 3D last-dim cat currently supports only packed_rgba tensor storage")
        operation_context.require_contiguous(tensor, "3D last-dim cat")
        if len(shape) != 3:
            raise RuntimeError("gm45 3D last-dim cat supports only 3D tensors")
        if shape[0] != batch or shape[1] != rows:
            raise RuntimeError("gm45 3D last-dim cat requires matching leading dimensions")

    widths = tuple(shape[2] for shape in shapes)
    out_shape = (batch, rows, sum(widths))
    out_owner = operation_context.output_texture(out_shape)
    params = (
        len(tensors),
        batch * rows,
        widths,
        tuple(tensor._storage_offset for tensor in tensors),
        tuple(tensor._owner.layout.texture_width for tensor in tensors),
        tuple(tensor._owner.layout.texture_height for tensor in tensors),
        out_owner.layout.texture_width,
    )
    program, input_locs = _cat_lastdim_program(params)
    rt = operation_context.gl_runtime()

    lines = [
        "gm45.kernel -> 3D last-dim cat shader:",
        f"  dim={dim}",
        f"  inputs={len(tensors)}",
    ]
    for index, tensor in enumerate(tensors):
        lines.append(
            f"  input {index}: texture #{tensor._owner.texture} "
            f"shape={list(tensor.shape)} offset={tensor._storage_offset}"
        )
    lines.append(f"  -> output texture #{out_owner.texture} shape={list(out_shape)} offset=0")
    diagnostics.trace("\n".join(lines))

    operation_context.attach_output(out_owner)
    status = gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER)
    if status != gm.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(f"gm45 3D last-dim cat framebuffer incomplete: 0x{status:04x}")

    gm.glUseProgram(program)
    texture_units = [gm.GL_TEXTURE0, gm.GL_TEXTURE1, gm.GL_TEXTURE2, gm.GL_TEXTURE3]
    for index, tensor in enumerate(tensors):
        gm.glActiveTexture(texture_units[index])
        gm.glBindTexture(gm.GL_TEXTURE_2D, tensor._owner.texture)
        gm.glUniform1i(input_locs[index], index)

    with profiling.gpu_timer("Cat"):
        operation_context.draw_fullscreen_quad()
    diagnostics.trace(f"gm45.opengl -> submitted 3D last-dim cat fullscreen quad, output texture #{out_owner.texture}")

    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after 3D last-dim cat: 0x{err:04x}")
    return MatrixManTensor._from_owner(out_owner, out_shape)
