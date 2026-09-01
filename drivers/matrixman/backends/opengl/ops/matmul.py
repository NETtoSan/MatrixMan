"""OpenGL matrix multiplication operation boundary."""

from __future__ import annotations

import torch

from .. import diagnostics, gpumatrix as gm, operation_context, resources
from ..storage import StorageLayout, contiguous_strides
from ....tensor import MatrixManTensor
from ..tensor import owner_from_texture


def new_empty_matrix_texture(n: int):
    layout = StorageLayout("matrix2d_red", n, n, n * n)
    texture = resources.allocate_matrix_texture(n)
    owner = owner_from_texture(texture, layout)
    diagnostics.trace(f"gm45.texture_alloc -> matrix output texture #{owner.texture} ({n}x{n}, RGBA32F red channel)")
    return owner


def render_matrix_binary(kind: str, left: MatrixManTensor, right: MatrixManTensor, alpha: float = 1.0) -> MatrixManTensor:
    if left.dim() != 2 or left.shape[0] != left.shape[1]:
        raise RuntimeError(f"gm45 {kind} only supports square 2D matrices")
    if alpha != 1.0:
        raise RuntimeError(f"gm45 legacy matrix {kind} only supports alpha=1")
    if left.dtype != torch.float32 or right.dtype != torch.float32:
        raise RuntimeError(f"gm45 {kind} only supports float32")
    if left._owner.layout.kind != "matrix2d_red" or right._owner.layout.kind != "matrix2d_red":
        raise RuntimeError(f"gm45 {kind} currently requires legacy matrix2d_red texture storage")
    if left._storage_offset != 0 or right._storage_offset != 0:
        raise RuntimeError(f"gm45 {kind} does not support nonzero storage offsets for legacy matrix storage")

    n = int(left.shape[0])
    out_owner = new_empty_matrix_texture(n)
    runtime = operation_context.gl_runtime()
    program, left_loc, right_loc = operation_context.program(kind, n)
    symbol = "+" if kind == "add" else "x"
    diagnostics.trace(
        f"gm45.kernel -> {kind} shader: texture #{left._owner.texture} "
        f"{symbol} texture #{right._owner.texture} -> texture #{out_owner.texture}"
    )

    operation_context.attach_output(out_owner, n, n)
    status = gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER)
    if status != gm.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(f"gm45 framebuffer incomplete: 0x{status:04x}")

    gm.glUseProgram(program)
    gm.glActiveTexture(gm.GL_TEXTURE0)
    gm.glBindTexture(gm.GL_TEXTURE_2D, left._owner.texture)
    gm.glUniform1i(left_loc, 0)
    gm.glActiveTexture(gm.GL_TEXTURE1)
    gm.glBindTexture(gm.GL_TEXTURE_2D, right._owner.texture)
    gm.glUniform1i(right_loc, 1)

    operation_context.draw_fullscreen_quad()
    diagnostics.trace(f"gm45.opengl -> submitted fullscreen quad to GLSL fragment shader, output texture #{out_owner.texture}")

    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after {kind}: 0x{err:04x}")
    return operation_context.tensor_from_owner(out_owner, (n, n))


def render_matmul(left: MatrixManTensor, right: MatrixManTensor) -> MatrixManTensor:
    if not isinstance(left, MatrixManTensor) or not isinstance(right, MatrixManTensor):
        raise RuntimeError("gm45 matmul requires both inputs to be gm45 tensors")
    if left.dim() != 2 or right.dim() != 2:
        raise RuntimeError("gm45 matmul requires 2D matrices")
    if int(left.shape[1]) != int(right.shape[0]):
        raise RuntimeError(
            f"gm45 matmul shapes cannot be multiplied ({left.shape[0]}x{left.shape[1]} and "
            f"{right.shape[0]}x{right.shape[1]})"
        )
    if left.dtype != torch.float32 or right.dtype != torch.float32:
        raise RuntimeError("gm45 matmul only supports float32")
    if tuple(left._logical_strides) != contiguous_strides(tuple(left.shape)) or tuple(right._logical_strides) != contiguous_strides(tuple(right.shape)):
        raise RuntimeError("gm45 matmul currently requires contiguous logical matrices")
    if left._storage_offset != 0 or right._storage_offset != 0:
        raise RuntimeError("gm45 matmul does not support nonzero storage offsets")

    m, k = (int(value) for value in left.shape)
    _, n = (int(value) for value in right.shape)
    if m == k == n and left._owner.layout.kind == "matrix2d_red" and right._owner.layout.kind == "matrix2d_red":
        return render_matrix_binary("matmul", left, right)
    return _render_rectangular_matmul(left, right, m, k, n)


def _read_function(name: str, kind: str, width: int, height: int) -> str:
    if kind == "matrix2d_red":
        return f"""
float {name}(sampler2D tex, int linear_index)
{{
    int x = linear_index - (linear_index / {width}) * {width};
    int y = linear_index / {width};
    vec2 uv = (vec2(float(x), float(y)) + vec2(0.5, 0.5)) / vec2(float({width}), float({height}));
    return texture2D(tex, uv).r;
}}
"""
    return f"""
float {name}(sampler2D tex, int linear_index)
{{
    int texel = linear_index / 4;
    int component = linear_index - texel * 4;
    int x = texel - (texel / {width}) * {width};
    int y = texel / {width};
    vec4 value = texture2D(tex, (vec2(float(x), float(y)) + vec2(0.5, 0.5)) / vec2(float({width}), float({height})));
    if (component == 0) return value.r;
    if (component == 1) return value.g;
    if (component == 2) return value.b;
    return value.a;
}}
"""


def _shader_source(params: tuple) -> bytes:
    m, k, n, left_kind, right_kind, left_w, left_h, right_w, right_h, out_w = params
    return f"""
#version 120
uniform sampler2D left_tex;
uniform sampler2D right_tex;

{_read_function('read_left', left_kind, left_w, left_h)}
{_read_function('read_right', right_kind, right_w, right_h)}

float matmul_at(int linear_index)
{{
    if (linear_index >= {m * n}) return 0.0;
    int row = linear_index / {n};
    int col = linear_index - row * {n};
    float acc = 0.0;
    for (int kk = 0; kk < {k}; ++kk)
        acc += read_left(left_tex, row * {k} + kk) * read_right(right_tex, kk * {n} + col);
    return acc;
}}

void main()
{{
    int base = (int(floor(gl_FragCoord.y)) * {out_w} + int(floor(gl_FragCoord.x))) * 4;
    gl_FragColor = vec4(matmul_at(base), matmul_at(base + 1), matmul_at(base + 2), matmul_at(base + 3));
}}
""".encode("ascii")


def _render_rectangular_matmul(left: MatrixManTensor, right: MatrixManTensor, m: int, k: int, n: int) -> MatrixManTensor:
    out_owner = operation_context.output_texture((m, n))
    runtime = operation_context.gl_runtime()
    left_layout = left._owner.layout
    right_layout = right._owner.layout
    params = (
        m, k, n, left_layout.kind, right_layout.kind,
        left_layout.texture_width, left_layout.texture_height,
        right_layout.texture_width, right_layout.texture_height,
        out_owner.layout.texture_width,
    )
    if params not in runtime.packed_matmul_programs:
        diagnostics.trace(f"gm45.compile -> rectangular matmul GLSL fragment shader params={params}")
        program = gm.make_program(_shader_source(params))
        runtime.packed_matmul_programs[params] = program
        runtime.packed_matmul_uniforms[params] = (
            gm.glGetUniformLocation(program, b"left_tex"),
            gm.glGetUniformLocation(program, b"right_tex"),
        )
    program = runtime.packed_matmul_programs[params]
    left_loc, right_loc = runtime.packed_matmul_uniforms[params]
    diagnostics.trace(
        f"gm45.kernel -> rectangular matmul: texture #{left._owner.texture} x "
        f"texture #{right._owner.texture} -> texture #{out_owner.texture}"
    )
    operation_context.attach_output(out_owner)
    operation_context.framebuffer_complete("gm45 framebuffer incomplete for matmul")
    gm.glUseProgram(program)
    gm.glActiveTexture(gm.GL_TEXTURE0)
    gm.glBindTexture(gm.GL_TEXTURE_2D, left._owner.texture)
    gm.glUniform1i(left_loc, 0)
    gm.glActiveTexture(gm.GL_TEXTURE1)
    gm.glBindTexture(gm.GL_TEXTURE_2D, right._owner.texture)
    gm.glUniform1i(right_loc, 1)
    operation_context.draw_fullscreen_quad()
    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after matmul: 0x{err:04x}")
    return operation_context.tensor_from_owner(out_owner, (m, n))
