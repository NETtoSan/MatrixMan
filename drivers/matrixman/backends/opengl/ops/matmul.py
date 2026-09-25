"""OpenGL matrix multiplication operation boundary."""

from __future__ import annotations

import torch

from .. import diagnostics, gpumatrix as gm, kernels, operation_context, resources
from ..storage import StorageLayout, contiguous_strides
from ....tensor import MatrixManTensor
from ..tensor import owner_from_texture


def _prepare_gemm_operand(value, parameter_kind: str) -> MatrixManTensor:
    """Upload a CPU GEMM parameter view, without doing arithmetic on CPU."""
    if isinstance(value, MatrixManTensor):
        return value
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise RuntimeError(
            f"gm45 GEMM {parameter_kind} must be a MatrixMan tensor or CPU float32 parameter"
        )
    if value.dtype != torch.float32:
        raise RuntimeError(f"gm45 GEMM CPU {parameter_kind} must be float32")
    owner = resources.cached_parameter_texture(
        value,
        f"gemm_{parameter_kind}",
        operation="GEMM",
        parameter_role=parameter_kind,
    )
    return MatrixManTensor._from_owner(owner, tuple(int(size) for size in value.shape))


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
    left = _prepare_gemm_operand(left, "mat1")
    right = _prepare_gemm_operand(right, "mat2")
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
    for name, tensor in (("left", left), ("right", right)):
        if tensor._owner.layout.kind not in {"packed_rgba", "matrix2d_red"}:
            raise RuntimeError(f"gm45 matmul {name} requires packed_rgba or matrix2d_red storage")
        if tensor._owner.layout.kind == "matrix2d_red" and tensor._storage_offset != 0:
            raise RuntimeError(f"gm45 matmul does not support nonzero storage offsets for {name} matrix2d_red storage")

    m, k = (int(value) for value in left.shape)
    _, n = (int(value) for value in right.shape)
    if m == k == n and left._owner.layout.kind == "matrix2d_red" and right._owner.layout.kind == "matrix2d_red":
        return render_matrix_binary("matmul", left, right)
    return _render_gemm(left, right, m, k, n)


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
    (
        m, k, n, left_kind, right_kind, left_offset, right_offset,
        left_w, left_h, right_w, right_h, out_w,
        bias_kind, bias_offset, bias_w, bias_h, bias_rank, alpha, beta,
    ) = params
    return f"""
#version 120
uniform sampler2D left_tex;
uniform sampler2D right_tex;
uniform sampler2D bias_tex;

{_read_function('read_left', left_kind, left_w, left_h)}
{_read_function('read_right', right_kind, right_w, right_h)}
{_read_function('read_bias', bias_kind, bias_w, bias_h)}

float matmul_at(int linear_index)
{{
    if (linear_index >= {m * n}) return 0.0;
    int row = linear_index / {n};
    int col = linear_index - row * {n};
    float acc = 0.0;
    for (int kk = 0; kk < {k}; ++kk)
        acc += read_left(left_tex, {left_offset} + row * {k} + kk) * read_right(right_tex, {right_offset} + kk * {n} + col);
    acc *= {alpha};
    if ({bias_rank} != 0)
        acc += {beta} * read_bias(bias_tex, {bias_offset} + ({'col' if bias_rank == 1 else f'row * {n} + col'}));
    return acc;
}}

void main()
{{
    int base = (int(floor(gl_FragCoord.y)) * {out_w} + int(floor(gl_FragCoord.x))) * 4;
    gl_FragColor = vec4(matmul_at(base), matmul_at(base + 1), matmul_at(base + 2), matmul_at(base + 3));
}}
""".encode("ascii")


def _validate_gemm_inputs(left: MatrixManTensor, right: MatrixManTensor) -> tuple[int, int, int]:
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
    for name, tensor in (("left", left), ("right", right)):
        if tensor._owner.layout.kind not in {"packed_rgba", "matrix2d_red"}:
            raise RuntimeError(f"gm45 matmul {name} requires packed_rgba or matrix2d_red storage")
        if tensor._owner.layout.kind == "matrix2d_red" and tensor._storage_offset != 0:
            raise RuntimeError(f"gm45 matmul does not support nonzero storage offsets for {name} matrix2d_red storage")
    return int(left.shape[0]), int(left.shape[1]), int(right.shape[1])


def _render_gemm(
    left: MatrixManTensor,
    right: MatrixManTensor,
    m: int,
    k: int,
    n: int,
    *,
    bias: MatrixManTensor | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> MatrixManTensor:
    out_owner = operation_context.output_texture((m, n))
    runtime = operation_context.gl_runtime()
    left_layout = left._owner.layout
    right_layout = right._owner.layout
    bias_layout = bias._owner.layout if bias is not None else left_layout
    bias_rank = len(bias.shape) if bias is not None else 0
    params = (
        m, k, n, left_layout.kind, right_layout.kind,
        int(left._storage_offset), int(right._storage_offset),
        left_layout.texture_width, left_layout.texture_height,
        right_layout.texture_width, right_layout.texture_height,
        out_owner.layout.texture_width,
        bias_layout.kind, int(bias._storage_offset) if bias is not None else 0,
        bias_layout.texture_width, bias_layout.texture_height, bias_rank,
        kernels.glsl_float(alpha), kernels.glsl_float(beta),
    )
    if params not in runtime.packed_matmul_programs:
        diagnostics.trace(f"gm45.compile -> generic GEMM GLSL fragment shader params={params}")
        program = gm.make_program(_shader_source(params))
        runtime.packed_matmul_programs[params] = program
        runtime.packed_matmul_uniforms[params] = (
            gm.glGetUniformLocation(program, b"left_tex"),
            gm.glGetUniformLocation(program, b"right_tex"),
        )
        runtime.packed_matmul_bias_uniforms[params] = gm.glGetUniformLocation(program, b"bias_tex")
    program = runtime.packed_matmul_programs[params]
    left_loc, right_loc = runtime.packed_matmul_uniforms[params]
    diagnostics.trace(
        f"gm45.kernel -> generic GEMM {m}x{k} @ {k}x{n}: texture #{left._owner.texture} x "
        f"texture #{right._owner.texture} -> texture #{out_owner.texture}"
    )
    operation_context.attach_output(out_owner)
    operation_context.framebuffer_complete("gm45 framebuffer incomplete for GEMM")
    gm.glUseProgram(program)
    gm.glActiveTexture(gm.GL_TEXTURE0)
    gm.glBindTexture(gm.GL_TEXTURE_2D, left._owner.texture)
    gm.glUniform1i(left_loc, 0)
    gm.glActiveTexture(gm.GL_TEXTURE1)
    gm.glBindTexture(gm.GL_TEXTURE_2D, right._owner.texture)
    gm.glUniform1i(right_loc, 1)
    if bias is not None:
        gm.glActiveTexture(gm.GL_TEXTURE2)
        gm.glBindTexture(gm.GL_TEXTURE_2D, bias._owner.texture)
        gm.glUniform1i(runtime.packed_matmul_bias_uniforms[params], 2)
    operation_context.draw_fullscreen_quad()
    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after GEMM: 0x{err:04x}")
    return operation_context.tensor_from_owner(out_owner, (m, n))


def render_addmm(input_tensor, mat1, mat2, beta: float = 1.0, alpha: float = 1.0) -> MatrixManTensor:
    """Render beta*input + alpha*(mat1 @ mat2) in one GPU shader."""
    input_tensor = _prepare_gemm_operand(input_tensor, "input")
    mat1 = _prepare_gemm_operand(mat1, "mat1")
    mat2 = _prepare_gemm_operand(mat2, "mat2")
    m, k, n = _validate_gemm_inputs(mat1, mat2)
    if input_tensor.dtype != torch.float32:
        raise RuntimeError("gm45 addmm input must be a float32 MatrixManTensor")
    if input_tensor.dim() not in {1, 2}:
        raise RuntimeError("gm45 addmm input must be a 1D bias or 2D [M,N] tensor")
    expected = (n,) if input_tensor.dim() == 1 else (m, n)
    if tuple(int(value) for value in input_tensor.shape) != expected:
        raise RuntimeError(f"gm45 addmm input shape {tuple(input_tensor.shape)} must be {expected}")
    if input_tensor._owner.layout.kind not in {"packed_rgba", "matrix2d_red"}:
        raise RuntimeError("gm45 addmm input requires packed_rgba or matrix2d_red storage")
    if tuple(input_tensor._logical_strides) != contiguous_strides(tuple(input_tensor.shape)):
        raise RuntimeError("gm45 addmm input currently requires contiguous logical storage")
    if input_tensor._owner.layout.kind == "matrix2d_red" and input_tensor._storage_offset != 0:
        raise RuntimeError("gm45 addmm does not support nonzero offsets for matrix2d_red input storage")
    return _render_gemm(mat1, mat2, m, k, n, bias=input_tensor, alpha=float(alpha), beta=float(beta))
