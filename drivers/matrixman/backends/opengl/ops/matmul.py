"""OpenGL matrix multiplication operation boundary."""

from __future__ import annotations

import torch

from .. import diagnostics, gpumatrix as gm, operation_context, resources
from ..storage import StorageLayout
from ..tensor import Gm45Tensor, owner_from_texture


def new_empty_matrix_texture(n: int):
    layout = StorageLayout("matrix2d_red", n, n, n * n)
    texture = resources.allocate_matrix_texture(n)
    owner = owner_from_texture(texture, layout)
    diagnostics.trace(f"gm45.texture_alloc -> matrix output texture #{owner.texture} ({n}x{n}, RGBA32F red channel)")
    return owner


def render_matrix_binary(kind: str, left: Gm45Tensor, right: Gm45Tensor, alpha: float = 1.0) -> Gm45Tensor:
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


def render_matmul(left: Gm45Tensor, right: Gm45Tensor) -> Gm45Tensor:
    if not isinstance(left, Gm45Tensor) or not isinstance(right, Gm45Tensor):
        raise RuntimeError("gm45 matmul requires both inputs to be gm45 tensors")
    if left.shape != right.shape:
        raise RuntimeError("gm45 matmul requires equal shapes")
    return render_matrix_binary("matmul", left, right)
