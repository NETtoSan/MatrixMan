#!/usr/bin/env python3
"""
Tiny matrix addition and multiplication through the graphics pipeline.

This deliberately uses classic OpenGL rasterization:
  CPU matrices -> OpenGL float textures -> GLSL fragment shader
  -> framebuffer texture -> glReadPixels -> CPU verification

No CUDA, Torch, OpenCL, Vulkan, or compute shaders.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys

import numpy as np


def load_library(name: str) -> ctypes.CDLL:
    path = ctypes.util.find_library(name)
    if not path:
        raise RuntimeError(f"could not find {name}")
    return ctypes.CDLL(path)


sdl = load_library("SDL2")
gl = load_library("GL")


SDL_INIT_VIDEO = 0x00000020
SDL_WINDOW_OPENGL = 0x00000002
SDL_WINDOW_HIDDEN = 0x00000008
SDL_GL_CONTEXT_MAJOR_VERSION = 17
SDL_GL_CONTEXT_MINOR_VERSION = 18
SDL_GL_CONTEXT_PROFILE_MASK = 21
SDL_GL_CONTEXT_PROFILE_COMPATIBILITY = 0x0002

GL_FALSE = 0
GL_TRUE = 1
GL_TEXTURE_2D = 0x0DE1
GL_RGBA = 0x1908
GL_FLOAT = 0x1406
GL_RGBA32F = 0x8814
GL_NEAREST = 0x2600
GL_CLAMP_TO_EDGE = 0x812F
GL_TEXTURE_MIN_FILTER = 0x2801
GL_TEXTURE_MAG_FILTER = 0x2800
GL_TEXTURE_WRAP_S = 0x2802
GL_TEXTURE_WRAP_T = 0x2803
GL_COLOR_BUFFER_BIT = 0x00004000
GL_FRAMEBUFFER = 0x8D40
GL_COLOR_ATTACHMENT0 = 0x8CE0
GL_FRAMEBUFFER_COMPLETE = 0x8CD5
GL_VERTEX_SHADER = 0x8B31
GL_FRAGMENT_SHADER = 0x8B30
GL_COMPILE_STATUS = 0x8B81
GL_LINK_STATUS = 0x8B82
GL_INFO_LOG_LENGTH = 0x8B84
GL_QUADS = 0x0007
GL_TEXTURE0 = 0x84C0
GL_TEXTURE1 = 0x84C1
GL_TEXTURE2 = 0x84C2
GL_TEXTURE3 = 0x84C3
GL_TEXTURE4 = 0x84C4


sdl.SDL_Init.argtypes = [ctypes.c_uint32]
sdl.SDL_Init.restype = ctypes.c_int
sdl.SDL_Quit.argtypes = []
sdl.SDL_GL_SetAttribute.argtypes = [ctypes.c_int, ctypes.c_int]
sdl.SDL_GL_SetAttribute.restype = ctypes.c_int
sdl.SDL_CreateWindow.argtypes = [
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_uint32,
]
sdl.SDL_CreateWindow.restype = ctypes.c_void_p
sdl.SDL_DestroyWindow.argtypes = [ctypes.c_void_p]
sdl.SDL_GL_CreateContext.argtypes = [ctypes.c_void_p]
sdl.SDL_GL_CreateContext.restype = ctypes.c_void_p
sdl.SDL_GL_DeleteContext.argtypes = [ctypes.c_void_p]
sdl.SDL_GL_GetProcAddress.argtypes = [ctypes.c_char_p]
sdl.SDL_GL_GetProcAddress.restype = ctypes.c_void_p
sdl.SDL_GetError.restype = ctypes.c_char_p


def sdl_check(ok: bool, action: str) -> None:
    if not ok:
        error = sdl.SDL_GetError()
        raise RuntimeError(f"{action}: {error.decode() if error else 'unknown SDL error'}")


def proc(name: str, restype, *argtypes):
    addr = sdl.SDL_GL_GetProcAddress(name.encode("ascii"))
    if addr:
        return ctypes.CFUNCTYPE(restype, *argtypes)(addr)
    try:
        return gl_proc(name, restype, *argtypes)
    except AttributeError as exc:
        raise RuntimeError(f"OpenGL function unavailable: {name}") from exc


def gl_proc(name: str, restype, *argtypes):
    fn = getattr(gl, name)
    fn.restype = restype
    fn.argtypes = list(argtypes)
    return fn


glGetString = gl_proc("glGetString", ctypes.c_char_p, ctypes.c_uint)
glViewport = gl_proc("glViewport", None, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
glClearColor = gl_proc("glClearColor", None, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float)
glClear = gl_proc("glClear", None, ctypes.c_uint)
glGenTextures = gl_proc("glGenTextures", None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint))
glBindTexture = gl_proc("glBindTexture", None, ctypes.c_uint, ctypes.c_uint)
glTexParameteri = gl_proc("glTexParameteri", None, ctypes.c_uint, ctypes.c_uint, ctypes.c_int)
glTexImage2D = gl_proc(
    "glTexImage2D",
    None,
    ctypes.c_uint,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_void_p,
)
glActiveTexture = gl_proc("glActiveTexture", None, ctypes.c_uint)
glUseProgram = gl_proc("glUseProgram", None, ctypes.c_uint)
glBegin = gl_proc("glBegin", None, ctypes.c_uint)
glEnd = gl_proc("glEnd", None)
glFlush = gl_proc("glFlush", None)
glVertex2f = gl_proc("glVertex2f", None, ctypes.c_float, ctypes.c_float)
glReadPixels = gl_proc(
    "glReadPixels",
    None,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_void_p,
)
glFinish = gl_proc("glFinish", None)
glDeleteTextures = gl_proc("glDeleteTextures", None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint))
glGetError = gl_proc("glGetError", ctypes.c_uint)

glCreateShader = proc("glCreateShader", ctypes.c_uint, ctypes.c_uint)
glShaderSource = proc(
    "glShaderSource",
    None,
    ctypes.c_uint,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_int),
)
glCompileShader = proc("glCompileShader", None, ctypes.c_uint)
glGetShaderiv = proc("glGetShaderiv", None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_int))
glGetShaderInfoLog = proc(
    "glGetShaderInfoLog",
    None,
    ctypes.c_uint,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_char_p,
)
glCreateProgram = proc("glCreateProgram", ctypes.c_uint)
glAttachShader = proc("glAttachShader", None, ctypes.c_uint, ctypes.c_uint)
glLinkProgram = proc("glLinkProgram", None, ctypes.c_uint)
glGetProgramiv = proc("glGetProgramiv", None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_int))
glGetProgramInfoLog = proc(
    "glGetProgramInfoLog",
    None,
    ctypes.c_uint,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_char_p,
)
glDeleteShader = proc("glDeleteShader", None, ctypes.c_uint)
glDeleteProgram = proc("glDeleteProgram", None, ctypes.c_uint)
glGetUniformLocation = proc("glGetUniformLocation", ctypes.c_int, ctypes.c_uint, ctypes.c_char_p)
glUniform1i = proc("glUniform1i", None, ctypes.c_int, ctypes.c_int)
glGenFramebuffers = proc("glGenFramebuffers", None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint))
glBindFramebuffer = proc("glBindFramebuffer", None, ctypes.c_uint, ctypes.c_uint)
glFramebufferTexture2D = proc(
    "glFramebufferTexture2D",
    None,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_int,
)
glCheckFramebufferStatus = proc("glCheckFramebufferStatus", ctypes.c_uint, ctypes.c_uint)
glDeleteFramebuffers = proc("glDeleteFramebuffers", None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint))


VERTEX_SHADER = b"""
#version 120
void main()
{
    gl_Position = gl_Vertex;
}
"""


ADD_FRAGMENT_SHADER_TEMPLATE = """
#version 120
uniform sampler2D matrix_a;
uniform sampler2D matrix_b;

void main()
{
    /*
       The output framebuffer is N x N pixels.
       This fragment computes exactly one matrix element:

           C[row, col] = A[row, col] + B[row, col]

       gl_FragCoord.xy is the current pixel center, e.g. 0.5, 1.5, ...
       Dividing by N gives the matching normalized texture coordinate.
    */
    vec2 uv = gl_FragCoord.xy / float(N);
    float a = texture2D(matrix_a, uv).r;
    float b = texture2D(matrix_b, uv).r;

    gl_FragColor = vec4(a + b, 0.0, 0.0, 1.0);
}
"""


MUL_FRAGMENT_SHADER_TEMPLATE = """
#version 120
uniform sampler2D matrix_a;
uniform sampler2D matrix_b;

void main()
{
    /*
       Naive matrix multiplication in a fragment shader.

       The rasterizer launches one fragment per output pixel. We treat that
       pixel as one output element C[row, col], then do the full dot product
       for that element inside this GPU shader invocation:

           C[row, col] = sum(A[row, k] * B[k, col])

       This is educational, not fast. For tiny matrices, readback/context
       overhead dominates. The point is that the multiply-accumulate loop
       below runs in GLSL on the Intel GPU.
    */
    float col = floor(gl_FragCoord.x);
    float row = floor(gl_FragCoord.y);
    float acc = 0.0;

    for (int k = 0; k < N; ++k) {
        float fk = float(k);
        float a = texture2D(
            matrix_a,
            vec2((fk + 0.5) / float(N), (row + 0.5) / float(N))
        ).r;
        float b = texture2D(
            matrix_b,
            vec2((col + 0.5) / float(N), (fk + 0.5) / float(N))
        ).r;
        acc += a * b;
    }

    gl_FragColor = vec4(acc, 0.0, 0.0, 1.0);
}
"""


def shader_with_size(template: str, n: int) -> bytes:
    # GLSL 1.20 on this GM45 driver is happiest with a compile-time loop bound.
    return template.replace("N", str(n)).encode("ascii")


def compile_shader(kind: int, source: bytes) -> int:
    shader = glCreateShader(kind)
    source_p = ctypes.c_char_p(source)
    length = ctypes.c_int(len(source))
    glShaderSource(shader, 1, ctypes.byref(source_p), ctypes.byref(length))
    glCompileShader(shader)

    status = ctypes.c_int()
    glGetShaderiv(shader, GL_COMPILE_STATUS, ctypes.byref(status))
    if status.value != GL_TRUE:
        log_len = ctypes.c_int()
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, ctypes.byref(log_len))
        log = ctypes.create_string_buffer(max(log_len.value, 1))
        glGetShaderInfoLog(shader, len(log), None, log)
        raise RuntimeError(log.value.decode(errors="replace"))
    return shader


def make_program(fragment_source: bytes) -> int:
    vertex = compile_shader(GL_VERTEX_SHADER, VERTEX_SHADER)
    fragment = compile_shader(GL_FRAGMENT_SHADER, fragment_source)
    program = glCreateProgram()
    glAttachShader(program, vertex)
    glAttachShader(program, fragment)
    glLinkProgram(program)
    glDeleteShader(vertex)
    glDeleteShader(fragment)

    status = ctypes.c_int()
    glGetProgramiv(program, GL_LINK_STATUS, ctypes.byref(status))
    if status.value != GL_TRUE:
        log_len = ctypes.c_int()
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, ctypes.byref(log_len))
        log = ctypes.create_string_buffer(max(log_len.value, 1))
        glGetProgramInfoLog(program, len(log), None, log)
        raise RuntimeError(log.value.decode(errors="replace"))
    return program


def matrix_texture(matrix: np.ndarray) -> int:
    n = matrix.shape[0]

    # Store one scalar matrix element per texel in the red channel.
    # RGBA32F is used because this old GL path supports float RGBA textures
    # more consistently than single-channel float render/storage formats.
    data = np.zeros((n, n, 4), dtype=np.float32)
    data[:, :, 0] = matrix.astype(np.float32)

    texture = ctypes.c_uint()
    glGenTextures(1, ctypes.byref(texture))
    glBindTexture(GL_TEXTURE_2D, texture.value)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA32F,
        n,
        n,
        0,
        GL_RGBA,
        GL_FLOAT,
        data.ctypes.data_as(ctypes.c_void_p),
    )
    return texture.value


def empty_result_texture(n: int) -> int:
    texture = ctypes.c_uint()
    glGenTextures(1, ctypes.byref(texture))
    glBindTexture(GL_TEXTURE_2D, texture.value)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, n, n, 0, GL_RGBA, GL_FLOAT, None)
    return texture.value


def run_fragment_shader_matrix_op(a: np.ndarray, b: np.ndarray, fragment_source: bytes) -> np.ndarray:
    if a.shape != b.shape or len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("both inputs must be square matrices with the same shape")

    n = a.shape[0]

    # The only operation-specific arithmetic is in the fragment shader program.
    # Python uploads textures, draws a quad, and reads the resulting texture back.
    program = make_program(fragment_source)
    tex_a = matrix_texture(a)
    tex_b = matrix_texture(b)
    tex_out = empty_result_texture(n)
    fbo = ctypes.c_uint()

    glGenFramebuffers(1, ctypes.byref(fbo))
    glBindFramebuffer(GL_FRAMEBUFFER, fbo.value)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_out, 0)
    status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
    if status != GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(f"framebuffer is incomplete: 0x{status:04x}")

    glViewport(0, 0, n, n)
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)

    glUseProgram(program)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, tex_a)
    glUniform1i(glGetUniformLocation(program, b"matrix_a"), 0)
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, tex_b)
    glUniform1i(glGetUniformLocation(program, b"matrix_b"), 1)

    glBegin(GL_QUADS)
    glVertex2f(-1.0, -1.0)
    glVertex2f(1.0, -1.0)
    glVertex2f(1.0, 1.0)
    glVertex2f(-1.0, 1.0)
    glEnd()
    glFinish()

    pixels = np.zeros((n, n, 4), dtype=np.float32)
    glReadPixels(0, 0, n, n, GL_RGBA, GL_FLOAT, pixels.ctypes.data_as(ctypes.c_void_p))

    err = glGetError()
    if err:
        raise RuntimeError(f"OpenGL error: 0x{err:04x}")

    glUseProgram(0)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    textures = (ctypes.c_uint * 3)(tex_a, tex_b, tex_out)
    glDeleteTextures(3, textures)
    glDeleteFramebuffers(1, ctypes.byref(fbo))
    glDeleteProgram(program)

    return pixels[:, :, 0]


def add_on_gpu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    source = shader_with_size(ADD_FRAGMENT_SHADER_TEMPLATE, a.shape[0])
    return run_fragment_shader_matrix_op(a, b, source)


def multiply_on_gpu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    source = shader_with_size(MUL_FRAGMENT_SHADER_TEMPLATE, a.shape[0])
    return run_fragment_shader_matrix_op(a, b, source)


def print_check(name: str, gpu: np.ndarray, cpu: np.ndarray) -> None:
    matches = np.allclose(gpu, cpu, rtol=1e-5, atol=1e-5)
    print(f"\n{name} GPU fragment shader result:")
    print(gpu)
    print(f"\n{name} CPU NumPy reference:")
    print(cpu)
    print("max abs diff:", float(np.max(np.abs(gpu - cpu))))
    print("matches CPU reference:", bool(matches))


def main() -> int:
    sdl_check(sdl.SDL_Init(SDL_INIT_VIDEO) == 0, "SDL_Init failed")
    window = None
    context = None

    try:
        sdl.SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2)
        sdl.SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1)
        sdl.SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_COMPATIBILITY)

        window = sdl.SDL_CreateWindow(
            b"gpumatrix",
            0,
            0,
            64,
            64,
            SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN,
        )
        sdl_check(bool(window), "SDL_CreateWindow failed")

        context = sdl.SDL_GL_CreateContext(window)
        sdl_check(bool(context), "SDL_GL_CreateContext failed")

        renderer = glGetString(0x1F01)
        version = glGetString(0x1F02)
        print("OpenGL renderer:", renderer.decode() if renderer else "unknown")
        print("OpenGL version: ", version.decode() if version else "unknown")

        n = 4
        a = np.arange(1, n * n + 1, dtype=np.float32).reshape(n, n)
        b = np.arange(n * n, 0, -1, dtype=np.float32).reshape(n, n)

        np.set_printoptions(precision=3, suppress=True)
        print("\nA:")
        print(a)
        print("\nB:")
        print(b)

        add_gpu = add_on_gpu(a, b)
        print_check("C = A + B", add_gpu, a + b)

        mul_gpu = multiply_on_gpu(a, b)
        print_check("C = A x B", mul_gpu, a @ b)
        return 0
    finally:
        if context:
            sdl.SDL_GL_DeleteContext(context)
        if window:
            sdl.SDL_DestroyWindow(window)
        sdl.SDL_Quit()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
