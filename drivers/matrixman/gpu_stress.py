#!/usr/bin/env python3
"""
60-second legacy OpenGL fragment-shader matrix stress test.

This is intentionally an educational GPGPU experiment for old Intel GM45 /
GMA 4500MHD style hardware:

  CPU creates matrices
  -> matrices are uploaded to OpenGL float textures
  -> GLSL 1.20 fragment shaders do C = A + B and C = A x B
  -> results are written into framebuffer-attached textures
  -> occasional readback checks correctness against NumPy

The tested GPU operations never fall back to CPU arithmetic. NumPy is used only
for periodic validation after the GPU has produced a result.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import shutil
import subprocess
import time

import numpy as np

from . import gpumatrix as gm


GL_VENDOR = 0x1F00
GL_RENDERER = 0x1F01
GL_VERSION = 0x1F02
GL_EXTENSIONS = 0x1F03
GL_SHADING_LANGUAGE_VERSION = 0x8B8C
GL_MAX_TEXTURE_SIZE = 0x0D33

glGetIntegerv = gm.gl_proc("glGetIntegerv", None, ctypes.c_uint, ctypes.POINTER(ctypes.c_int))
glTexSubImage2D = gm.gl_proc(
    "glTexSubImage2D",
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


ADD_SHADER = """
#version 120
uniform sampler2D left_tex;
uniform sampler2D right_tex;

void main()
{
    /*
       One fragment equals one output matrix element.
       This shader is the arithmetic kernel for:

           out[row, col] = left[row, col] + right[row, col]
    */
    vec2 uv = gl_FragCoord.xy / float(N);
    float left_value = texture2D(left_tex, uv).r;
    float right_value = texture2D(right_tex, uv).r;
    gl_FragColor = vec4(left_value + right_value, 0.0, 0.0, 1.0);
}
"""


MUL_SHADER = """
#version 120
uniform sampler2D left_tex;
uniform sampler2D right_tex;

void main()
{
    /*
       One fragment equals one output matrix element.
       This shader is the arithmetic kernel for naive matrix multiplication:

           out[row, col] = sum(left[row, k] * right[k, col])

       The loop runs inside GLSL on the programmable graphics engine.
    */
    float col = floor(gl_FragCoord.x);
    float row = floor(gl_FragCoord.y);
    float acc = 0.0;

    for (int k = 0; k < N; ++k) {
        float fk = float(k);
        float left_value = texture2D(
            left_tex,
            vec2((fk + 0.5) / float(N), (row + 0.5) / float(N))
        ).r;
        float right_value = texture2D(
            right_tex,
            vec2((col + 0.5) / float(N), (fk + 0.5) / float(N))
        ).r;
        acc += left_value * right_value;
    }

    gl_FragColor = vec4(acc, 0.0, 0.0, 1.0);
}
"""


def shader_source(template: str, n: int) -> bytes:
    # GLSL 1.20 on this driver supports simple constant-bounded loops reliably.
    return template.replace("N", str(n)).encode("ascii")


def gl_string(enum: int) -> str:
    value = gm.glGetString(enum)
    return value.decode(errors="replace") if value else "unavailable"


def gl_int(enum: int) -> int:
    value = ctypes.c_int()
    glGetIntegerv(enum, ctypes.byref(value))
    return value.value


def extension_supported(name: str) -> bool:
    return name in gl_string(GL_EXTENSIONS).split()


def print_capabilities() -> None:
    print("OpenGL capability probe:")
    print("  vendor:  ", gl_string(GL_VENDOR))
    print("  renderer:", gl_string(GL_RENDERER))
    print("  version: ", gl_string(GL_VERSION))
    print("  GLSL:    ", gl_string(GL_SHADING_LANGUAGE_VERSION))
    print("  max texture size:", gl_int(GL_MAX_TEXTURE_SIZE))
    for ext in [
        "GL_ARB_texture_float",
        "GL_ATI_texture_float",
        "GL_EXT_framebuffer_object",
        "GL_ARB_fragment_shader",
        "GL_ARB_shader_objects",
    ]:
        print(f"  {ext}: {extension_supported(ext)}")


def print_telemetry_availability() -> None:
    print("\nGPU telemetry availability:")
    print("  /sys/class/drm: available, but this sandbox exposes limited card details")
    print("  /sys/class/hwmon:", "available" if list_hwmon_names() else "no readable hwmon devices found")
    print("  intel_gpu_top:", shutil.which("intel_gpu_top") or "not found")
    print("  sensors:", shutil.which("sensors") or "not found")

    gpu_top = sample_intel_gpu_top()
    if gpu_top:
        print("  intel_gpu_top sample:")
        for line in gpu_top.splitlines()[:6]:
            print("   ", line)
    else:
        print("  intel_gpu_top sample: unavailable without extra permissions or unsupported counters")


def list_hwmon_names() -> list[str]:
    names = []
    try:
        for path in sorted(__import__("pathlib").Path("/sys/class/hwmon").glob("hwmon*/name")):
            try:
                names.append(f"{path.parent.name}:{path.read_text().strip()}")
            except OSError:
                pass
    except OSError:
        pass
    return names


def sample_intel_gpu_top() -> str | None:
    tool = shutil.which("intel_gpu_top")
    if not tool:
        return None
    try:
        # This often needs elevated permissions. Failure is informational only.
        result = subprocess.run(
            ["timeout", "1", tool, "-s", "250"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = result.stdout.strip()
    return text if result.returncode == 0 and text else None


def matrix_to_rgba(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    rgba = np.zeros((n, n, 4), dtype=np.float32)
    rgba[:, :, 0] = matrix.astype(np.float32)
    rgba[:, :, 3] = 1.0
    return rgba


def create_texture(n: int, initial: np.ndarray | None = None) -> int:
    texture = ctypes.c_uint()
    gm.glGenTextures(1, ctypes.byref(texture))
    gm.glBindTexture(gm.GL_TEXTURE_2D, texture.value)
    gm.glTexParameteri(gm.GL_TEXTURE_2D, gm.GL_TEXTURE_MIN_FILTER, gm.GL_NEAREST)
    gm.glTexParameteri(gm.GL_TEXTURE_2D, gm.GL_TEXTURE_MAG_FILTER, gm.GL_NEAREST)
    gm.glTexParameteri(gm.GL_TEXTURE_2D, gm.GL_TEXTURE_WRAP_S, gm.GL_CLAMP_TO_EDGE)
    gm.glTexParameteri(gm.GL_TEXTURE_2D, gm.GL_TEXTURE_WRAP_T, gm.GL_CLAMP_TO_EDGE)
    data = matrix_to_rgba(initial) if initial is not None else None
    ptr = data.ctypes.data_as(ctypes.c_void_p) if data is not None else None
    gm.glTexImage2D(gm.GL_TEXTURE_2D, 0, gm.GL_RGBA32F, n, n, 0, gm.GL_RGBA, gm.GL_FLOAT, ptr)
    return texture.value


def update_texture(texture: int, matrix: np.ndarray) -> None:
    data = matrix_to_rgba(matrix)
    gm.glBindTexture(gm.GL_TEXTURE_2D, texture)
    glTexSubImage2D(
        gm.GL_TEXTURE_2D,
        0,
        0,
        0,
        matrix.shape[0],
        matrix.shape[1],
        gm.GL_RGBA,
        gm.GL_FLOAT,
        data.ctypes.data_as(ctypes.c_void_p),
    )


def read_texture(texture: int, fbo: ctypes.c_uint, n: int) -> np.ndarray:
    gm.glBindFramebuffer(gm.GL_FRAMEBUFFER, fbo.value)
    gm.glFramebufferTexture2D(gm.GL_FRAMEBUFFER, gm.GL_COLOR_ATTACHMENT0, gm.GL_TEXTURE_2D, texture, 0)
    pixels = np.zeros((n, n, 4), dtype=np.float32)
    gm.glReadPixels(0, 0, n, n, gm.GL_RGBA, gm.GL_FLOAT, pixels.ctypes.data_as(ctypes.c_void_p))
    return pixels[:, :, 0]


def make_inputs(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.uniform(-0.5, 0.5, (n, n)).astype(np.float32)

    # Keep B numerically tame so repeated feedback does not explode. The matrix
    # is still dense enough to exercise texture reads and multiply-accumulate.
    b = rng.uniform(-0.25 / n, 0.25 / n, (n, n)).astype(np.float32)
    b += np.eye(n, dtype=np.float32) * 0.25
    return a, b


class StressState:
    def __init__(self, n: int):
        self.n = n
        self.add_program = gm.make_program(shader_source(ADD_SHADER, n))
        self.mul_program = gm.make_program(shader_source(MUL_SHADER, n))
        self.left_add = gm.glGetUniformLocation(self.add_program, b"left_tex")
        self.right_add = gm.glGetUniformLocation(self.add_program, b"right_tex")
        self.left_mul = gm.glGetUniformLocation(self.mul_program, b"left_tex")
        self.right_mul = gm.glGetUniformLocation(self.mul_program, b"right_tex")
        self.fbo = ctypes.c_uint()
        gm.glGenFramebuffers(1, ctypes.byref(self.fbo))

        a, b = make_inputs(n, seed=1)
        self.cpu_b = b
        self.tex_a = create_texture(n, a)
        self.tex_b = create_texture(n, b)
        self.tex_c = create_texture(n)
        self.tex_d = create_texture(n)
        self.tex_e = create_texture(n)
        self.tex_f = create_texture(n)
        self.textures = [self.tex_a, self.tex_b, self.tex_c, self.tex_d, self.tex_e, self.tex_f]

        gm.glViewport(0, 0, n, n)
        status = self.check_target(self.tex_c)
        if status != gm.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"framebuffer incomplete for {n}x{n}: 0x{status:04x}")

    def check_target(self, out_tex: int) -> int:
        gm.glBindFramebuffer(gm.GL_FRAMEBUFFER, self.fbo.value)
        gm.glFramebufferTexture2D(gm.GL_FRAMEBUFFER, gm.GL_COLOR_ATTACHMENT0, gm.GL_TEXTURE_2D, out_tex, 0)
        return gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER)

    def render_binary(self, program: int, left_loc: int, right_loc: int, left_tex: int, right_tex: int, out_tex: int) -> None:
        gm.glBindFramebuffer(gm.GL_FRAMEBUFFER, self.fbo.value)
        gm.glFramebufferTexture2D(gm.GL_FRAMEBUFFER, gm.GL_COLOR_ATTACHMENT0, gm.GL_TEXTURE_2D, out_tex, 0)
        gm.glUseProgram(program)

        gm.glActiveTexture(gm.GL_TEXTURE0)
        gm.glBindTexture(gm.GL_TEXTURE_2D, left_tex)
        gm.glUniform1i(left_loc, 0)

        gm.glActiveTexture(gm.GL_TEXTURE1)
        gm.glBindTexture(gm.GL_TEXTURE_2D, right_tex)
        gm.glUniform1i(right_loc, 1)

        gm.glBegin(gm.GL_QUADS)
        gm.glVertex2f(-1.0, -1.0)
        gm.glVertex2f(1.0, -1.0)
        gm.glVertex2f(1.0, 1.0)
        gm.glVertex2f(-1.0, 1.0)
        gm.glEnd()

    def iteration(self) -> None:
        # C = A x B
        self.render_binary(self.mul_program, self.left_mul, self.right_mul, self.tex_a, self.tex_b, self.tex_c)
        # D = C + A
        self.render_binary(self.add_program, self.left_add, self.right_add, self.tex_c, self.tex_a, self.tex_d)
        # E = D x B
        self.render_binary(self.mul_program, self.left_mul, self.right_mul, self.tex_d, self.tex_b, self.tex_e)
        # F = E + C
        self.render_binary(self.add_program, self.left_add, self.right_add, self.tex_e, self.tex_c, self.tex_f)

        # Feed F into the next iteration as A without copying. The old A texture
        # becomes the next spare output texture for F.
        self.tex_a, self.tex_f = self.tex_f, self.tex_a

    def validate_next_iteration(self) -> tuple[float, bool]:
        # Read current A only for validation. The main loop otherwise avoids
        # CPU-GPU transfers.
        current_a = read_texture(self.tex_a, self.fbo, self.n)
        expected_c = current_a @ self.cpu_b
        expected_d = expected_c + current_a
        expected_e = expected_d @ self.cpu_b
        expected_f = expected_e + expected_c

        self.iteration()
        gm.glFinish()
        actual_f = read_texture(self.tex_a, self.fbo, self.n)
        max_error = float(np.max(np.abs(actual_f - expected_f)))
        valid = bool(np.allclose(actual_f, expected_f, rtol=5e-4, atol=5e-4))
        return max_error, valid

    def regenerate_inputs(self, seed: int) -> None:
        a, b = make_inputs(self.n, seed)
        update_texture(self.tex_a, a)
        update_texture(self.tex_b, b)
        self.cpu_b = b

    def gl_error(self) -> int:
        return gm.glGetError()

    def cleanup(self) -> None:
        gm.glUseProgram(0)
        gm.glBindFramebuffer(gm.GL_FRAMEBUFFER, 0)
        textures = (ctypes.c_uint * len(self.textures))(*self.textures)
        gm.glDeleteTextures(len(self.textures), textures)
        gm.glDeleteFramebuffers(1, ctypes.byref(self.fbo))
        gm.glDeleteProgram(self.add_program)
        gm.glDeleteProgram(self.mul_program)


def create_state_with_fallback(preferred: int) -> StressState:
    for n in [preferred, 128] if preferred != 128 else [128]:
        try:
            state = StressState(n)
            start = time.monotonic()
            state.iteration()
            gm.glFinish()
            warmup_seconds = time.monotonic() - start
            err = state.gl_error()
            if err:
                raise RuntimeError(f"OpenGL error during warmup: 0x{err:04x}")
            if n > 128 and warmup_seconds > 2.0:
                print(f"{n}x{n} warmup took {warmup_seconds:.2f}s; falling back to 128x128 for steadier reporting")
                state.cleanup()
                continue
            print(f"Using matrix size: {n}x{n} (warmup iteration {warmup_seconds:.3f}s)")
            return state
        except Exception as exc:
            print(f"{n}x{n} setup/warmup failed: {exc}")
            try:
                state.cleanup()
            except Exception:
                pass
    raise RuntimeError("no stable matrix size found")


def run_stress(seconds: float, preferred_size: int, validate_interval: float, regen_interval: float) -> int:
    print_capabilities()
    print_telemetry_availability()

    state = create_state_with_fallback(preferred_size)
    n = state.n
    started = time.monotonic()
    deadline = started + seconds
    next_report = started + 1.0
    next_validate = started + validate_interval
    next_regen = started + regen_interval

    iterations = 0
    matmuls = 0
    shader_ops = 0
    gl_errors_seen = False
    last_validation = "not yet checked"
    max_observed_error = 0.0
    seed = 2

    try:
        while True:
            now = time.monotonic()
            if now >= deadline:
                break

            if now >= next_validate:
                max_error, valid = state.validate_next_iteration()
                max_observed_error = max(max_observed_error, max_error)
                last_validation = f"valid={valid}, max_abs_error={max_error:.6g}"
                next_validate += validate_interval
            else:
                state.iteration()
                # This stress test reports completed GPU work, not merely GL
                # commands queued by the CPU. Finishing every iteration is less
                # throughput-friendly, but it gives honest stability counters
                # on this old OpenGL stack.
                gm.glFinish()

            iterations += 1
            matmuls += 2
            shader_ops += 4

            if now >= next_regen:
                state.regenerate_inputs(seed)
                seed += 1
                next_regen += regen_interval

            now = time.monotonic()
            if now >= next_report:
                gl_error = state.gl_error()
                if gl_error:
                    gl_errors_seen = True
                    gl_error_text = f"0x{gl_error:04x}"
                else:
                    gl_error_text = "none"

                elapsed = now - started
                matmuls_per_sec = matmuls / elapsed
                iterations_per_sec = iterations / elapsed
                gflops = matmuls_per_sec * (2.0 * n * n * n) / 1e9
                print(
                    f"t={elapsed:6.2f}s size={n} ops={shader_ops} "
                    f"matmul/s={matmuls_per_sec:7.2f} iter/s={iterations_per_sec:7.2f} "
                    f"est_GFLOPS={gflops:7.3f} correctness={last_validation} gl_errors={gl_error_text}",
                    flush=True,
                )
                next_report += 1.0

        gm.glFinish()
        total_runtime = time.monotonic() - started
        avg_matmuls = matmuls / total_runtime if total_runtime > 0 else math.nan
        avg_iterations = iterations / total_runtime if total_runtime > 0 else math.nan
        avg_gflops = avg_matmuls * (2.0 * n * n * n) / 1e9

        print("\nFinal stress-test summary:")
        print(f"  total runtime: {total_runtime:.3f}s")
        print(f"  matrix size: {n}x{n}")
        print(f"  total matmuls: {matmuls}")
        print(f"  total shader operations: {shader_ops}")
        print(f"  average matmuls/sec: {avg_matmuls:.3f}")
        print(f"  average iterations/sec: {avg_iterations:.3f}")
        print(f"  estimated average GFLOPS: {avg_gflops:.6f}")
        print(f"  maximum observed numerical error: {max_observed_error:.6g}")
        print(f"  last correctness status: {last_validation}")
        print(f"  any OpenGL errors: {gl_errors_seen}")
    finally:
        state.cleanup()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Legacy OpenGL GLSL matrix stress test")
    parser.add_argument("--seconds", type=float, default=60.0, help="stress duration, default: 60")
    parser.add_argument("--size", type=int, default=256, help="preferred square matrix size, default: 256")
    parser.add_argument("--validate-interval", type=float, default=5.0, help="seconds between correctness readbacks")
    parser.add_argument("--regen-interval", type=float, default=15.0, help="seconds between CPU-side input refreshes")
    args = parser.parse_args()

    gm.sdl_check(gm.sdl.SDL_Init(gm.SDL_INIT_VIDEO) == 0, "SDL_Init failed")
    window = None
    context = None
    try:
        gm.sdl.SDL_GL_SetAttribute(gm.SDL_GL_CONTEXT_MAJOR_VERSION, 2)
        gm.sdl.SDL_GL_SetAttribute(gm.SDL_GL_CONTEXT_MINOR_VERSION, 1)
        gm.sdl.SDL_GL_SetAttribute(gm.SDL_GL_CONTEXT_PROFILE_MASK, gm.SDL_GL_CONTEXT_PROFILE_COMPATIBILITY)
        window = gm.sdl.SDL_CreateWindow(
            b"gpu_stress",
            0,
            0,
            64,
            64,
            gm.SDL_WINDOW_OPENGL | gm.SDL_WINDOW_HIDDEN,
        )
        gm.sdl_check(bool(window), "SDL_CreateWindow failed")
        context = gm.sdl.SDL_GL_CreateContext(window)
        gm.sdl_check(bool(context), "SDL_GL_CreateContext failed")
        return run_stress(args.seconds, args.size, args.validate_interval, args.regen_interval)
    finally:
        if context:
            gm.sdl.SDL_GL_DeleteContext(context)
        if window:
            gm.sdl.SDL_DestroyWindow(window)
        gm.sdl.SDL_Quit()


if __name__ == "__main__":
    raise SystemExit(main())
