# MatrixMan OpenGL backend

## Role and framework integration

MatrixMan is an experimental PyTorch `PrivateUse1` backend for GPUs that lack
a usable native PyTorch compute backend. The current working backend is
OpenGL. PyTorch's `PrivateUse1` device name is registered as `matrixman`. Names
such as `MatrixManTensor`, `matrixman:0`, and `to_device()` are backend-neutral;
`Gm45Tensor` and `to_gm45()` remain deprecated compatibility names even though
the implementation is no longer limited to GM45 hardware.

The dispatch path is:

```text
PyTorch ATen operation
  -> PrivateUse1 dispatch bridge
  -> OpenGL backend operation
  -> packed texture/framebuffer resources
  -> GLSL fragment shader
  -> MatrixManTensor result
```

`MatrixManTensor` stores tensor metadata and an owner for an OpenGL texture. Views,
strides, offsets, and shape transformations are tracked as metadata where
possible. Arithmetic stays on the selected backend. If an operation is not
implemented, MatrixMan fails explicitly; it does not silently perform tensor
arithmetic on the CPU.

## Storage model

The OpenGL backend uses float textures, normally RGBA32F. A contiguous tensor
is flattened in ordinary NCHW order and packs four consecutive float32
scalars into each texel:

```text
logical scalar index 0,1,2,3 -> texel.r,g,b,a
logical scalar index 4,5,6,7 -> next texel.r,g,b,a
```

RGBA lanes are packing lanes, not color channels or image semantics. For
`[N,C,H,W]`, the scalar index is derived from logical NCHW coordinates,
divided by four to find the texel, and reduced modulo four to select the
component. The linear texel stream is laid out in a 2D atlas.

Upload copies CPU float32 data into the RGBA32F texture. Supported operators
create and render into additional textures, so intermediate tensors remain
GPU-resident. CPU work is used for metadata, dispatch, uploads, diagnostics,
and explicit final readback. Calling `.cpu()` performs the intentional
GPU-to-CPU readback and synchronization point.

## Runtime architecture

The OpenGL implementation is split under
`drivers/matrixman/backends/opengl/`:

| Module | Responsibility |
| --- | --- |
| `backend.py` | Backend façade, lifecycle, device information, and public hooks. |
| `runtime.py` | SDL/OpenGL context and context-owned caches/resources. |
| `resources.py` | Texture allocation, upload/readback, scratch textures, parameter caches. |
| `../tensor.py` | Backend-neutral `MatrixManTensor` wrapper and metadata helpers. |
| `tensor.py` | OpenGL texture ownership and metadata-aware readback. |
| `metadata.py` | Shape, stride, offset, and view metadata. |
| `factories.py` | PyTorch `PrivateUse1` factories and registration. |
| `dispatch.py` | ATen dispatch bridge. |
| `kernels.py`, `render.py`, `operation_context.py` | Shared shader, framebuffer, and operation services. |
| `convolution.py` | Conv2D shaders, physical tiling, synchronization, and consolidation. |
| `ops/` | Arithmetic, activation, normalization, concat, pooling, resize, softmax, matmul, and postprocessing. |
| `diagnostics.py` | Trace and unsupported-operation reporting. |
| `profiling.py` | Opt-in CPU counters and OpenGL timer-query profiling. |

`implementation.py` is a compatibility/re-export façade, not the normal
runtime implementation. `backends/cuda/` and `backends/opencl/` are
placeholders and remain unimplemented.

## Conv2D execution

Conv2D is implemented as a fragment shader. Each output fragment writes four
consecutive flattened output scalars. The baseline shader computes those
scalars independently and uses flattened scalar addressing for input and
weight texture reads.

If the physical output atlas exceeds the configured physical dimensions,
Conv2D renders into a grid of smaller physical textures. After all tiles are
complete, the tile textures are consolidated into the normal output atlas.
If the complete output atlas fits within the physical limit, Conv2D takes the
direct one-shot path and no consolidation is needed.

The synchronization controls are:

```text
MATRIXMAN_TILE_LIMIT=256       # default
MATRIXMAN_TILE_SYNC=per_tile   # default
```

`per_tile` is the validated GM45-safe production behavior. Other modes and a
larger limit such as 512 are experiments measured on newer test GPUs. They
must not be treated as safe GM45 defaults. The optional
`MATRIXMAN_SKIP_PRE_CONSOLIDATION_SYNC=1` experiment skips only the audited
pre-consolidation barrier; it does not change the production default.

## Physical tile validation diagnostic

To validate physical Conv render sizes on the current OpenGL GPU and driver,
run:

```text
python -m drivers.matrixman.diagnostics.opengl_tile_limit
```

The invocation is identical on Linux and Windows PowerShell. The diagnostic
reports OpenGL limits, estimates memory before each candidate, runs real
MatrixMan Conv workloads against a CPU reference, and isolates each candidate
in a child process with a timeout. It reports the largest tile size validated
by that run; this is diagnostic evidence, not a universal recommendation. The
YOLO-shaped `[1,64,80,80]` / `320x320` packed Conv case is included to show the
256 tiled versus 512 direct behavior where the GPU supports both.

Optional overrides use the same Python arguments on both platforms, for
example `--sizes 256,512,1024 --max-size 1024 --timeout 30`. The existing
GM45-safe production defaults remain `MATRIXMAN_TILE_LIMIT=256` and
`MATRIXMAN_TILE_SYNC=per_tile`.

## Conv tiling demonstration

For a focused correctness check of the real OpenGL Conv2D path, run:

```text
python -m drivers.matrixman.diagnostics.opengl_conv_tiling_demo --tile-size 16
python -m drivers.matrixman.diagnostics.opengl_conv_tiling_demo --tile-size 256
python -m drivers.matrixman.diagnostics.opengl_conv_tiling_demo --tile-size 512 --compare-direct
```

The diagnostic temporarily overrides `matrixman.config.tileLimit`, executes
`torch.nn.functional.conv2d` with a MatrixMan tensor, and restores the prior
configuration before exiting. Its default workload is `[1,8,32,32]` with an
`[8,8,3,3]` weight and bias. Use `--preset yolo80` for the historical
`[1,64,80,80]` to `[1,64,80,80]` workload. The result remains on the real
OpenGL packed-texture/tiled shader path until the explicit validation readback.
The report obtains tile coordinates and physical dimensions from existing
diagnostic Conv telemetry and compares the readback with an independent NumPy
float32 reference using `rtol=5e-4` and `atol=5e-4`.

## Compatibility and profiling

Startup probes OpenGL availability and selects the OpenGL backend when a
usable context can be created. Compatibility tests exercise texture upload,
shader arithmetic, supported operators, Conv2D, and tile/consolidation paths.
The active driver, OpenGL/GLSL versions, framebuffer behavior, texture limits,
and actual workload all affect compatibility.

`MATRIXMAN_PROFILE=1` enables CPU dispatch timing and OpenGL counters. The
optional `MATRIXMAN_GPU_TIMING=1` control uses deferred OpenGL timer queries
(`GL_TIME_ELAPSED` through timer-query extensions) when supported. Queries are
collected at later synchronization points; the profiler does not add a
`glFinish()` after every operator. It reports CPU submission time separately
from GPU elapsed time and synchronization time.

## Step 10B spatial reuse

The opt-in control is:

```text
MATRIXMAN_CONV_SPATIAL_REUSE=1
```

Naïve channel-vectorization is invalid for the current flattened NCHW
packing: four contiguous values in a texture are not generally four input
channels. For suitable tensors, however, the four output lanes represent
four adjacent spatial x positions. A 3×3 stride-1 convolution can then share
the six horizontal input positions needed by those outputs. The experimental
shader loads a shared 3×6 spatial neighborhood and reuses the three weight
values across all four output lanes.

Eligibility is deliberately conservative: contiguous batch-1 NCHW storage,
groups=1, float32 packed storage, 3×3 kernel, stride=1, padding=1,
dilation=1, output width divisible by four, direct rendering within the
physical limit, and a fully occupied output atlas. Left/right row-boundary
fragments use baseline GPU Conv semantics. Unsupported cases fall back to
baseline GPU Conv, never CPU arithmetic.

The initial implementation had an edge-window indexing error caused by
clamping the shared neighborhood origin. The final design separates the
logical window origin and boundary eligibility. The dominant
`[1,64,80,80]` Conv and the tiny deterministic Conv now report zero
mismatches against the baseline within the existing tolerance.

The optimization remains opt-in and device-dependent. Its measured benefit
must be established independently for each driver and workload.
