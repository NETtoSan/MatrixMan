# MatrixMan testing

Run commands from the repository root. MatrixMan prefers CUDA when the legacy
CUDA probe succeeds, and falls back to OpenGL when CUDA is unavailable:

```text
MatrixMan probe:
  CUDA: available
    device: NVIDIA GeForce GT 720M
    compute capability: 2.1
  OpenGL: available
  OpenCL: not implemented
MatrixMan selected: CUDA
```

An explicit OpenGL override remains available even when CUDA is detected:

```text
MATRIXMAN_BACKEND=opengl python3 -m drivers.matrixman --check
MatrixMan selected: OpenGL
```

High-level operation tracing is independent of backend selection and profiling
and is disabled by default:

```python
from drivers import matrixman
matrixman.trace = True
```

The environment equivalent is `MATRIXMAN_TRACE=1` (use `0` to disable it).

The OpenGL tests require SDL2, a usable OpenGL context, floating-point texture
and framebuffer support, and a graphical environment. A headless failure such as
`SDL_Init failed: No available video device` is an environment failure, not a
numerical MatrixMan result.

## Test layers

| Layer | Entry point | Purpose |
| --- | --- | --- |
| Public demo | `demo/main-tracking.py` | User-facing VisDrone YOLO demonstration. |
| Benchmark | `drivers/matrixman/benchmarks/yolo_benchmark.py` | Warmed performance runs, profiling, GPU timing, and JSON results. |
| Diagnostics | `drivers/matrixman/diagnostics/` | Focused correctness, compatibility, address, and stress tests. |

The public demo deliberately does not expose benchmark warm-up, profiling
internals, GPU tuning, or diagnostic checkpoint options.

### Backend versus frontend diagnostics

Diagnostic location describes the layer being tested; it does not imply that
every operator needs two copies of a test.

Directory placement describes the layer actually tested, not merely the
backend eventually selected.

Modules named `matrixman_*` are frontend diagnostics and must keep numerical
validation independent of backend-specific telemetry. In particular, optional
OpenGL unsupported-operation reports are omitted on CUDA; their absence is not
an operation failure. Run portable checks explicitly against each implemented
backend when hardware is available:

```bash
MATRIXMAN_BACKEND=cuda python3 -m drivers.matrixman.diagnostics.matrixman_conv_demo
MATRIXMAN_BACKEND=opengl python3 -m drivers.matrixman.diagnostics.matrixman_conv_demo
```

- `drivers/matrixman/backends/cuda/<op>.py` is a CUDA backend diagnostic when
  it is run as an executable module. It may inspect `CudaExecutionBackend`,
  embedded PTX, CUDA pointers, allocation ownership, logical strides, storage
  offsets, and explicit CUDA Driver API readback.
- `drivers/matrixman/diagnostics/matrixman_<op>_demo.py` is a frontend
  diagnostic. It exercises the public PyTorch/ATen operation through
  PrivateUse1, `MatrixManTensor`, dispatch, and the selected backend. It should
  remain backend-neutral unless its purpose is explicitly backend-specific;
  backend metadata may be displayed conditionally.

The current placement is intentional. CUDA-focused checks remain beside the
CUDA implementation, while the recently renamed `matrixman_div_demo.py` and
`matrixman_mul_demo.py` are frontend checks inherited from the former generic
operator demo family. Do not move them or add duplicate CUDA copies merely for
directory symmetry during CUDA bring-up. Revisit duplication and possible
diagnostics subdirectories in a separate layering audit after the YOLO forward
path reaches a stable endpoint.

The direct CUDA backend diagnostics currently include `add.py`, `sub.py`,
`mul.py`, `div.py`, `batch_norm.py`, `cat.py`, `conv2d.py`, `split.py`,
`silu.py`, and `upsample.py`; `gpumatrix.py` also retains the low-level
matmul diagnostic. These scripts call `CudaExecutionBackend` and use explicit
Driver API readback. The corresponding `matrixman_*_demo.py` scripts exercise
the frontend where present.

Temporary placement exceptions are intentional: `expand.py`, `transpose.py`,
and `unsqueeze.py` validate metadata-only frontend operations; `stack.py`,
`fill.py`, `arange.py`, and `softmax.py` validate their ATen/MatrixMan paths;
and `split_views.py` is a specialized frontend view-layout regression. They
do not claim to be raw CUDA kernel diagnostics and should not be duplicated
solely to satisfy directory symmetry.

## Compatibility and core checks

```bash
python3 -m drivers.matrixman.compatibility
python3 -m drivers.matrixman --check
python3 -m drivers.matrixman.diagnostics.matrixman_pytorch_demo
python3 -m drivers.matrixman.diagnostics.matrixman_pytorch_demo --trace
python3 -m drivers.matrixman.diagnostics.matrixman_mm_demo
```

The compatibility commands report the active renderer, OpenGL limits,
numerical checks, and convolution tiling behavior. The PyTorch demo verifies
MatrixMan tensor upload, matmul, addition, and explicit readback.

Focused operation checks include:

```bash
python3 -m drivers.matrixman.diagnostics.matrixman_add_demo
python3 -m drivers.matrixman.diagnostics.matrixman_batchnorm_demo
python3 -m drivers.matrixman.diagnostics.matrixman_cat_demo
python3 -m drivers.matrixman.diagnostics.matrixman_grouped_conv_demo
python3 -m drivers.matrixman.diagnostics.matrixman_silu_demo
python3 -m drivers.matrixman.diagnostics.matrixman_softmax_demo
python3 -m drivers.matrixman.diagnostics.matrixman_maxpool_demo
python3 -m drivers.matrixman.diagnostics.matrixman_upsample_demo
python3 -m drivers.matrixman.diagnostics.opengl_storage_demo
```

Additional address, split, transpose, arithmetic, sigmoid, and concatenation
diagnostics are available in the same directory. These scripts compare
read-back GPU results with CPU references where appropriate; unsupported
tensor arithmetic must fail rather than silently execute on CPU.

## Public YOLO demo

```bash
python3 demo/main-tracking.py --imgsz 320 --frames 5 --no-display
```

Useful options are `--model`, `--video`, `--imgsz`, `--conf`, `--iou`,
`--frames`, and `--no-display`. The demo loads the VisDrone model and video,
preprocesses frames, runs inference through MatrixMan, explicitly reads back
the final output, performs detection postprocessing, and optionally draws
results. Its output is intentionally concise and user-oriented.

Do not use the public demo as a benchmark or diagnostic harness. Use the
dedicated runner and scripts below for those purposes.

## YOLO benchmark

The warmed benchmark is documented in [benchmarking.md](benchmarking.md). A
minimal run is:

```bash
python3 -m drivers.matrixman.benchmarks.yolo_benchmark \
  --imgsz 320 --warmup 2 --frames 5 \
  --env MATRIXMAN_TILE_LIMIT=512 \
  --env MATRIXMAN_TILE_SYNC=end \
  --env MATRIXMAN_PROFILE=1 \
  --env MATRIXMAN_GPU_TIMING=1
```

## Convolution and specialized diagnostics

```bash
python3 -m drivers.matrixman.diagnostics.matrixman_conv_demo
python3 -m drivers.matrixman.diagnostics.opengl_conv_target_diagnostic
python3 -m drivers.matrixman.diagnostics.opengl_conv_target_diagnostic --production-tiles
python3 -m drivers.matrixman.diagnostics.matrixman_conv_isolation
python3 -m drivers.matrixman.diagnostics.matrixman_conv_10a_diagnostic
python3 -m drivers.matrixman.diagnostics.opengl_address_diagnostic
```

The Step 10B diagnostic compares baseline GPU Conv with the opt-in spatial
reuse path on the dominant `[1,64,80,80]` 3×3 workload and also runs a tiny
deterministic Conv. The address diagnostic does not perform convolution.

For low-level OpenGL checks:

```bash
python3 -m drivers.matrixman.backends.opengl.gpumatrix
python3 -m drivers.matrixman.diagnostics.cpu_gpu_benchmark --skip-stress
python3 -m drivers.matrixman.backends.opengl.gpu_stress --seconds 10 --size 128
```

The first command is a standalone shader test. The benchmark uses the selected
MatrixMan frontend/backend for its core comparison; its optional stress mode
remains a legacy OpenGL-only stress test and is run only when OpenGL is
selected. Use `MATRIXMAN_PROFILE=1` with the benchmark to include exact CUDA
Driver API allocation/copy timing and transfer bandwidth attribution. These
are not public MatrixMan model demonstrations.

## Hardware notes

Verified OpenGL environments include Intel GM45/GMA 4500MHD, Intel HD
Graphics 4400 with Mesa i915, and NVIDIA GT 720M-class/NVD7 with Mesa
nouveau. The production-safe defaults remain:

```text
MATRIXMAN_TILE_LIMIT=256
MATRIXMAN_TILE_SYNC=per_tile
```

Larger tile limits and alternate synchronization modes are experiments for
specific newer GPUs. They are not GM45-safe defaults.
