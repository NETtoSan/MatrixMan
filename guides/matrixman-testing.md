# MatrixMan testing guide

Run these commands from the repository root. MatrixMan uses its selected
backend during package startup. On a working OpenGL machine, the normal
startup lines are:

```text
MatrixMan probe: OpenCL=not implemented, OpenGL=available
MatrixMan selected: OpenGL
```

This means that OpenGL was probed successfully and selected. The testers use
real OpenGL rendering; they need SDL2, a usable OpenGL 2.1-compatible context,
and a graphical environment. In a headless environment they can fail before
the numerical test with `SDL_Init failed: No available video device`.

## Start here: matrix multiplication

### Standalone OpenGL matrix multiplication

```bash
python3 drivers/matrixman/backends/opengl/gpumatrix.py
```

This is the standalone OpenGL/MatrixMan matrix multiplication diagnostic used
to verify GPU matrix multiplication against a CPU/NumPy reference. It creates
deterministic `4x4` float32 matrices (`1..16` and `16..1`), stores them in
RGBA32F textures, executes GLSL 1.20 fragment-shader addition and
multiplication, reads back the red channel, and reports the maximum absolute
difference and `np.allclose` result against NumPy.

The old root path `drivers/matrixman/gpumatrix.py` is now a compatibility
wrapper and is not the executable tester. Use the OpenGL path above.

### MatrixMan PyTorch matmul demo

```bash
python3 -m drivers.matrixman.diagnostics.gm45_pytorch_demo
python3 -m drivers.matrixman.diagnostics.gm45_pytorch_demo --trace
```

This exercises the MatrixMan-facing PyTorch path: CPU float32 tensors are
uploaded to MatrixMan, `torch.matmul` runs through the selected OpenGL backend,
and the result is read back explicitly. It generates deterministic random
`16x16` tensors with `torch.manual_seed(123)`, checks addition, matrix
multiplication, and a matmul-then-add chain, and compares against CPU PyTorch
using `torch.allclose(rtol=5e-4, atol=5e-4)`. Successful output reports small
maximum errors and `matches=True`.

## OpenGL benchmarks and stress tests

### CPU versus GPU matrix multiplication benchmark

```bash
python3 -m drivers.matrixman.diagnostics.cpu_gpu_benchmark --skip-stress
python3 -m drivers.matrixman.diagnostics.cpu_gpu_benchmark
python3 -m drivers.matrixman.diagnostics.cpu_gpu_benchmark --seconds-per-bench 3 --stress-seconds 60
```

This is a lower-level OpenGL benchmark, not the MatrixMan PyTorch tensor API.
It tests `256x256` and `512x512` matrices, generated from seeded NumPy uniform
data. It compares NumPy `A @ B` with GPU shader-only execution and GPU
execution including uploads/readback. GPU validation uses
`np.allclose(rtol=5e-4, atol=5e-4)`. The default benchmark duration is three
seconds per path; `--skip-stress` omits the additional stress phase.

The actual options are `--seconds-per-bench`, `--skip-stress`, and
`--stress-seconds`.

### Legacy OpenGL matrix stress test

```bash
python3 -m drivers.matrixman.backends.opengl.gpu_stress --seconds 10 --size 128
python3 -m drivers.matrixman.backends.opengl.gpu_stress
```

This directly exercises repeated GLSL add/matmul chains. It prefers a square
matrix size of `256x256` by default, periodically reads back the GPU result,
and compares the expected chain against CPU NumPy calculations. It reports
matmul throughput, estimated GFLOPS, OpenGL errors, and maximum observed
error. Its options are `--seconds`, `--size`, `--validate-interval`, and
`--regen-interval`.

## MatrixMan operation diagnostics

These scripts use `drivers.matrixman` and validate a focused supported
operation by reading the result back and comparing it with a CPU PyTorch
reference. A successful run generally reports a small `max_abs_error` and
`allclose: True`. Options shown below are present in the scripts.

| Command | What it checks |
|---|---|
| `python3 -m drivers.matrixman.diagnostics.gm45_add_demo`<br>`... --trace` | Packed NCHW addition cases. |
| `python3 -m drivers.matrixman.diagnostics.gm45_add_strided_demo` | Stride-aware addition used by YOLO decode. |
| `python3 -m drivers.matrixman.diagnostics.gm45_arange_demo` | Device-side `arange` values and CPU comparison. |
| `python3 -m drivers.matrixman.diagnostics.gm45_batchnorm_demo`<br>`... --trace` | Eval/inference BatchNorm correctness. |
| `python3 -m drivers.matrixman.diagnostics.gm45_cat_demo`<br>`... --trace` | Packed NCHW channel concatenation. |
| `python3 -m drivers.matrixman.diagnostics.gm45_cat_dim1_3d_demo` | Rank-3 channel concatenation for box decode. |
| `python3 -m drivers.matrixman.diagnostics.gm45_cat_lastdim_demo`<br>`... --trace` | Final-dimension concatenation in the detection head. |
| `python3 -m drivers.matrixman.diagnostics.gm45_div_demo` | Scalar division for YOLO decode shapes. |
| `python3 -m drivers.matrixman.diagnostics.gm45_mul_demo` | Broadcast multiplication for YOLO decode; this is elementwise multiplication, not matrix multiplication. |
| `python3 -m drivers.matrixman.diagnostics.gm45_sigmoid_demo` | Sigmoid on YOLO detection-score shapes. |
| `python3 -m drivers.matrixman.diagnostics.gm45_silu_demo`<br>`... --trace` | In-place SiLU correctness. |
| `python3 -m drivers.matrixman.diagnostics.gm45_softmax_demo` | DFL softmax on the traced YOLO shape. |
| `python3 -m drivers.matrixman.diagnostics.gm45_maxpool_demo`<br>`... --trace` | YOLO-subset max pooling. |
| `python3 -m drivers.matrixman.diagnostics.gm45_upsample_demo`<br>`... --trace` | YOLO-subset nearest-neighbor upsampling. |
| `python3 -m drivers.matrixman.diagnostics.gm45_grouped_conv_demo` | Depthwise and general grouped Conv2D regressions. |
| `python3 -m drivers.matrixman.diagnostics.gm45_split_demo`<br>`... --trace` | Metadata-only split behavior. |
| `python3 -m drivers.matrixman.diagnostics.gm45_split_3d_demo` | 3D split/chunk behavior for DFL decoding. |
| `python3 -m drivers.matrixman.diagnostics.gm45_transpose_demo` | Metadata-only transpose and unsqueeze. |
| `python3 -m drivers.matrixman.diagnostics.gm45_storage_demo`<br>`... --trace` | GPU-resident RGBA32F storage and CPU round trips. |

The `... --trace` notation means append `--trace` to the full command; it is
not a separate shell command.

## Convolution and address diagnostics

These are more specialized than the ordinary operation tests.

### Basic Conv2D correctness

```bash
python3 -m drivers.matrixman.diagnostics.gm45_conv_demo
python3 -m drivers.matrixman.diagnostics.gm45_conv_demo --trace
```

Runs several supported 1x1 and 3x3 Conv2D cases, with and without bias, and
compares the read-back result against CPU PyTorch Conv2D.

### Grouped Conv2D

```bash
python3 -m drivers.matrixman.diagnostics.gm45_grouped_conv_demo
```

Runs deterministic depthwise and general grouped Conv2D regressions and
compares them with CPU PyTorch.

### Packed-address diagnostic

```bash
python3 -m drivers.matrixman.diagnostics.gm45_address_diagnostic
```

Checks packed RGBA addressing for input, weight, and output shapes, including
larger YOLO-like tensors. It is an address test and deliberately performs no
convolution.

### Convolution arithmetic probes

```bash
python3 -m drivers.matrixman.diagnostics.gm45_conv_arithmetic_diagnostic
python3 -m drivers.matrixman.diagnostics.gm45_conv_arithmetic_diagnostic --dispatch-only --tiles 1
python3 -m drivers.matrixman.diagnostics.gm45_conv_arithmetic_diagnostic --dispatch-only --tiles 1 --reuse --reset
```

These standalone probes isolate shader arithmetic and dispatch behavior. They
compare selected products/sums or full Conv2D values with CPU PyTorch. The
normal path explicitly says that MatrixMan convolution is not called.
`--dispatch-only` is an argv-driven diagnostic mode; it also accepts
`--tiles N`, `--reset`, and `--reuse`.

### Physical convolution target diagnostic

```bash
python3 -m drivers.matrixman.diagnostics.gm45_conv_target_diagnostic
python3 -m drivers.matrixman.diagnostics.gm45_conv_target_diagnostic --production-tiles
```

Checks framebuffer target sizes and reports completeness, GL errors, and
numerical comparisons. `--production-tiles` runs the existing tiled production
path and prints per-tile comparisons. It is a diagnostic of physical targets,
not a change to production behavior.

### Large Conv2D isolation diagnostic

```bash
python3 -m drivers.matrixman.diagnostics.gm45_conv_isolation
```

Runs the exact large Detect-head Conv2D divergence investigation plus synthetic
regressions. It requires the repository's VisDrone model, Ultralytics, and the
video/model environment used by the diagnostic.

## Compatibility and YOLO operation discovery

### Compatibility checker

```bash
python3 -m drivers.matrixman.compatibility
python3 -m drivers.matrixman --check
```

Reports OpenGL capabilities and runs MatrixMan numerical compatibility checks.
It requires a working SDL/OpenGL context. Successful output includes the
detected renderer/version and passing numerical checks.

### Ultralytics operator discovery

```bash
python3 -m drivers.matrixman.diagnostics.gm45_yolo_test
python3 -m drivers.matrixman.diagnostics.gm45_yolo_test --model yolov8n.yaml --imgsz 64 --limit 20
```

This is primarily an operator-inventory tool. It uses Ultralytics plus
PyTorch FakeTensor/meta execution and a TorchDispatch recorder to identify
operations needed by YOLO; it does not claim to perform ordinary CPU fallback
arithmetic. Its options are `--model`, `--imgsz`, and `--limit`.

## Broader model/inference test

```bash
python3 demo/main-tracking.py --imgsz 320
```

This is substantially broader than matrix multiplication. It loads the
VisDrone model and video, runs model inference through MatrixMan/OpenGL, and
exercises the accumulated tensor, convolution, normalization, activation,
pooling, upsampling, and YOLO decode paths. A successful run reaches model
loading and inference logs while preserving the existing detailed line:

```text
backend: MatrixMan / Intel GM45 / OpenGL 2.1 / GLSL 1.20
```

The demo's actual options are `--model`, `--video`, `--imgsz`, `--conf`,
`--iou`, `--frames`, `--no-display`, `--validate-cpu`,
`--diagnose-divergence`, and `--diagnostic-checkpoints`. `--imgsz` must be a
positive multiple of 32 and no larger than 640. The default model/video paths
are under `demo/`; use `--no-display` for runs where OpenCV windows are not
available.

The smaller examples are also runnable from the repository root:

```bash
python3 demo/example/pytorch_example.py
python3 demo/example/yolo_example.py
```

They demonstrate MatrixMan tensor use and minimal Ultralytics inference,
respectively; they are examples rather than dedicated diagnostics.

## What is duplicated or obsolete?

There are two distinct matrix-multiplication paths:

1. `backends/opengl/gpumatrix.py` is the small deterministic `4x4` low-level
   OpenGL tester.
2. `diagnostics/gm45_pytorch_demo.py` is the MatrixMan-facing PyTorch `16x16`
   matmul test.

They overlap in purpose but exercise different layers. The
`cpu_gpu_benchmark.py` and `backends/opengl/gpu_stress.py` scripts are related
benchmarks/stress tools, not obsolete copies. The many `gm45_*_demo.py` files
are focused operation regressions, not alternate matrix multiplication tests.
