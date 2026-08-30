# MatrixMan testing

Run commands from the repository root. MatrixMan currently selects OpenGL when
the SDL/OpenGL probe succeeds:

```text
MatrixMan probe: OpenCL=not implemented, OpenGL=available
MatrixMan selected: OpenGL
```

The tests require SDL2, a usable OpenGL context, floating-point texture and
framebuffer support, and a graphical environment. A headless failure such as
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

## Compatibility and core checks

```bash
python3 -m drivers.matrixman.compatibility
python3 -m drivers.matrixman --check
python3 -m drivers.matrixman.diagnostics.gm45_pytorch_demo
python3 -m drivers.matrixman.diagnostics.gm45_pytorch_demo --trace
```

The compatibility commands report the active renderer, OpenGL limits,
numerical checks, and convolution tiling behavior. The PyTorch demo verifies
MatrixMan tensor upload, matmul, addition, and explicit readback.

Focused operation checks include:

```bash
python3 -m drivers.matrixman.diagnostics.gm45_add_demo
python3 -m drivers.matrixman.diagnostics.gm45_batchnorm_demo
python3 -m drivers.matrixman.diagnostics.gm45_cat_demo
python3 -m drivers.matrixman.diagnostics.gm45_grouped_conv_demo
python3 -m drivers.matrixman.diagnostics.gm45_silu_demo
python3 -m drivers.matrixman.diagnostics.gm45_softmax_demo
python3 -m drivers.matrixman.diagnostics.gm45_maxpool_demo
python3 -m drivers.matrixman.diagnostics.gm45_upsample_demo
python3 -m drivers.matrixman.diagnostics.gm45_storage_demo
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
python3 -m drivers.matrixman.diagnostics.gm45_conv_demo
python3 -m drivers.matrixman.diagnostics.gm45_conv_target_diagnostic
python3 -m drivers.matrixman.diagnostics.gm45_conv_target_diagnostic --production-tiles
python3 -m drivers.matrixman.diagnostics.gm45_conv_isolation
python3 -m drivers.matrixman.diagnostics.gm45_conv_10a_diagnostic
python3 -m drivers.matrixman.diagnostics.gm45_address_diagnostic
```

The Step 10B diagnostic compares baseline GPU Conv with the opt-in spatial
reuse path on the dominant `[1,64,80,80]` 3×3 workload and also runs a tiny
deterministic Conv. The address diagnostic does not perform convolution.

For low-level OpenGL checks:

```bash
python3 drivers/matrixman/backends/opengl/gpumatrix.py
python3 -m drivers.matrixman.diagnostics.cpu_gpu_benchmark --skip-stress
python3 -m drivers.matrixman.backends.opengl.gpu_stress --seconds 10 --size 128
```

The first command is a standalone shader test; the latter two are lower-level
OpenGL benchmark/stress tools, not public MatrixMan model demonstrations.

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
