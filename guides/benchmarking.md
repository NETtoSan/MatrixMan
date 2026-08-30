# MatrixMan benchmarking

## Separation from the public demo

`demo/main-tracking.py` is the public-facing YOLO showcase. It intentionally
contains only model/video setup, backend identification, preprocessing,
MatrixMan inference, explicit final output readback, detection postprocessing,
drawing, and simple user-level timing/FPS output.

It is not a diagnostic tool and does not own warm-up epochs, profiling
internals, GPU tuning controls, JSON logging, timer-query reporting, or
backend-specific performance analysis.

## Dedicated runner

Use `drivers/matrixman/benchmarks/yolo_benchmark.py`. The runner owns:

- warm-up frames that complete readback and postprocessing;
- the measured-frame epoch and `profile_reset()` after warm-up;
- upload, inference, readback, postprocessing, and total timings;
- MatrixMan CPU profiling and OpenGL GPU timer-query collection;
- the `--variant` label for baseline/optimization comparisons;
- repeated `--env NAME=VALUE` backend settings;
- JSON output using `matrixman.yolo-benchmark.v1`;
- a backend-neutral result shape reserved for future OpenCL runs.

Runtime/program/texture and parameter caches remain alive across the profiling
reset. The reset clears statistics so the JSON represents warmed steady-state
frames.

## Command shape

```bash
python3 -m drivers.matrixman.benchmarks.yolo_benchmark \
  --imgsz 320 --warmup 2 --frames 5 \
  --variant NAME \
  --env MATRIXMAN_TILE_LIMIT=512 \
  --env MATRIXMAN_TILE_SYNC=end \
  --env MATRIXMAN_PROFILE=1 \
  --env MATRIXMAN_GPU_TIMING=1 \
  --json textlogs/NAME.json
```

`--model`, `--video`, `--conf`, and `--iou` are also available. OpenCL is
reserved in the result schema but is not implemented; the current runner
accepts the implemented OpenGL backend only.

## Reproducible OpenGL comparisons

These commands use two warm-up frames followed by five measured frames.

### GT 720M-class / nouveau

Baseline:

```bash
DRI_PRIME=1 python3 -m drivers.matrixman.benchmarks.yolo_benchmark \
  --imgsz 320 --warmup 2 --frames 5 --variant gt720m-baseline \
  --env MATRIXMAN_TILE_LIMIT=512 --env MATRIXMAN_TILE_SYNC=end \
  --env MATRIXMAN_PROFILE=1 --env MATRIXMAN_GPU_TIMING=1 \
  --json textlogs/gt720m-baseline.json
```

Spatial reuse:

```bash
DRI_PRIME=1 python3 -m drivers.matrixman.benchmarks.yolo_benchmark \
  --imgsz 320 --warmup 2 --frames 5 --variant gt720m-spatial-reuse \
  --env MATRIXMAN_TILE_LIMIT=512 --env MATRIXMAN_TILE_SYNC=end \
  --env MATRIXMAN_PROFILE=1 --env MATRIXMAN_GPU_TIMING=1 \
  --env MATRIXMAN_CONV_SPATIAL_REUSE=1 \
  --json textlogs/gt720m-spatial-reuse.json
```

### Intel HD Graphics 4400 / i915

Baseline:

```bash
env -u DRI_PRIME python3 -m drivers.matrixman.benchmarks.yolo_benchmark \
  --imgsz 320 --warmup 2 --frames 5 --variant hd4400-baseline \
  --env MATRIXMAN_TILE_LIMIT=512 --env MATRIXMAN_TILE_SYNC=end \
  --env MATRIXMAN_PROFILE=1 --env MATRIXMAN_GPU_TIMING=1 \
  --json textlogs/hd4400-baseline.json
```

Spatial reuse:

```bash
env -u DRI_PRIME python3 -m drivers.matrixman.benchmarks.yolo_benchmark \
  --imgsz 320 --warmup 2 --frames 5 --variant hd4400-spatial-reuse \
  --env MATRIXMAN_TILE_LIMIT=512 --env MATRIXMAN_TILE_SYNC=end \
  --env MATRIXMAN_PROFILE=1 --env MATRIXMAN_GPU_TIMING=1 \
  --env MATRIXMAN_CONV_SPATIAL_REUSE=1 \
  --json textlogs/hd4400-spatial-reuse.json
```

## Recorded results

These are warmed steady-state results for this exact MatrixMan/OpenGL YOLO
320×320 workload, not general GPU, gaming, or compute rankings:

| Device/profile | Time per frame | FPS | Comparison |
| --- | ---: | ---: | --- |
| GT 720M baseline | ~1.654 s | ~0.60 | — |
| GT 720M spatial reuse | ~1.612 s | ~0.62 | ~2.5% lower frame time |
| HD 4400 baseline | ~0.723 s | ~1.38 | — |
| HD 4400 spatial reuse | ~0.448 s | ~2.23 | ~1.61× overall speedup; ~38% lower frame time |

Spatial reuse cuts Conv GPU time substantially on HD 4400. The gain is small
on GT 720M/nouveau despite the same algorithm. These measurements used Intel
Haswell HD Graphics 4400 with Mesa i915 and NVIDIA GF117/GT720M-class hardware
with Mesa nouveau. They did not use CUDA or proprietary NVIDIA drivers.

The benchmark JSON also records per-frame timing, readback bytes, candidate
counts, CPU profiling, GPU timer-query capability/results, unresolved/dropped
queries, synchronization counters, and Conv signature samples when enabled.
