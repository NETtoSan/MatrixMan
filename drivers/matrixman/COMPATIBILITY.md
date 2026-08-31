# MatrixMan Compatibility and Diagnostics

MatrixMan's compatibility checker answers two questions: can the current
OpenGL implementation provide the capabilities MatrixMan needs, and do small
numerical backend tests agree with CPU PyTorch? It is a practical smoke test,
not a guarantee that every model or operator will work. The deeper diagnostic
modes are development tools for investigating driver and workload behavior.

## Normal Checks

Run either public entrypoint:

```bash
python3 -m drivers.matrixman.compatibility
python3 -m drivers.matrixman --check
```

The report includes the renderer, OpenGL/GLSL versions, relevant limits and
extensions, numerical tests for the supported paths, and the large-convolution
quirk check. Its status labels are `COMPATIBLE`, `PARTIALLY COMPATIBLE`, and
`UNSUPPORTED`. A large one-shot convolution failure does not by itself mean
that MatrixMan is unsupported when the normal tiled production path passes.

## Verified Hardware

The primary confirmed platform is:

```text
ThinkPad X200
Intel GMA 4500MHD / GM45
Linux Mesa
OpenGL 2.1
GLSL 1.20
```

On this machine, MatrixMan tensor operations, convolution, the conservative
tiled large-convolution path, and real Ultralytics/YOLO inference have been
executed successfully. The model must still use operations implemented by the
current MatrixMan subset; this does not imply that every Ultralytics model is
supported.

## GM45 Baseline

The validated production defaults are:

```text
MATRIXMAN_TILE_LIMIT=256
MATRIXMAN_TILE_SYNC=per_tile
```

`per_tile` renders one physical convolution tile and calls `glFinish()` before
rendering the next. On the GM45 this is currently required for correctness.
The 256x256 limit is a conservative GM45-validated default, not a universal
safe maximum and not a claim about other GPUs.

Large physical convolution draws can become nondeterministically unstable on
GM45/Mesa. The same heavier diagnostic configuration may pass in one fresh
process and fail in another. The corruption can occur in a physical
convolution tile before consolidation. By contrast, consolidation-only tests
fed with known-good uploaded tiles were stable with exact zero error across the
tested geometries.

The investigation did not establish a simple width, height, area, alignment,
or render-order threshold. Heavier fragment-shader workloads appear more
likely to expose the problem: a 64-channel 3x3 workload (576 MACs per output)
was intermittently unstable at large physical tiles, while a 16-channel 3x3
workload (144 MACs per output) passed repeated tests. This is an observed
engineering hypothesis, not a formal hardware limit or Intel erratum.

## Diagnostic Controls

The standard compatibility command is the normal entrypoint. The following
development flags exist in `compatibility.py`:

```bash
python3 -m drivers.matrixman.compatibility --test-large-tiled
python3 -m drivers.matrixman.compatibility --test-consolidation
```

`--test-large-tiled` runs one fresh-context tiled convolution diagnostic and
reports physical tile geometry, per-tile validation when enabled by the
diagnostic path, and consolidated validation. `--test-consolidation` tests the
GPU consolidation path using deterministic uploaded tile data without running
convolution.

`--internal-large-one-shot` is an internal child-process mode used by the
normal checker to probe the known-unsafe 512x512 one-shot path. It is not a
recommended compatibility test. A 512-sized physical convolution render has
previously hung the GM45 graphics stack; do not use it as a routine sweep.

### Production Settings

These settings affect normal convolution execution as well as diagnostics:

* `MATRIXMAN_TILE_LIMIT`: positive maximum physical tile dimension; default
  `256`.
* `MATRIXMAN_TILE_SYNC`: `per_tile`, `end`, `flush`, or `none`; default
  `per_tile`. The alternatives are experimental. On the verified GM45,
  `end`, `flush`, and `none` produced corrupted results and are not promoted.

### Diagnostic-Only Settings

The following rectangular/order/workload controls are used by
`--test-large-tiled` and do not affect ordinary MatrixMan or YOLO execution:

* `MATRIXMAN_DIAG_TILE_WIDTH` and `MATRIXMAN_DIAG_TILE_HEIGHT`: diagnostic
  physical dimensions, enabled by that diagnostic's rectangular-tile mode.
* `MATRIXMAN_DIAG_TILE_ORDER`: `normal`, `reverse`, `column`, or
  `reverse_column`.
* `MATRIXMAN_DIAG_CONV_WORKLOAD`: `heavy`, `medium`, `light`, or
  `one_by_one`.

For example, a rectangular diagnostic can be run in a fresh process with:

```bash
MATRIXMAN_DIAG_TILE_WIDTH=440 MATRIXMAN_DIAG_TILE_HEIGHT=400 \
python3 -m drivers.matrixman.compatibility --test-large-tiled
```

Diagnostic readbacks and timings are for localization only and are not
representative of production performance. Run risky configurations one at a
time in separate processes because a bad Mesa workload may poison or hang the
graphics context.

`MATRIXMAN_PROFILE=1` enables the selected backend's profiler (CUDA and
OpenGL retain independent implementations);
`MATRIXMAN_PROFILE_DETAIL=1` adds detailed profiling output. These are
profiling controls, not capability requirements.

Python configuration is also available:

```python
import matrixman

matrixman.prefer("cuda")       # "cuda", "opengl", or "auto"
matrixman.profiling = True     # explicit profiling override
matrixman.trace = True         # high-level operation tracing
```

An explicit Python preference wins over `MATRIXMAN_BACKEND`; `"auto"` clears
the Python override and restores normal environment-then-automatic selection.
For profiling, an explicit Python setting wins over `MATRIXMAN_PROFILE`, which
wins over the legacy CUDA-only `MATRIXMAN_CUDA_PROFILE` setting. The legacy
variable remains accepted by the CUDA profiler for compatibility.

Tracing is independent of both selection and profiling. It is disabled by
default; use `MATRIXMAN_TRACE=1` (or `matrixman.trace = True`) to enable
high-level operation messages, and `MATRIXMAN_TRACE=0` (or `False`) to disable
them.

## What Has Been Established

On the verified GM45 platform:

* Float texture upload/readback, floating-point framebuffer rendering, the
  supported elementwise path, Conv2D, BatchNorm, and in-place SiLU pass the
  corresponding practical checks.
* The normal production tiled 512x512 logical convolution path passes with
  physical tiles no larger than 256x256.
* Per-tile `glFinish()` is required by the tested GM45 workload.
* The intentionally unsafe large one-shot path can fail numerically or become
  unstable, while the validated tiled fallback remains the supported path.
* Standalone consolidation is not independently implicated by the tested
  deterministic consolidation diagnostics.

These findings describe the tested GM45/Mesa path. They do not define a
universal OpenGL rule.

## Unverified Hardware

Other Intel GPUs, AMD/Radeon GPUs, NVIDIA GPUs, Apple Silicon, ROCm-related
platforms, other drivers, and other OpenGL implementations are currently
unverified. They are portability experiments and project goals, not supported
platforms. OpenGL support alone is insufficient: driver behavior, GLSL
features, floating-point textures and render targets, texture limits, shader
behavior, and the required ATen operator subset all matter.

Run the normal compatibility probe before assuming a machine works. MatrixMan
does not claim ROCm, CUDA, Apple, Radeon, or universal OpenGL compatibility.
