# MatrixMan Windows Environment Variables

This is the complete inventory of `MATRIXMAN_*` variables currently read by
MatrixMan source code. Values are process environment variables; they do not
modify the source tree.

## PowerShell vs cmd.exe

Identify the shell before copying a command:

```text
PS C:\Users\admin\Documents\GitHub\MatrixMan>   # PowerShell
C:\Users\admin\Documents\GitHub\MatrixMan>      # cmd.exe
```

PowerShell:

```powershell
$env:MATRIXMAN_BACKEND="opengl"
$env:MATRIXMAN_BACKEND="opengl"; python demo/main-tracking.py
Remove-Item Env:MATRIXMAN_BACKEND
```

cmd.exe:

```cmd
set MATRIXMAN_BACKEND=opengl
set "MATRIXMAN_BACKEND=opengl" && python demo/main-tracking.py
set MATRIXMAN_BACKEND=
```

PowerShell and cmd.exe use different syntax. `set "VAR=value" && command` is
cmd.exe syntax and is not valid in older PowerShell versions. Unix syntax such
as `MATRIXMAN_BACKEND=opengl python ...` is not Windows syntax. These commands
affect only the current shell/session. `setx` changes future shells, not the
current one, and persistent development settings can be confusing; temporary
per-terminal settings are recommended.

## Quick Reference

| Variable | Backend/Subsystem | Purpose | Default | Example value |
|---|---|---|---|---|
| `MATRIXMAN_BACKEND` | shared selector | Select `auto`, `cuda`, or `opengl` | automatic capability order | `opengl` |
| `MATRIXMAN_PROFILE` | shared profiling | Enable selected-backend profiling | off | `1` |
| `MATRIXMAN_CUDA_PROFILE` | CUDA legacy profiling | CUDA-only fallback profiling switch | off | `1` |
| `MATRIXMAN_TRACE` | shared tracing | High-level operation trace | off | `1` |
| `MATRIXMAN_DEBUG` | OpenGL diagnostics | OpenGL low-level debug output | off | `1` |
| `MATRIXMAN_PROFILE_DETAIL` | OpenGL profiling | Detailed profiler report | off | `1` |
| `MATRIXMAN_GPU_TIMING` | OpenGL profiling | Deferred GPU timer queries | off | `1` |
| `MATRIXMAN_TILE_LIMIT` | OpenGL convolution | Physical tile limit or validated `auto` selection | `256` | `512` or `auto` |
| `MATRIXMAN_TILE_SYNC` | OpenGL convolution | Inter-tile synchronization mode | `per_tile` | `end` |
| `MATRIXMAN_CONV_SPATIAL_REUSE` | OpenGL convolution | Experimental spatial reuse | off | `1` |
| `MATRIXMAN_SKIP_PRE_CONSOLIDATION_SYNC` | OpenGL convolution | Skip pre-consolidation sync experiment | off | `1` |
| `MATRIXMAN_DIAGNOSTIC_TILES` | OpenGL compatibility | Capture tile diagnostics | off | `1` |
| `MATRIXMAN_DIAGNOSTIC_RECT_TILES` | OpenGL diagnostics | Enable rectangular/order tile experiments | off | `1` |
| `MATRIXMAN_DIAG_TILE_WIDTH` | OpenGL diagnostics | Diagnostic tile width | tile limit | `440` |
| `MATRIXMAN_DIAG_TILE_HEIGHT` | OpenGL diagnostics | Diagnostic tile height | tile limit | `400` |
| `MATRIXMAN_DIAG_TILE_ORDER` | OpenGL diagnostics | Tile traversal order | `normal` | `reverse` |
| `MATRIXMAN_DIAG_CONV_WORKLOAD` | OpenGL compatibility | Large-convolution workload | `heavy` | `light` |
| `MATRIXMAN_GPU_POSTPROCESS` | YOLO benchmark | GPU detection postprocessing | off | `1` |
| `MATRIXMAN_CUDA_DEBUG` | CUDA | Print CUDA/PTX debug information | off | `1` |
| `MATRIXMAN_CUDA_DISABLE_ASYNC_QUEUE` | CUDA | Disable asynchronous queueing | off | `1` |
| `MATRIXMAN_CUDA_DISABLE_ALLOC_POOL` | CUDA allocator | Disable CUDA allocation pool | off | `1` |
| `MATRIXMAN_CUDA_DISABLE_SPECIALIZED_CONV` | CUDA convolution | Disable specialized kernels | off | `1` |
| `MATRIXMAN_CUDA_CONV3X3_VARIANT` | CUDA convolution | Select 3x3 variant | `plane` | `plane_legacy` |
| `MATRIXMAN_CUDA_LEGACY_MODULE_LOAD` | CUDA loader | Use `cuModuleLoadData` instead of `cuModuleLoadDataEx` | off | `1` |
| `MATRIXMAN_AUDIT_CPU_LEAKS` | shared audit | Enable CPU materialization audit | off | `1` |
| `MATRIXMAN_TILE_AUTOTUNE_REFRESH` | OpenGL autotuning | Force a fresh tile autotune | off | `1` |

Boolean variables use the common MatrixMan truth rule: unset, empty, `0`,
`false`, `no`, and `off` are false; `1`, `true`, `yes`, and `on` are true.
Other values are rejected by the centralized configuration parser.

## Shared / Backend Selection Variables

`MATRIXMAN_BACKEND` is an enum (`auto`, `cuda`, `opengl`). An explicit Python
preference takes precedence over the environment variable. Invalid values are
rejected by the selector. `MATRIXMAN_PROFILE` controls selected-backend
profiling; `MATRIXMAN_CUDA_PROFILE` is the legacy CUDA fallback when the
canonical profile setting is absent. `MATRIXMAN_TRACE` enables high-level trace
messages. These are implemented in `drivers/matrixman/config.py` and
`drivers/matrixman/selector.py`.

## OpenGL Variables

Sources: `backends/opengl/diagnostics.py`, `profiling.py`, `convolution.py`,
`compatibility.py`, and `benchmarks/yolo_benchmark.py`.

- `MATRIXMAN_DEBUG`, `MATRIXMAN_PROFILE_DETAIL`, `MATRIXMAN_GPU_TIMING`, and
  `MATRIXMAN_CONV_SPATIAL_REUSE` are booleans. They affect diagnostics,
  profiler detail/timer queries, or an experimental convolution path; keep them
  off for ordinary runs.
- `MATRIXMAN_TILE_LIMIT` is a positive integer or `auto`. Invalid or
  non-positive values raise an error. `auto` validates and caches a
  device/driver-specific OpenGL limit; unset uses the 256 default.
- `MATRIXMAN_TILE_SYNC` accepts `per_tile`, `end`, `flush`, or `none`; invalid
  values raise an error. It changes synchronization/performance, not intended
  numerical semantics.
- `MATRIXMAN_SKIP_PRE_CONSOLIDATION_SYNC` is a boolean experiment.
- `MATRIXMAN_DIAGNOSTIC_TILES` and `MATRIXMAN_DIAGNOSTIC_RECT_TILES` are
  boolean diagnostic switches.
- `MATRIXMAN_DIAG_TILE_WIDTH` and `MATRIXMAN_DIAG_TILE_HEIGHT` are positive
  integers used only when rectangular diagnostics are enabled; invalid values
  raise an error.
- `MATRIXMAN_DIAG_TILE_ORDER` accepts `normal`, `reverse`, `column`, or
  `reverse_column` when rectangular diagnostics are enabled; invalid values
  raise an error.
- `MATRIXMAN_DIAG_CONV_WORKLOAD` accepts `heavy`, `medium`, `light`, or
  `one_by_one`; invalid values raise an error in the compatibility report.
- `MATRIXMAN_GPU_POSTPROCESS` is read by the YOLO benchmark only. It is a
  boolean and selects experimental GPU detection reduction; it is not a core
  OpenGL operator switch.

Most OpenGL tuning and diagnostic variables are experimental. They are safe to
leave unset; use them for controlled diagnostics or performance experiments.

Exact source map:

| Variable | Source file(s) |
|---|---|
| `MATRIXMAN_BACKEND` | `drivers/matrixman/selector.py` |
| `MATRIXMAN_PROFILE` | `drivers/matrixman/config.py` |
| `MATRIXMAN_CUDA_PROFILE` | `drivers/matrixman/config.py` |
| `MATRIXMAN_TRACE` | `drivers/matrixman/config.py` |
| `MATRIXMAN_DEBUG` | `drivers/matrixman/backends/opengl/diagnostics.py` |
| `MATRIXMAN_PROFILE_DETAIL` | `drivers/matrixman/backends/opengl/profiling.py` |
| `MATRIXMAN_GPU_TIMING` | `drivers/matrixman/backends/opengl/profiling.py` |
| `MATRIXMAN_TILE_LIMIT` | `drivers/matrixman/backends/opengl/convolution.py`, `profiling.py`, `compatibility.py` |
| `MATRIXMAN_TILE_SYNC` | `drivers/matrixman/backends/opengl/convolution.py`, `profiling.py`, `compatibility.py` |
| `MATRIXMAN_CONV_SPATIAL_REUSE` | `drivers/matrixman/backends/opengl/convolution.py`, `diagnostics/matrixman_conv_10a_diagnostic.py` |
| `MATRIXMAN_SKIP_PRE_CONSOLIDATION_SYNC` | `drivers/matrixman/backends/opengl/convolution.py` |
| `MATRIXMAN_DIAGNOSTIC_TILES` | `drivers/matrixman/backends/opengl/convolution.py`, `compatibility.py`, `diagnostics/opengl_conv_target_diagnostic.py` |
| `MATRIXMAN_DIAGNOSTIC_RECT_TILES` | `drivers/matrixman/backends/opengl/convolution.py`, `compatibility.py` |
| `MATRIXMAN_DIAG_TILE_WIDTH` | `drivers/matrixman/backends/opengl/convolution.py`, `compatibility.py` |
| `MATRIXMAN_DIAG_TILE_HEIGHT` | `drivers/matrixman/backends/opengl/convolution.py`, `compatibility.py` |
| `MATRIXMAN_DIAG_TILE_ORDER` | `drivers/matrixman/backends/opengl/convolution.py`, `compatibility.py` |
| `MATRIXMAN_DIAG_CONV_WORKLOAD` | `drivers/matrixman/compatibility.py` |
| `MATRIXMAN_GPU_POSTPROCESS` | `drivers/matrixman/benchmarks/yolo_benchmark.py` |
| `MATRIXMAN_CUDA_DEBUG` | `drivers/matrixman/backends/cuda/gpumatrix.py` |
| `MATRIXMAN_CUDA_DISABLE_ASYNC_QUEUE` | `drivers/matrixman/backends/cuda/gpumatrix.py` |
| `MATRIXMAN_CUDA_DISABLE_ALLOC_POOL` | `drivers/matrixman/backends/cuda/gpumatrix.py` |
| `MATRIXMAN_CUDA_DISABLE_SPECIALIZED_CONV` | `drivers/matrixman/backends/cuda/gpumatrix.py`, `backend.py` |
| `MATRIXMAN_CUDA_CONV3X3_VARIANT` | `drivers/matrixman/backends/cuda/backend.py` |
| `MATRIXMAN_CUDA_LEGACY_MODULE_LOAD` | `drivers/matrixman/backends/cuda/gpumatrix.py` |
| `MATRIXMAN_AUDIT_CPU_LEAKS` | `drivers/matrixman/audit.py` |
| `MATRIXMAN_TILE_AUTOTUNE_REFRESH` | `drivers/matrixman/config.py`, `drivers/matrixman/backends/opengl/runtime.py`, `drivers/matrixman/diagnostics/opengl_tile_limit.py` |

## CUDA Variables

Sources: `backends/cuda/gpumatrix.py`, `backends/cuda/backend.py`, and
`config.py`.

- `MATRIXMAN_CUDA_DEBUG`: boolean; prints CUDA/PTX debug information.
- `MATRIXMAN_CUDA_DISABLE_ASYNC_QUEUE`: boolean; disables asynchronous queue
  behavior for synchronized experiments and may reduce performance.
- `MATRIXMAN_CUDA_DISABLE_ALLOC_POOL`: boolean; disables the CUDA allocation
  pool and changes allocator performance behavior.
- `MATRIXMAN_CUDA_DISABLE_SPECIALIZED_CONV`: boolean; forces the general CUDA
  convolution path instead of specialized variants.
- `MATRIXMAN_CUDA_CONV3X3_VARIANT`: string. The recognized specialized value is
  `plane_legacy`; the normal default is `plane`. Other values simply do not
  select the legacy specialized variant and fall back to normal selection.
- `MATRIXMAN_CUDA_LEGACY_MODULE_LOAD`: boolean; selects legacy
  `cuModuleLoadData` rather than `cuModuleLoadDataEx`. It is a legacy/diagnostic
  loader experiment.
- `MATRIXMAN_CUDA_PROFILE`: legacy boolean profiling setting, used only when
  `MATRIXMAN_PROFILE` is not set.

These variables are CUDA-only and do not affect OpenGL or OpenCL. They are
mostly diagnostic, allocator, synchronization, or kernel-selection controls.

## OpenCL Variables

No OpenCL-specific environment variables were found in the current source tree.

## CPU Leakage Audit

`MATRIXMAN_AUDIT_CPU_LEAKS` is a shared boolean opt-in. When enabled, the audit
prints compact category lines such as:

```text
[MatrixMan/Audit] allowed_bookkeeping
[MatrixMan/Audit] explicit_readback
[MatrixMan/Audit] explicit_cpu_transfer
[MatrixMan/Audit] placeholder_metadata
[MatrixMan/Audit][WARNING] unexpected_cpu_materialization
```

Example PowerShell usage:

```powershell
$env:MATRIXMAN_AUDIT_CPU_LEAKS="1"; $env:MATRIXMAN_BACKEND="opengl"; python demo/main-tracking.py --imgsz 320
```

The audit tracks tensor/data paths, not total process CPU utilization. Python,
OpenCV, and OpenGL driver submission can use CPU without indicating that
MatrixMan tensor computation fell back to CPU.

## Useful Windows Recipes

PowerShell examples use semicolons; cmd.exe examples use `&&`:

```powershell
$env:MATRIXMAN_BACKEND="opengl"; python -m drivers.matrixman --check
$env:MATRIXMAN_BACKEND="cuda"; python -m drivers.matrixman --check
$env:MATRIXMAN_AUDIT_CPU_LEAKS="1"; $env:MATRIXMAN_BACKEND="opengl"; python demo/main-tracking.py --imgsz 320
$env:MATRIXMAN_CUDA_DISABLE_ASYNC_QUEUE="1"; $env:MATRIXMAN_BACKEND="cuda"; python -m drivers.matrixman.diagnostics.matrixman_conv_demo
$env:MATRIXMAN_CUDA_CONV3X3_VARIANT="plane_legacy"; $env:MATRIXMAN_BACKEND="cuda"; python -m drivers.matrixman.diagnostics.matrixman_conv_demo
$env:MATRIXMAN_PROFILE="1"; $env:MATRIXMAN_PROFILE_DETAIL="1"; $env:MATRIXMAN_BACKEND="opengl"; python -m drivers.matrixman --check
$env:MATRIXMAN_DEBUG="1"; $env:MATRIXMAN_TRACE="1"; $env:MATRIXMAN_BACKEND="opengl"; python -m drivers.matrixman --check
$env:MATRIXMAN_DIAGNOSTIC_RECT_TILES="1"; $env:MATRIXMAN_DIAG_TILE_WIDTH="440"; $env:MATRIXMAN_DIAG_TILE_HEIGHT="400"; python -m drivers.matrixman --check
```

```cmd
set "MATRIXMAN_BACKEND=opengl" && python -m drivers.matrixman --check
set "MATRIXMAN_BACKEND=cuda" && python -m drivers.matrixman --check
set "MATRIXMAN_AUDIT_CPU_LEAKS=1" && set "MATRIXMAN_BACKEND=opengl" && python demo/main-tracking.py --imgsz 320
set "MATRIXMAN_CUDA_DISABLE_ASYNC_QUEUE=1" && set "MATRIXMAN_BACKEND=cuda" && python -m drivers.matrixman.diagnostics.matrixman_conv_demo
set "MATRIXMAN_CUDA_CONV3X3_VARIANT=plane_legacy" && set "MATRIXMAN_BACKEND=cuda" && python -m drivers.matrixman.diagnostics.matrixman_conv_demo
set "MATRIXMAN_PROFILE=1" && set "MATRIXMAN_PROFILE_DETAIL=1" && set "MATRIXMAN_BACKEND=opengl" && python -m drivers.matrixman --check
set "MATRIXMAN_DEBUG=1" && set "MATRIXMAN_TRACE=1" && set "MATRIXMAN_BACKEND=opengl" && python -m drivers.matrixman --check
```

Inspect current values with `$env:MATRIXMAN_BACKEND` in PowerShell or
`echo %MATRIXMAN_BACKEND%` in cmd.exe. List all MatrixMan variables with:

```powershell
Get-ChildItem Env: | Where-Object Name -Like "MATRIXMAN*"
```

```cmd
set MATRIXMAN
```

## Persistent Windows Environment Variables

Session-local assignments are recommended for development. `setx VAR value`
and Windows environment settings affect future processes/shells, not the
current shell. Persistent settings can unintentionally force a backend or
diagnostic mode in later work.

## Troubleshooting

- Do not paste cmd.exe `&&` syntax into older PowerShell versions.
- Do not use Unix `NAME=value command` syntax in PowerShell or cmd.exe.
- Unset variables remain active until removed or the shell is closed.
- Check for an old forced backend with `$env:MATRIXMAN_BACKEND` or
  `echo %MATRIXMAN_BACKEND%`.
- Variable names are case-insensitive on Windows, but spelling and underscores
  must match the MatrixMan names documented here.
- A typo is ignored because the program never reads that name.

## Source Audit

- **Unique active `MATRIXMAN_*` names found:** 26.
- **Source files containing environment reads:** `drivers/matrixman/audit.py`,
  `selector.py`, `config.py`, `compatibility.py`,
  `config.py`, `backends/opengl/diagnostics.py`,
  `backends/opengl/profiling.py`, `backends/opengl/convolution.py`,
  `backends/cuda/gpumatrix.py`, `backends/cuda/backend.py`, and
  `benchmarks/yolo_benchmark.py`.
- **Documentation/scripts:** documentation repeats current names for examples;
  no additional documentation-only MatrixMan variable was found. Diagnostic
  modules mutate the same current variables but do not introduce new names.
- **OpenCL:** no OpenCL-specific variables found.
- **Ambiguous cases manually reviewed:** `MATRIXMAN_GPU_POSTPROCESS` is
  benchmark-only; `MATRIXMAN_DIAGNOSTIC_*` variables are compatibility/
  diagnostic-only; CUDA variant values other than `plane_legacy` fall through
  to normal kernel selection rather than being rejected.

All 26 currently consumed `MATRIXMAN_*` variables are represented in the Quick
Reference table and described above. The canonical mapping and lifecycle
reference is [`matrixman-configuration.md`](matrixman-configuration.md).
