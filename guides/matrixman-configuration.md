# MatrixMan runtime configuration

This is the canonical reference for the mapping between `MATRIXMAN_*`
environment variables and the typed attributes on `matrixman.config`.

```python
from drivers import matrixman

print(matrixman.config)
print(matrixman.config.asDict())
```

The configuration object is created when `drivers.matrixman` is imported. Its
precedence is:

```text
built-in default -> environment-derived value -> explicit Python assignment
```

Environment values are strings. The configuration layer parses them into
normalized Python values. For example, `"512"` becomes `512`, `"1"` becomes
`True`, `"false"` becomes `False`, and `"auto"` remains the string `"auto"`
where that value is supported.

Boolean environment values are case-insensitive after trimming. The accepted
true spellings are `1`, `true`, `yes`, and `on`; the accepted false spellings
are the empty string, `0`, `false`, `no`, and `off`. Other boolean values raise
`ValueError` while configuration is loaded or assigned.

## Python-native configuration

Assignments use camelCase attributes and are type-checked immediately:

```python
from drivers import matrixman

matrixman.config.backend = "opengl"
matrixman.config.tileLimit = "auto"
matrixman.config.tileSync = "end"
matrixman.config.convSpatialReuse = True
matrixman.config.profile = True
matrixman.config.gpuTiming = True
```

An explicit Python assignment overrides the corresponding environment-derived
value for the life of the current configuration state. Assign settings before
the first MatrixMan backend use when possible. Configuration changes do not
create or destroy GPU contexts. Changing `backend` after a backend is
initialized cannot switch that active backend; the selector requires the
preference to be set before first MatrixMan device use. Settings captured while
an operation or backend is initialized may likewise not recreate already-created
contexts, programs, allocators, or other resources.

The public lifecycle helpers are:

```python
matrixman.config.reloadFromEnvironment()  # discard Python overrides; reload all env values
matrixman.config.reset()                  # discard overrides; restore built-in defaults
values = matrixman.config.asDict()        # shallow copy of current values
```

`reset()` does not modify `os.environ`. The object has a compact readable
`repr`, for example:
`MatrixManConfig(backend='auto', tileLimit=256, resolvedTileLimit=256, ...)`.

## Configuration reference

The following 26 variables are the active `MATRIXMAN_*` mappings in the
current source. Diagnostic and benchmark-only controls are included because
they are intentionally supported by the centralized configuration object.

### Backend selection

| Environment variable | Python attribute | Type | Default | Accepted values | Description | Lifecycle notes |
| --- | --- | --- | --- | --- | --- | --- |
| `MATRIXMAN_BACKEND` | `config.backend` | `str` | `"auto"` | `auto`, `cuda`, `opengl` | Selects automatic capability-based selection, CUDA, or OpenGL. | `auto` means automatic selection; it is not a backend implementation name. Set before first backend use. |

`auto` probes usable backends and selects the first available implementation in
the selector's capability order. An explicit `cuda` or `opengl` request fails
if that backend is unavailable. This setting is independent from
`tileLimit="auto"`.

### OpenGL tiling and convolution

| Environment variable | Python attribute | Type | Default | Accepted values | Description | Lifecycle notes |
| --- | --- | --- | --- | --- | --- | --- |
| `MATRIXMAN_TILE_LIMIT` | `config.tileLimit` | `int` or `"auto"` | `256` | positive integer, `auto` | Physical OpenGL convolution tile limit. `auto` invokes validated tile autotuning. | Resolved when the OpenGL runtime initializes; ordinary integer values are used as configured. |
| `MATRIXMAN_TILE_SYNC` | `config.tileSync` | `str` | `"per_tile"` | `per_tile`, `end`, `flush`, `none` | Controls synchronization around tiled convolution work. | Read during convolution; changing it does not recreate an OpenGL context. |
| `MATRIXMAN_CONV_SPATIAL_REUSE` | `config.convSpatialReuse` | `bool` | `False` | boolean spellings above | Enables the experimental OpenGL spatial-reuse convolution path. | Set before the convolution that should use it. |
| `MATRIXMAN_SKIP_PRE_CONSOLIDATION_SYNC` | `config.skipPreConsolidationSync` | `bool` | `False` | boolean spellings above | Enables the experimental skip-before-consolidation synchronization path. | Set before the relevant tiled convolution. |
| `MATRIXMAN_DIAGNOSTIC_TILES` | `config.diagnosticTiles` | `bool` | `False` | boolean spellings above | Captures OpenGL tiled-convolution diagnostic snapshots. | Diagnostic-only; affects later tiled convolution dispatches. |
| `MATRIXMAN_DIAGNOSTIC_RECT_TILES` | `config.diagnosticRectTiles` | `bool` | `False` | boolean spellings above | Enables diagnostic rectangular tile dimensions and traversal-order experiments. | Diagnostic-only; affects later tiled convolution dispatches. |
| `MATRIXMAN_DIAG_TILE_WIDTH` | `config.diagTileWidth` | `int` or `None` | `None` | positive integer or unset/empty | Diagnostic tile width when rectangular diagnostics are enabled. | `None` means use the configured tile limit. |
| `MATRIXMAN_DIAG_TILE_HEIGHT` | `config.diagTileHeight` | `int` or `None` | `None` | positive integer or unset/empty | Diagnostic tile height when rectangular diagnostics are enabled. | `None` means use the configured tile limit. |
| `MATRIXMAN_DIAG_TILE_ORDER` | `config.diagTileOrder` | `str` | `"normal"` | `normal`, `reverse`, `column`, `reverse_column` when used by rectangular diagnostics | Selects diagnostic tile traversal order. | Validated when rectangular diagnostic traversal is used. |
| `MATRIXMAN_DIAG_CONV_WORKLOAD` | `config.diagConvWorkload` | `str` | `"heavy"` | `heavy`, `medium`, `light`, `one_by_one` | Selects the workload for the compatibility convolution diagnostic. | Validated by that diagnostic; not a general convolution selector. |

The production OpenGL defaults remain `MATRIXMAN_TILE_LIMIT=256` and
`MATRIXMAN_TILE_SYNC=per_tile`.

### Profiling, diagnostics, and benchmark controls

| Environment variable | Python attribute | Type | Default | Accepted values | Description | Lifecycle notes |
| --- | --- | --- | --- | --- | --- | --- |
| `MATRIXMAN_PROFILE` | `config.profile` | `bool` | `False` | boolean spellings above | Enables profiling for the selected backend. | If a backend is active, a Python assignment updates its profiler; set before initialization for complete coverage. |
| `MATRIXMAN_CUDA_PROFILE` | `config.cudaProfile` | `bool` | `False` | boolean spellings above | Legacy CUDA profiling flag used by `profiling_enabled(legacy_cuda=True)`. | Applies as the legacy CUDA contribution; `profile` is the canonical selected-backend setting. |
| `MATRIXMAN_PROFILE_DETAIL` | `config.profileDetail` | `bool` | `False` | boolean spellings above | Enables detailed OpenGL profiler output. | Read by the OpenGL profiler. |
| `MATRIXMAN_GPU_TIMING` | `config.gpuTiming` | `bool` | `False` | boolean spellings above | Enables deferred OpenGL GPU timer queries when supported. | Read during OpenGL profiler initialization and timing. |
| `MATRIXMAN_TRACE` | `config.trace` | `bool` | `False` | boolean spellings above | Enables high-level MatrixMan operation tracing. | Can be changed at runtime; it does not select or recreate a backend. |
| `MATRIXMAN_DEBUG` | `config.debug` | `bool` | `False` | boolean spellings above | Enables low-level OpenGL diagnostic output. | Separate from high-level `trace`. |
| `MATRIXMAN_GPU_POSTPROCESS` | `config.gpuPostprocess` | `bool` | `False` | boolean spellings above | Enables experimental GPU detection postprocessing in the YOLO benchmark. | Benchmark-only; not a general operator switch. |
| `MATRIXMAN_AUDIT_CPU_LEAKS` | `config.auditCpuLeaks` | `bool` | `False` | boolean spellings above | Enables the CPU-materialization audit. | Consulted by later tensor activity and reported at shutdown. |
| `MATRIXMAN_TILE_AUTOTUNE_REFRESH` | `config.tileAutotuneRefresh` | `bool` | `False` | boolean spellings above | Forces OpenGL tile autotuning to retune instead of using a matching cached entry. | Consulted when OpenGL initializes with `tileLimit="auto"`. |

### CUDA controls

| Environment variable | Python attribute | Type | Default | Accepted values | Description | Lifecycle notes |
| --- | --- | --- | --- | --- | --- | --- |
| `MATRIXMAN_CUDA_DEBUG` | `config.cudaDebug` | `bool` | `False` | boolean spellings above | Prints CUDA/PTX debugging information when available. | Read during CUDA execution/runtime setup. |
| `MATRIXMAN_CUDA_DISABLE_ASYNC_QUEUE` | `config.cudaDisableAsyncQueue` | `bool` | `False` | boolean spellings above | Disables CUDA asynchronous queueing. | Captured when the CUDA execution backend is initialized. |
| `MATRIXMAN_CUDA_DISABLE_ALLOC_POOL` | `config.cudaDisableAllocPool` | `bool` | `False` | boolean spellings above | Disables the CUDA allocation pool. | Captured when the CUDA execution backend is initialized. |
| `MATRIXMAN_CUDA_DISABLE_SPECIALIZED_CONV` | `config.cudaDisableSpecializedConv` | `bool` | `False` | boolean spellings above | Forces the general CUDA convolution path instead of specialized kernels. | Read when CUDA convolution dispatches are selected. |
| `MATRIXMAN_CUDA_CONV3X3_VARIANT` | `config.cudaConv3x3Variant` | `str` | `"plane"` | `plane_legacy` selects the recognized legacy 3x3 specialization; other strings fall through to normal selection | Selects the CUDA 3x3 convolution variant when the specialized shape matches. | Read at CUDA convolution dispatch time; unsupported strings are not rejected by the config parser. |
| `MATRIXMAN_CUDA_LEGACY_MODULE_LOAD` | `config.cudaLegacyModuleLoad` | `bool` | `False` | boolean spellings above | Uses `cuModuleLoadData` instead of `cuModuleLoadDataEx` for the CUDA module. | Captured during CUDA execution-backend initialization. |

## Requested and resolved tile limits

`matrixman.config.tileLimit` is the requested setting. It remains either the
integer requested by the user or the string `"auto"`. `resolvedTileLimit` is a
read-only integer used by OpenGL convolution:

```python
from drivers import matrixman

matrixman.config.tileLimit = "auto"
# After OpenGL initialization, for example:
print(matrixman.config.tileLimit)         # "auto"
print(matrixman.config.resolvedTileLimit) # 1536 (illustrative)
```

An output such as `tileLimit="auto"` and `resolvedTileLimit=1536` means that
autotuning was requested and MatrixMan validated and cached a concrete value
for the current device/driver. `1536` is not a universal or portable value;
the result can differ across GPUs and driver versions.

## OpenGL tile autotuning and cache

Use either of these equivalent forms:

```text
MATRIXMAN_TILE_LIMIT=auto
```

```python
from drivers import matrixman
matrixman.config.tileLimit = "auto"
```

During OpenGL runtime initialization, MatrixMan gathers the vendor, renderer,
OpenGL, and GLSL identity strings. It first looks for a matching validated
entry in the cache. If none exists, or if
`MATRIXMAN_TILE_AUTOTUNE_REFRESH=1` (or
`matrixman.config.tileAutotuneRefresh = True`), it runs the isolated validation
worker, chooses the largest passing configured size, and caches the result for
reuse. Cache entries are keyed by those device/driver identity strings and
include the autotune schema version.

The cache file is:

```text
Windows: %LOCALAPPDATA%\matrixman\opengl_tile_autotune.json
Linux and other non-Windows platforms: $XDG_CACHE_HOME/matrixman/opengl_tile_autotune.json
  or, when XDG_CACHE_HOME is unset, ~/.cache/matrixman/opengl_tile_autotune.json
```

If validation, the worker, or cache I/O fails, autotuning safely resolves to
the validated fallback `256`; synchronization is not changed automatically.
The resolved value is written to `config.resolvedTileLimit`, while the
requested `config.tileLimit` stays `"auto"`.

## Environment examples

PowerShell:

```powershell
$env:MATRIXMAN_TILE_LIMIT="auto"
$env:MATRIXMAN_TILE_SYNC="end"
$env:MATRIXMAN_CONV_SPATIAL_REUSE="1"

python demo/main-tracking.py --imgsz 320
```

Linux:

```bash
MATRIXMAN_TILE_LIMIT=auto \
MATRIXMAN_TILE_SYNC=end \
MATRIXMAN_CONV_SPATIAL_REUSE=1 \
python demo/main-tracking.py --imgsz 320
```

The Python-native equivalent is:

```python
from drivers import matrixman

matrixman.config.tileLimit = "auto"
matrixman.config.tileSync = "end"
matrixman.config.convSpatialReuse = True
matrixman.config.profile = True
matrixman.config.gpuTiming = True
```
