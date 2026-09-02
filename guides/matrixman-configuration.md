# MatrixMan runtime configuration

`from drivers import matrixman` exposes the process configuration as
`matrixman.config`. Environment variables are loaded when MatrixMan's config
module is imported. Python assignments then take precedence until
`matrixman.config.reloadFromEnvironment()` is called. Neither reload nor
assignment creates or destroys an OpenGL context. Settings which affect
backend execution should be assigned before first backend use.

```python
from drivers import matrixman

matrixman.config.tileLimit = "auto"
matrixman.config.tileSync = "per_tile"
matrixman.config.convSpatialReuse = False
print(matrixman.config)
```

`tileLimit` retains the requested value (`"auto"` or an integer), while
`resolvedTileLimit` is the integer currently used by Conv. With
`MATRIXMAN_TILE_LIMIT=auto`, OpenGL uses the cached result of the isolated
physical-tile validation diagnostic or runs that validation and caches the
result. Cache refresh is controlled by `tileAutotuneRefresh`. A complete
autotune failure falls back to the validated 256 limit; tile synchronization
is never changed automatically.

## Environment inventory

The table lists the active `MATRIXMAN_*` settings discovered in the Python
implementation. Diagnostic-only settings are included because they are part
of the supported compatibility tooling. The CUDA settings are retained for
the legacy CUDA backend.

| Environment variable | Attribute | Type | Default | Accepted values | Main code locations |
| --- | --- | --- | --- | --- | --- |
| `MATRIXMAN_BACKEND` | `backend` | string | `auto` | `auto`, `cuda`, `opengl` | `config.py`, `selector.py` |
| `MATRIXMAN_TILE_LIMIT` | `tileLimit` | int/`"auto"` | `256` | positive integer, `auto` | `config.py`, `convolution.py` |
| `MATRIXMAN_TILE_SYNC` | `tileSync` | string | `per_tile` | `per_tile`, `end`, `flush`, `none` | `config.py`, `convolution.py`, `profiling.py` |
| `MATRIXMAN_CONV_SPATIAL_REUSE` | `convSpatialReuse` | bool | `False` | `1/0`, true/false, yes/no, on/off | `config.py`, `convolution.py` |
| `MATRIXMAN_SKIP_PRE_CONSOLIDATION_SYNC` | `skipPreConsolidationSync` | bool | `False` | boolean forms | `config.py`, `convolution.py` |
| `MATRIXMAN_PROFILE` | `profile` | bool | `False` | boolean forms | `config.py`, OpenGL/CUDA profiling |
| `MATRIXMAN_CUDA_PROFILE` | `cudaProfile` | bool | `False` | boolean forms | `config.py`, CUDA profiling |
| `MATRIXMAN_PROFILE_DETAIL` | `profileDetail` | bool | `False` | boolean forms | `config.py`, OpenGL profiling |
| `MATRIXMAN_GPU_TIMING` | `gpuTiming` | bool | `False` | boolean forms | `config.py`, OpenGL profiling |
| `MATRIXMAN_TRACE` | `trace` | bool | `False` | boolean forms | `config.py`, diagnostics |
| `MATRIXMAN_DEBUG` | `debug` | bool | `False` | boolean forms | `config.py`, OpenGL diagnostics |
| `MATRIXMAN_GPU_POSTPROCESS` | `gpuPostprocess` | bool | `False` | boolean forms | `config.py`, YOLO benchmark |
| `MATRIXMAN_AUDIT_CPU_LEAKS` | `auditCpuLeaks` | bool | `False` | boolean forms | `config.py`, `audit.py` |
| `MATRIXMAN_DIAGNOSTIC_TILES` | `diagnosticTiles` | bool | `False` | boolean forms | `config.py`, OpenGL Conv diagnostics |
| `MATRIXMAN_DIAGNOSTIC_RECT_TILES` | `diagnosticRectTiles` | bool | `False` | boolean forms | `config.py`, OpenGL Conv diagnostics |
| `MATRIXMAN_DIAG_TILE_WIDTH` | `diagTileWidth` | int/`None` | `None` | positive integer or unset | `config.py`, compatibility diagnostic |
| `MATRIXMAN_DIAG_TILE_HEIGHT` | `diagTileHeight` | int/`None` | `None` | positive integer or unset | `config.py`, compatibility diagnostic |
| `MATRIXMAN_DIAG_TILE_ORDER` | `diagTileOrder` | string | `normal` | `normal`, `reverse`, `column`, `reverse_column` | `config.py`, convolution |
| `MATRIXMAN_DIAG_CONV_WORKLOAD` | `diagConvWorkload` | string | `heavy` | workload name supported by diagnostic | `config.py`, compatibility diagnostic |
| `MATRIXMAN_TILE_AUTOTUNE_REFRESH` | `tileAutotuneRefresh` | bool | `False` | boolean forms | `config.py`, tile validation/runtime |
| `MATRIXMAN_CUDA_DEBUG` | `cudaDebug` | bool | `False` | boolean forms | `config.py`, CUDA driver backend |
| `MATRIXMAN_CUDA_DISABLE_ASYNC_QUEUE` | `cudaDisableAsyncQueue` | bool | `False` | boolean forms | `config.py`, CUDA driver backend |
| `MATRIXMAN_CUDA_DISABLE_ALLOC_POOL` | `cudaDisableAllocPool` | bool | `False` | boolean forms | `config.py`, CUDA driver backend |
| `MATRIXMAN_CUDA_DISABLE_SPECIALIZED_CONV` | `cudaDisableSpecializedConv` | bool | `False` | boolean forms | `config.py`, CUDA Conv |
| `MATRIXMAN_CUDA_CONV3X3_VARIANT` | `cudaConv3x3Variant` | string | `plane` | `plane`, `plane_legacy` and current variants | `config.py`, CUDA Conv |
| `MATRIXMAN_CUDA_LEGACY_MODULE_LOAD` | `cudaLegacyModuleLoad` | bool | `False` | boolean forms | `config.py`, CUDA driver backend |

`MATRIXMAN_*` names used only to set up an isolated subprocess remain
implementation details of diagnostics; their child processes load the same
configuration layer before backend initialization.
