# MatrixMan

MatrixMan is an experimental PyTorch `PrivateUse1` backend, with the validated
PrivateUse1 operator path currently running on Intel GM45 through OpenGL 2.1 /
GLSL 1.20 fragment shaders. A legacy CUDA backend can now be selected when its
device is available, but CUDA PrivateUse1 operator coverage is not yet
implemented. Other GPUs remain unverified.

The package is organized as:

- `backend.py`: canonical backend-neutral MatrixMan frontend and backend access
- `gm45_backend.py`: deprecated compatibility re-export for historical imports
- `gpumatrix.py`: legacy OpenGL 2.1 context and texture helpers
- `diagnostics/`: standalone regressions, benchmarks, and hardware probes

Basic use:

```python
import torch
from drivers import matrixman

matrixman.init()
x = matrixman.to_device(torch.randn(16, 16, dtype=torch.float32))
```

Only explicitly supported operations run on the GM45. Unsupported operations
raise instead of falling back to CPU arithmetic. Set `MATRIXMAN_TRACE=1` (or
`matrixman.trace = True`) for high-level operation logs. The CUDA-specific
`MATRIXMAN_CUDA_DEBUG=1` flag remains available for low-level implementation
diagnostics; normal execution is quiet.

For CUDA and OpenGL inference, `matrixman.auto_prepare(model)` returns a narrow
model-level wrapper that prepares an eligible eval model on its first MatrixMan
inference and reuses the backend-specific cached prepared copy afterward. The
original model is not mutated. The first inference includes preparation cost;
call `matrixman.prepare(model, backend="cuda")` or
`matrixman.prepare(model, backend="opengl")` when eager preparation is preferred.
Training and autograd-enabled calls bypass lazy preparation. Disable it with
`$env:MATRIXMAN_DISABLE_AUTO_PREPARE="1"`; explicit `matrixman.prepare()` is
still available. Raw `model(matrixman_tensor)` calls cannot be intercepted
safely without global PyTorch monkey-patching, so the wrapper is required for
lazy behavior.

The tracking demo enables this wrapper by default:

```powershell
$env:MATRIXMAN_BACKEND="cuda"
python demo/main-tracking.py --imgsz 320
```

Use `$env:MATRIXMAN_BACKEND="opengl"` for the OpenGL path. Use
`--no-auto-prepare` for an unprepared baseline, or `--prepare` when eager
preparation before the first frame is preferred.

The minimal inference pattern is:

```python
from drivers import matrixman

model.eval()
model = matrixman.auto_prepare(model)
input_tensor = matrixman.to_device(input_tensor)
with torch.no_grad():
    prediction = model(input_tensor)
prediction = prediction.cpu()
```

Run the capability and numerical probe with:

```bash
python -m drivers.matrixman.compatibility
# or
python -m drivers.matrixman --check
```
