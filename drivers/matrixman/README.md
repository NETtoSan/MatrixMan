# MatrixMan

MatrixMan is an experimental PyTorch `PrivateUse1` backend for Intel GM45.
It dispatches the supported tensor operations to OpenGL 2.1 / GLSL 1.20
fragment shaders and stores float32 tensors in packed `RGBA32F` textures.

The package is organized as:

- `gm45_backend.py`: PyTorch dispatch and GM45 kernels
- `gpumatrix.py`: legacy OpenGL 2.1 context and texture helpers
- `diagnostics/`: standalone regressions, benchmarks, and hardware probes

Basic use:

```python
import torch
from drivers import matrixman

matrixman.init()
x = matrixman.to_gm45(torch.randn(16, 16, dtype=torch.float32))
```

Only explicitly supported operations run on the GM45. Unsupported operations
raise instead of falling back to CPU arithmetic. Set `MATRIXMAN_DEBUG=1` for
detailed dispatch/OpenGL tracing; normal execution uses concise kernel logs.

Run the capability and numerical probe with:

```bash
python3 -m drivers.matrixman.compatibility
# or
python3 -m drivers.matrixman --check
```
