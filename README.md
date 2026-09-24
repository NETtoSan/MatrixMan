# MatrixMan

## What it is

MatrixMan is an experimental PyTorch backend for running tensor operations on GPUs that are unsupported by modern CUDA, ROCm, or vendor-specific compute stacks.

It uses PyTorch PrivateUse1 with backends including OpenGL and legacy CUDA. The OpenGL backend performs float32 tensor arithmetic through fragment shaders using packed GPU textures.

MatrixMan began as a YOLO/Ultralytics inference project, so some operator coverage and optimizations remain YOLO-centered. Generic PyTorch support is actively expanding; ResNet and simple MLP workloads are increasingly supported, but coverage is incomplete.

Detailed backend, configuration, and testing documentation remains in `guides/` and the other project docs.

## Usage

```python
import torch
from drivers import matrixman

model = ...  # a supported PyTorch model
matrixman.prepare(model)

x = matrixman.to_device(torch.randn(1, 3, 224, 224))

with torch.no_grad():
    y = model(x)

result = y.cpu()  # explicit readback
```

Discover and adjust configuration before backend initialization:

```python
from drivers import matrixman

matrixman.config.describe()

matrixman.config.sync_policy = "safe"
matrixman.config.activation_pool = "safe"
matrixman.config.scratch_pool = "safe"
matrixman.config.tile_limit = 256
```

Unsupported operations fail explicitly. MatrixMan does not silently fall back to CPU tensor arithmetic.

## Disclaimer

MatrixMan is experimental research software. Operator coverage is incomplete, performance varies heavily by GPU and driver, and some features remain designed around YOLO-style workloads.

The project is expanding toward more general PyTorch model compatibility, but it is not a drop-in replacement for CUDA, ROCm, OpenVINO, or production inference runtimes. Expect bugs, missing operators, driver-specific behavior, and questionable decisions involving very old GPUs.
