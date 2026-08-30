# MatrixMan

> **Does it run? YES.**  
> **Is it fast? NO.**  
> **Does your GPU run OpenGL? Welcome to MatrixMan.**

MatrixMan is an experimental PyTorch execution backend for GPUs that do not
have a usable native PyTorch compute backend.

No CUDA? No usable native backend? Ancient integrated graphics? But
programmable OpenGL shaders work? 

Excellent.

MatrixMan maps a supported subset of PyTorch tensor operations onto OpenGL and
GLSL fragment shaders. Tensors are packed into floating-point textures, the
shader performs the arithmetic, and framebuffer-backed textures hold the
results.

```text
PyTorch -> MatrixMan -> OpenGL -> fragment shaders -> unsupported/ancient GPU
                                                         -> hopefully numbers
```

OpenGL: “I am rendering pixels.”  
MatrixMan: “You are doing tensor math.”

This is not universal OpenGL magic. Compatibility depends on the driver,
OpenGL/GLSL version, floating-point texture support, and floating-point
framebuffer support.

## Implemented Operation Subset

MatrixMan is experimental and intentionally strict. It implements only the
operations needed by its current examples and model experiments, including:

- Conv2D, grouped convolution, and depthwise convolution
- inference-mode BatchNorm
- in-place SiLU (`aten.silu_.default`)
- implemented elementwise add, subtract, multiply, divide, sigmoid, and scalar add paths
- selected concatenation and stack paths
- MaxPool and nearest-neighbor upsampling paths
- the traced Softmax path
- selected view, reshape, transpose, squeeze, unsqueeze, flatten, expand, split, and broadcast metadata paths
- explicit GPU-to-CPU readback

Unsupported operations fail explicitly. They do not quietly run tensor
arithmetic on the CPU.

## Current Status

The working MatrixMan backend is currently OpenGL. The following are verified
hardware results, not a claim that all GPUs from these vendors or families are
supported:

| Hardware / driver | OpenGL / GLSL | MatrixMan result | YOLO 320 evidence |
| --- | --- | --- | --- |
| Intel GM45 / GMA 4500MHD, Mesa | 2.1 / 1.20 | Compatibility and YOLO path verified; conservative tiling required | historically about 5.4–5.7 s forward |
| Intel HD Graphics 4400, Mesa i915 | 4.6 compatibility profile / 4.60 | Compatibility suite PASS; 512x512 one-shot Conv PASS; tiling not required by that test | 2.210 s forward, 2.623 s total, about 0.38 FPS |
| NVIDIA GT 720M / NVD7, Mesa nouveau | 4.3 compatibility profile / 4.30 | Compatibility suite PASS; 512x512 one-shot Conv PASS; tiling not required by that test | 1.519 s forward, 1.975 s total, about 0.51 FPS |

The GM45 can corrupt large one-shot convolution renders. MatrixMan therefore
keeps physically small convolution tiles as its production baseline. Newer
tested GPUs tolerate larger one-shot renders in the compatibility test, but
that does not change the conservative defaults.

MatrixMan feeds the GM45 smaller tiles. It likes tiny bites.

Run the compatibility probe on a new machine before treating its result as
evidence for a model workload:

```bash
python3 -m drivers.matrixman.compatibility
```

### Experimental / Unverified

Other Intel, NVIDIA, AMD/Radeon, Apple, and accelerator configurations remain
unverified unless listed above. OpenGL support alone does not guarantee
MatrixMan compatibility: driver behavior, floating-point framebuffer support,
shader behavior, texture limits, and the supported ATen operator subset all
matter.

## Check Compatibility

Run the capability and numerical probe:

```bash
python3 -m drivers.matrixman.compatibility
```

Or:

```bash
python3 -m drivers.matrixman --check
```

The probe reports the renderer, OpenGL and GLSL versions, relevant extensions,
hardware limits, numerical self-tests, and whether large convolution tiling is
required. A known GM45-style result looks like:

```text
Float texture upload/readback  PASS
Elementwise shader             PASS
Conv2D                         PASS
BatchNorm                      PASS
SiLU_                          PASS

512x512 production tiled       PASS
512x512 one-shot               FAIL

convolution tiling required: yes
MatrixMan status: COMPATIBLE
```

An optional fast path can fail while MatrixMan remains compatible when its
validated fallback passes.

See [Compatibility and diagnostics](drivers/matrixman/COMPATIBILITY.md) for
verified hardware, diagnostic modes, and the GM45 findings behind these
defaults.

## Architecture

MatrixMan has a small explicit backend interface and currently selects OpenGL
as its only usable backend. The OpenGL implementation is split into modules
under `drivers/matrixman/backends/opengl/`:

```text
backend.py          backend façade and public entrypoint
runtime.py          SDL/OpenGL context and context-owned lifetime
resources.py        textures, uploads, readback resources, and caches
tensor.py           Gm45Tensor and OpenGL texture ownership
metadata.py         logical views, strides, offsets, and split metadata
kernels.py          shader/program lookup and shared kernel helpers
render.py           shared framebuffer and fullscreen-render plumbing
operation_context.py shared operation services
diagnostics.py      tracing and unsupported-operation reporting
profiling.py        timing and OpenGL counters
factories.py        PyTorch PrivateUse1 factory registration
dispatch.py         PyTorch/ATen dispatch bridge
convolution.py      dedicated tiled Conv2D subsystem
ops/                arithmetic, activation, normalization, pooling, resize,
                    concat, softmax, and matmul implementations
```

`implementation.py` is now only a small compatibility re-export façade. It is
not in the normal runtime execution path. The `backends/cuda/` and
`backends/opencl/` directories are reserved placeholder packages only; neither
backend is implemented or selected.

## Basic PyTorch Usage

Application code only needs the public package entry point:

```python
import torch
from drivers import matrixman

a_cpu = torch.randn((2, 2), dtype=torch.float32)
b_cpu = torch.randn((2, 2), dtype=torch.float32)

a = matrixman.to_gm45(a_cpu)
b = matrixman.to_gm45(b_cpu)

c = a + b                 # MatrixMan GPU tensor add
result = c.cpu()          # explicit readback to CPU

print(result)
```

The small example is also available as:

```bash
python3 demo/example/pytorch_example.py
```

## YOLO / Ultralytics Example

The minimal Ultralytics example sends a preprocessed input tensor into the
underlying PyTorch model through MatrixMan:

```python
import torch
from ultralytics import YOLO
from drivers import matrixman

model = YOLO("model.pt").model.eval()
input_mm = matrixman.to_gm45(input_cpu)

with torch.no_grad():
    output_mm = model(input_mm)

# Ultralytics configurations may return a tensor, tuple, list, or dict.
# The runnable example extracts the prediction tensor first, then reads back.
output_cpu = first_tensor(output_mm).cpu()
```

The runnable example accepts a local Ultralytics YOLO detection checkpoint and
an image. It does not download models automatically. A model is only expected
to work when the operations exercised by that particular model are supported:

```bash
python3 demo/example/yolo_example.py model.pt path/to/image.jpg --imgsz 320
```

The execution boundary is deliberately visible:

```text
image -> CPU preprocessing -> PyTorch tensor -> matrixman.to_gm45()
      -> Gm45Tensor -> Ultralytics/PyTorch forward
      -> supported tensor arithmetic through MatrixMan/OpenGL/GLSL
      -> explicit CPU readback -> CPU postprocessing
```

Python `model.forward` and its control flow still execute normally on the
CPU/PyTorch side. MatrixMan intercepts supported tensor operations involving a
`Gm45Tensor` and executes their arithmetic through OpenGL/GLSL.

MatrixMan is not specific to VisDrone. Ultralytics YOLO models can be
attempted when the ATen operations they exercise are supported; this does not
mean every YOLO or Ultralytics model is supported. The current VisDrone-style
checkpoint is tested evidence, not a general Ultralytics compatibility claim.

The historical public names `Gm45Tensor`, `to_gm45()`, and device name
`gm45:0` are retained for compatibility. They identify the current PyTorch
integration and do not imply that execution is restricted to GM45 hardware.

### Tested Model Evidence

The project's confirmed real-model validation uses a custom Ultralytics YOLO
detection checkpoint trained for VisDrone-style detection. The 320x320 timings
listed above are measurements from the verified OpenGL machines; the older
GM45 result is historical. VisDrone is evidence of a working trained model,
not a requirement for using MatrixMan.

## CPU Fallback Policy

CPU work is acceptable for dispatch, Python control flow, metadata, initial
uploads, explicit final readback, application preprocessing/postprocessing,
and narrowly required framework bookkeeping.

Supported tensor arithmetic must not silently move through CPU arithmetic. If
an arithmetic operation is not implemented, MatrixMan fails loudly so the
execution path is visible.

## Installation

Clone the repository:

```bash
git clone https://github.com/NETtoSan/MatrixMan.git
cd MatrixMan
```

This repository currently has no `setup.py`, `pyproject.toml`, or
`requirements.txt`. Installation is therefore manual: provide a compatible
Python environment with PyTorch and NumPy, and install Ultralytics/OpenCV for
the YOLO examples. MatrixMan also needs an SDL2 library and an OpenGL driver
that expose the required functions and floating-point framebuffer/texture
capabilities.

Run the compatibility probe before assuming a particular machine will work.

## Performance

Does it work? **YES.**

Is an Intel GMA 4500MHD a good neural-network accelerator?

**HAHAHAHAHAHAHAHAHA.**

Performance work is ongoing. The reported YOLO timings above are measurements
from the listed machines and are not benchmarks for other hardware or models.
The GM45-safe path retains synchronization and tiling overhead even on newer
tested GPUs because the production defaults must remain correct on GM45.

## Safe Defaults and Compatibility Notes

The conservative production defaults remain:

```text
MATRIXMAN_TILE_LIMIT=256
MATRIXMAN_TILE_SYNC=per_tile
```

The 256x256 physical tile limit and per-tile synchronization are required by
the validated GM45 behavior. The HD 4400 and GT 720M compatibility tests did
not require tiling for their 512x512 one-shot checks, but those results do not
promote larger renders to the general production baseline. Diagnostic tile
controls remain separate from ordinary model execution.

MatrixMan supports only a deliberately limited set of PyTorch operations and
tensor shapes. Unsupported operations fail explicitly rather than silently
performing CPU tensor arithmetic. Model compatibility therefore depends on the
actual ATen operations exercised by that model.

## Future Backends

The backend layout leaves room for future sibling implementations:

```text
drivers/matrixman/backends/
├── opengl/   verified current backend
├── cuda/     placeholder only; not implemented
└── opencl/   placeholder only; not implemented
```

CUDA and OpenCL support are not currently provided. A future backend will need
its own runtime, resource ownership, tensor representation, dispatch bridge,
operator implementations, and framework/factory integration. The current
OpenGL `Gm45Tensor` is not intended to be shared by those backends.

## What MatrixMan Is Not

MatrixMan is not:

- CUDA
- ROCm
- a replacement for a properly supported native PyTorch backend
- feature-complete
- currently fast
- guaranteed to work on every OpenGL GPU

If your GPU already works properly with CUDA, ROCm, MPS, or another maintained
PyTorch accelerator, please use that.

MatrixMan is for the situation where you look at an unsupported GPU and think:

> “But technically it can multiply numbers.”

## Philosophy

```text
Modern ML frameworks:
    This GPU is unsupported.

MatrixMan:
    It has shaders.

Modern ML frameworks:
    There is no supported compute runtime.

MatrixMan:
    It has shaders.

Modern ML frameworks:
    This architecture is from 2008.

MatrixMan:
    IT. HAS. SHADERS.
```

> **PyTorch says your GPU isn't supported?**  
> **Does it draw triangles? Give it to MatrixMan.**

That last line is a project slogan, not a compatibility guarantee. Whether a
GPU actually works is what the compatibility checker is for.
