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

## Current Status

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

## Tested Hardware

MatrixMan is currently confirmed working on:

```text
ThinkPad X200
Intel GMA 4500MHD / GM45
Mesa
OpenGL 2.1
GLSL 1.20
YOLO inference: working
```

The GM45 can corrupt the large one-shot convolution render path. MatrixMan
uses physically small convolution tiles, up to 256x256 texels, and has
validated the tiled path on this machine.

MatrixMan feeds the GM45 smaller tiles. It likes tiny bites.

That is a detected/validated quirk of this path, not a universal limitation of
all OpenGL GPUs.

Other GPUs are currently unverified. OpenGL support alone does not guarantee
MatrixMan compatibility: driver behavior, floating-point framebuffer support,
shader behavior, texture limits, and other implementation details matter.
Run the compatibility probe before assuming a machine works.

### Experimental / Unverified

Older AMD/Radeon hardware, NVIDIA GPUs, Apple GPUs, and Intel GPUs beyond the
X200's GM45 are portability experiments or future goals, not tested MatrixMan
platforms. The project is exploring whether such hardware can provide a
usable graphics-based execution path; in other words, we dont have all the old toys to test with! 

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

### Tested Model Evidence

The project's confirmed real-model validation uses a custom Ultralytics YOLO
detection checkpoint trained for VisDrone-style detection. The current
checkpoint has been tested at 320x320 through 640x640 on GM45. VisDrone is
evidence of a working trained model, not a requirement for using MatrixMan.

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

Performance work is ongoing. Likely areas include synchronization overhead,
tiled convolution overhead, parameter texture caching, Conv+BatchNorm
folding, Conv+BatchNorm+SiLU fusion, shader optimization, and reducing
unnecessary texture consolidation.

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
