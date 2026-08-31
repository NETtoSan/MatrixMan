# MatrixMan Examples

MatrixMan provides an experimental PyTorch execution path for GPUs that may
not have a usable native PyTorch compute backend, using graphics and GLSL
capabilities. It is currently verified on Intel GM45 under Linux/Mesa;
compatibility with other GPUs is untested and depends on the installed
OpenGL/GLSL driver and its floating-point texture and framebuffer support.

Run the small tensor example:

```bash
python3 demo/example/pytorch_example.py
```

Run the minimal Ultralytics YOLO example with a local model and image:

```bash
python3 demo/example/yolo_example.py model.pt path/to/image.jpg --imgsz 320
```

`matrixman.to_device(tensor)` uploads a float32 CPU tensor to MatrixMan. Supported
model arithmetic runs through the MatrixMan GPU path, and `tensor.cpu()` is an
explicit readback for the final result and any CPU-side postprocessing.
