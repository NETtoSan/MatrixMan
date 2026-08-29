# MatrixMan Examples

MatrixMan provides a PyTorch-compatible execution path for GPUs that may not
have a supported native PyTorch compute backend, using the GPU's graphics and
GLSL capabilities. Compatibility depends on the installed OpenGL/GLSL driver
and its floating-point texture and framebuffer support.

Run the small tensor example:

```bash
python3 demo/example/pytorch_example.py
```

Run the minimal VisDrone YOLO example with an image:

```bash
python3 demo/example/yolo_example.py path/to/image.jpg --imgsz 320
```

`matrixman.to_gm45(tensor)` uploads a float32 CPU tensor to MatrixMan. Supported
model arithmetic runs through the MatrixMan GPU path, and `tensor.cpu()` is an
explicit readback for the final result and any CPU-side postprocessing.
