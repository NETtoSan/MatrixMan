# MatrixMan GM45 Texture Storage

MatrixMan targets OpenGL 2.1 / GLSL 1.20 on Intel GM45, so it does not assume
texture arrays, shader storage buffers, image load/store, compute shaders, CUDA,
OpenCL, or Vulkan.

## General Tensor Storage

General 1D, 2D, 3D, and 4D tensors use one `GL_TEXTURE_2D` with `RGBA32F`
storage. Tensor values are flattened in normal contiguous PyTorch row-major
order, then packed four float32 values per texel:

```text
linear_index = contiguous row-major tensor offset
texel_index  = linear_index // 4
component    = linear_index % 4
texture_x    = texel_index % texture_width
texture_y    = texel_index // texture_width
```

The RGBA component mapping is:

```text
0 -> R
1 -> G
2 -> B
3 -> A
```

For a YOLO-style NCHW tensor with batch size 1:

```text
shape = [1, C, H, W]
linear_index = ((c * H) + y) * W + x
```

So:

```text
tensor[0, c, y, x]
  -> linear_index = ((c * H) + y) * W + x
  -> texel_index  = linear_index // 4
  -> component    = linear_index % 4
  -> texture coordinate at texel (texture_x, texture_y)
```

This layout preserves contiguous tensors and allows metadata-only reshape/view
operations when the element count is unchanged.

Metadata-only subviews may also carry a logical element offset into the same
texture storage:

```text
physical_linear_index = storage_offset + logical_linear_index
texel_index           = physical_linear_index // 4
component             = physical_linear_index % 4
```

The first supported use is `aten.split.Tensor` for batch-1 NCHW channel splits.
For example:

```text
x.shape = [1, 32, 16, 16]
a, b = torch.split(x, 16, dim=1)

a.shape = [1, 16, 16, 16], storage_offset = 0
b.shape = [1, 16, 16, 16], storage_offset = 16 * 16 * 16 = 4096
```

Both outputs reuse the same OpenGL texture. Later packed GLSL kernels add
`storage_offset` when sampling the input texture. Current offset-aware packed
kernels include Conv2D, eval BatchNorm, SiLU, and elementwise add.

`aten.cat.default` for supported NCHW channel concatenation is not
metadata-only. It launches a fragment shader that samples each input texture
using that input's `storage_offset`, then writes a new contiguous packed output
texture with `storage_offset = 0`.

The detection-head `aten.cat.default` case is also supported for 3D tensors
along the final logical dimension. This is row-wise concatenation, not a flat
append of whole input buffers:

```text
out[0, row, :] = cat(input0[0, row, :], input1[0, row, :], ...)
```

`aten.max_pool2d_with_indices.default` is implemented only for YOLO SPPF values
with `kernel_size=5`, `stride=1`, `padding=2`, `dilation=1`, and
`ceil_mode=False`. The GLSL shader treats padded out-of-bounds locations as
negative infinity. YOLO uses `MaxPool2d(return_indices=False)`, so the backend
returns an empty CPU int64 placeholder for the unused indices output.

`aten.upsample_nearest2d.default` is implemented only for the traced YOLO
nearest-neighbor `2x` NCHW cases. It uses explicit integer packed-element
addressing, not normalized texture filtering, so each output element maps to
`input[c, oy // 2, ox // 2]`.

## Legacy 2D Matrix Storage

Square 2D matrices still use the original shader-compatible layout:

```text
tensor[y, x] -> texture texel (x, y).r
```

The existing GLSL add and matmul kernels depend on this representation, so it is
kept for regression compatibility.

## Current Limitations

- float32 only
- 4D tensors require batch size 1
- only contiguous logical views are represented
- `view`, `reshape`, `flatten`, `squeeze`, `unsqueeze`, and the supported NCHW
  channel split case are metadata-only
- `permute` and `transpose` are not metadata-only yet because the current tensor
  wrapper does not track arbitrary strides in shader indexing
- convolution, eval BatchNorm, and SiLU are implemented for the traced YOLO
  subset only
- packed elementwise add supports identical logical shapes and scalar `alpha`
- packed cat supports batch-1 NCHW `dim=1` channel concatenation for 2-4 inputs
- packed 3D cat supports final-dimension row-wise concatenation for 2-4 inputs
- max pooling supports only the YOLO SPPF 5x5/stride-1/pad-2 values path;
  indices are not computed
- nearest upsampling supports only exact 2x NCHW spatial scaling
