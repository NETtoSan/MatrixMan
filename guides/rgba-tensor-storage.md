# MatrixMan Tensor Storage in RGBA32F Textures

MatrixMan gives a logical PyTorch tensor a physical OpenGL representation. The
logical shape remains tensor metadata; OpenGL only receives a 2D texture made
of floating-point RGBA texels.

```text
PyTorch tensor [N, C, H, W]
          |
          v
linear row-major scalar stream
          |
          v
four float32 values per RGBA32F texel
          |
          v
physical 2D OpenGL texture
```

RGBA does not mean that a tensor's channels are assigned to red, green, blue,
and alpha. MatrixMan treats the four RGBA lanes as four arbitrary scalar
storage slots:

```text
t0, t1, t2, t3, t4, t5, t6, t7, ...

texel 0 = [t0, t1, t2, t3]
texel 1 = [t4, t5, t6, t7]
```

OpenGL sees `vec4` float texels. MatrixMan supplies the tensor interpretation.

## Logical Addressing

For a contiguous NCHW tensor, the logical scalar index of
`tensor[n, c, y, x]` is:

```text
linear = ((n * C + c) * H + y) * W + x
```

The scalar stream is packed four values at a time:

```text
texel_index = linear // 4
component   = linear % 4
```

The component selects an RGBA lane:

```text
0 -> R
1 -> G
2 -> B
3 -> A
```

The packed texel index is then mapped into the physical 2D texture. MatrixMan's
current atlas helper computes:

```text
texels = ceil(numel / 4)
texture_width  = max(1, ceil(sqrt(texels)))
texture_height = ceil(texels / texture_width)
```

The physical texel coordinates are therefore:

```text
physical_x = texel_index % texture_width
physical_y = texel_index // texture_width
```

The storage packer represents the atlas as `[height, width, 4]` float32 data
and uploads it as an RGBA32F texture. The last texel may be only partly used.
The packer initializes the unused lanes to zero, but those lanes are outside
the logical tensor and should not be read as tensor values.

## Worked Example

For a tensor with shape `[1, 7, 3, 3]`:

```text
1 * 7 * 3 * 3 = 63 scalar floats
ceil(63 / 4) = 16 RGBA texels
```

The first and last texels look conceptually like this:

```text
texel 0  = [t0,  t1,  t2,  t3]
texel 1  = [t4,  t5,  t6,  t7]
...
texel 14 = [t56, t57, t58, t59]
texel 15 = [t60, t61, t62, unused]
```

For 16 texels, the current atlas dimensions are 4 by 4. The tensor shape is
still `[1, 7, 3, 3]`; `[4, 4]` is only the physical texture shape.

## Larger Tensors

RGBA packing is not a four-neural-channel limit. It is four scalar values per
physical texel. For `[1, 64, 320, 320]`:

```text
64 * 320 * 320 = 6,553,600 scalar floats
6,553,600 / 4 = 1,638,400 RGBA texels
```

The current square-root atlas calculation produces a 1280 by 1280 physical
texture for that exact element count. A tensor with 128 or 256 channels is
handled by the same flatten-and-pack rule.

## Views and Storage Offsets

MatrixMan keeps logical shape, logical strides, and `_storage_offset` as tensor
metadata. A metadata-only view can therefore refer to the same underlying
texture while interpreting a different region or layout.

For a logical index and stride tuple, the storage scalar index is conceptually:

```text
storage_index = storage_offset + sum(index[d] * stride[d])
```

That storage index then uses the same `// 4`, `% 4`, and atlas-coordinate rules.
This is how supported split/view operations can share GPU storage without
copying values. Not every operation accepts every non-contiguous or broadcast
stride pattern; kernels reject layouts they do not implement.

## Conv2D Reads and Writes

For an output scalar such as `output[n, cout, y, x]`, a Conv2D fragment shader
needs logical input locations such as `input[n, cin, y + ky, x + kx]`.
MatrixMan translates each location through the storage model:

```text
tensor[n, c, y, x]
          |
          v
logical scalar index
          |
          v
texel = index // 4, lane = index % 4
          |
          v
physical texture coordinate + RGBA component
```

The shader samples the input and weight RGBA32F textures, selects the required
float lanes, performs its multiply-accumulate work, and writes output values
into another RGBA32F texture. OpenGL does not understand N, C, H, or W;
MatrixMan's metadata and addressing code provide that meaning.

In particular:

```text
Tensor shape != texture shape
```

PyTorch may see `[1, 64, 80, 80]`, while OpenGL sees only a 2D RGBA32F atlas.

## Packing Versus Physical Tiling

These are separate mechanisms:

* **RGBA packing** answers: how are logical scalar values stored inside an
  OpenGL texture?
* **Physical Conv2D tiling** answers: how can a large convolution be rendered
  safely on hardware such as the Intel GM45?

On the validated GM45 path, a large logical convolution can be rendered into
multiple physical output tiles, currently using the conservative default
`MATRIXMAN_TILE_LIMIT=256`, and then consolidated into the normal logical
packed output texture. Tiling changes how a render is dispatched; it does not
change the logical tensor or the four-lane packing rule.

See [MatrixMan compatibility and diagnostics](../drivers/matrixman/COMPATIBILITY.md)
for the GM45-specific findings and diagnostic controls.
