"""Pure tensor packing and logical storage helpers for MatrixMan."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class StorageLayout:
    kind: str
    texture_width: int
    texture_height: int
    numel: int


def numel(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


def contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    strides = []
    for size in reversed(shape):
        strides.append(stride)
        stride *= size
    return tuple(reversed(strides))


def max_storage_index(shape: tuple[int, ...], strides: tuple[int, ...]) -> int:
    if numel(shape) == 0:
        return 0
    return sum((size - 1) * stride for size, stride in zip(shape, strides))


def packed_atlas_size(element_count: int) -> tuple[int, int]:
    texels = (element_count + 3) // 4
    width = max(1, math.ceil(math.sqrt(texels)))
    height = math.ceil(texels / width)
    return width, height


def pack_linear_rgba(matrix: np.ndarray) -> tuple[np.ndarray, StorageLayout]:
    flat = np.ascontiguousarray(matrix, dtype=np.float32).reshape(-1)
    width, height = packed_atlas_size(flat.size)
    rgba = np.zeros((height, width, 4), dtype=np.float32)
    rgba.reshape(-1)[: flat.size] = flat
    return rgba, StorageLayout("packed_rgba", width, height, flat.size)


def matrix_red_rgba(matrix: np.ndarray) -> tuple[np.ndarray, StorageLayout]:
    n = matrix.shape[0]
    rgba = np.zeros((n, n, 4), dtype=np.float32)
    rgba[:, :, 0] = matrix.astype(np.float32)
    rgba[:, :, 3] = 1.0
    return rgba, StorageLayout("matrix2d_red", n, n, n * n)
