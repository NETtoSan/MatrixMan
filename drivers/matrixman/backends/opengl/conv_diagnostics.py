"""Single-window Conv2D diagnostics viewer.

The viewer is intentionally lazy and diagnostic-only.  It uses the existing
OpenCV/HighGUI dependency used by the demos, and never creates a window or
performs a readback unless ``config.convDiag`` is enabled.
"""

from __future__ import annotations

import math


WINDOW_NAME = "MatrixMan Conv Diagnostics"

_cv2 = None
_np = None
_window_created = False
_window_create_count = 0
_paused = False
_live = True
_input_channel = 0
_output_channel = 0
_latest = None
_capture_count = 0
_raw_readback_count = 0
_status = "RUN"
_input_shape = "[]"
_output_shape = "[]"


def _load_gui():
    global _cv2, _np
    if _cv2 is None:
        import cv2
        import numpy as np

        _cv2 = cv2
        _np = np
    return _cv2, _np


def enabled() -> bool:
    from ...config import config

    return bool(config.convDiag)


def _ensure_window() -> None:
    global _window_created, _window_create_count
    if _window_created:
        return
    cv2, _ = _load_gui()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1580, 620)
    _window_created = True
    _window_create_count += 1


def _finite_heatmap(values, size=(360, 300)):
    cv2, np = _load_gui()
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        array = np.asarray(array).reshape(1, -1)
    finite = np.isfinite(array)
    if finite.any():
        valid = array[finite]
        low = float(valid.min())
        high = float(valid.max())
        if not math.isfinite(low) or not math.isfinite(high):
            low, high = 0.0, 1.0
        if high > low:
            normalized = np.zeros(array.shape, dtype=np.float32)
            normalized[finite] = (array[finite] - low) / (high - low)
        else:
            normalized = np.full(array.shape, 0.5, dtype=np.float32)
    else:
        normalized = np.zeros(array.shape, dtype=np.float32)
    image = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    image = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
    image[~finite] = (0, 0, 255)
    return cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)


def _raw_texture_image(values):
    """Map one physical RGBA texel to one source display pixel.

    The array is intentionally never flattened, transposed, reshaped, or
    converted through a logical tensor view.  Its first two dimensions are
    the physical OpenGL texture height and width, and its last dimension is
    the stored RGBA lane group.
    """
    cv2, np = _load_gui()
    raw = np.asarray(values, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[2] != 4:
        raise ValueError(f"expected physical RGBA texture, got shape {raw.shape}")

    normalized = np.zeros(raw.shape, dtype=np.float32)
    for lane in range(4):
        channel = raw[:, :, lane]
        finite = np.isfinite(channel)
        if not finite.any():
            continue
        valid = channel[finite]
        low = float(valid.min())
        high = float(valid.max())
        if math.isfinite(low) and math.isfinite(high) and high > low:
            normalized[:, :, lane][finite] = (channel[finite] - low) / (high - low)
        else:
            normalized[:, :, lane][finite] = 0.5

    # OpenCV stores colors as BGR, while the source texture is RGBA.  A is
    # kept associated with its texel and contributes to visible brightness.
    image = np.empty(raw.shape[:2] + (3,), dtype=np.float32)
    image[:, :, 0] = normalized[:, :, 2]
    image[:, :, 1] = normalized[:, :, 1]
    image[:, :, 2] = normalized[:, :, 0]
    image *= (0.5 + 0.5 * normalized[:, :, 3])[:, :, None]
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def _fit_physical_image(image, size=(356, 300)):
    """Scale a physical image without changing its width/height ratio."""
    cv2, np = _load_gui()
    target_width, target_height = size
    source_height, source_width = image.shape[:2]
    scale = min(target_width / max(1, source_width),
                target_height / max(1, source_height))
    width = max(1, int(round(source_width * scale)))
    height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x = (target_width - width) // 2
    y = (target_height - height) // 2
    result[y:y + height, x:x + width] = resized
    return result


def _text(canvas, value, origin, scale=0.52, color=(230, 230, 230), thickness=1):
    cv2, _ = _load_gui()
    cv2.putText(canvas, str(value), origin, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _shape(value) -> str:
    return "[" + ",".join(str(int(item)) for item in value) + "]"


def _panel(title, values=None, *, raw=False):
    cv2, np = _load_gui()
    panel = np.zeros((360, 380, 3), dtype=np.uint8)
    panel[:] = (28, 28, 28)
    _text(panel, title, (12, 24), scale=0.62, color=(255, 255, 255), thickness=2)
    if values is None:
        heatmap = np.zeros((300, 356, 3), dtype=np.uint8)
    elif raw:
        heatmap = _fit_physical_image(_raw_texture_image(values))
    else:
        heatmap = _finite_heatmap(values, size=(356, 300))
    panel[38:338, 12:368] = heatmap
    return panel


def _render_status_dots(canvas):
    """Render the fixed three-position status indicator without labels."""
    cv2, _ = _load_gui()
    inactive = (55, 55, 55)
    compile_active = (0, 220, 255)  # yellow in OpenCV's BGR order
    run_active = (0, 200, 0)

    if _status == "COMPILE":
        colors = (compile_active, inactive, inactive)
    elif _status == "READBACK":
        colors = (inactive, compile_active, inactive)
    else:
        colors = (inactive, inactive, run_active)

    for x, color in zip((30, 70, 110), colors):
        cv2.circle(canvas, (x, 30), 10, color, -1, lineType=cv2.LINE_AA)


def _render_latest():
    cv2, np = _load_gui()
    if _latest is None:
        panels = [_panel("INPUT"), _panel("KERNEL"), _panel("OUTPUT"), _panel("OPENGL RAW")]
    else:
        input_values = _latest["input_values"]
        output_values = _latest["output_values"]
        kernel_values = _latest["kernel_values"]
        raw_texture_values = _latest["raw_texture_values"]
        input_channel = _input_channel % max(1, int(input_values.shape[1]))
        output_channel = _output_channel % max(1, int(output_values.shape[1]))
        kernel_input_channel = input_channel % max(1, int(kernel_values.shape[1]))
        panels = [
            _panel("INPUT", input_values[0, input_channel]),
            _panel("KERNEL", kernel_values[output_channel, kernel_input_channel]),
            _panel("OUTPUT", output_values[0, output_channel]),
            _panel("OPENGL RAW", raw_texture_values, raw=True),
        ]
    canvas = np.zeros((580, 1560, 3), dtype=np.uint8)
    canvas[:] = (18, 18, 18)
    _render_status_dots(canvas)
    _text(canvas, f"Conv {_input_shape} -> {_output_shape}", (18, 70), scale=0.62)
    for index, panel in enumerate(panels):
        x = 10 + index * 385
        canvas[150:510, x:x + 380] = panel
    cv2.imshow(WINDOW_NAME, canvas)


def _handle_keys():
    global _paused, _live, _input_channel, _output_channel
    cv2, _ = _load_gui()
    key = cv2.waitKey(1) & 0xFF
    if key == 32:  # SPACE
        _paused = not _paused
        _live = not _paused
        _render_latest()
    elif key in (ord("r"), ord("R")):
        _paused = False
        _live = True
        _render_latest()
    elif key in (ord("i"), ord("I")) and _latest is not None:
        _input_channel += 1
        _render_latest()
    elif key in (ord("c"), ord("C"), ord("o"), ord("O")) and _latest is not None:
        _output_channel += 1
        _render_latest()


def update(*, input_tensor, weight_tensor, output_tensor, index: int,
           kernel: str, stride: str, padding: str, program: str,
           prepared: bool, fused: str) -> None:
    """Read back the visible Conv slices and update the persistent window."""
    global _latest, _capture_count, _raw_readback_count, _status
    if not enabled():
        return
    if _paused and not _live:
        if _window_created:
            _handle_keys()
        return
    set_context(input_tensor.shape, output_tensor.shape)
    set_status("READBACK")
    from . import tensor as tensor_module

    input_values = tensor_module.readback_tensor(
        input_tensor._owner, tuple(int(value) for value in input_tensor.shape),
        int(input_tensor._storage_offset), tuple(int(value) for value in input_tensor._logical_strides),
    ).numpy()
    output_values = tensor_module.readback_tensor(
        output_tensor._owner, tuple(int(value) for value in output_tensor.shape),
        int(output_tensor._storage_offset), tuple(int(value) for value in output_tensor._logical_strides),
    ).numpy()
    from . import resources, runtime

    # The logical readback above is for the existing INPUT/OUTPUT panel.  This
    # separate read uses the output owner's physical RGBA32F attachment
    # directly, preserving its texture geometry and texel/lane ordering.
    raw_texture_values = resources.read_texture_pixels(
        output_tensor._owner, runtime.runtime_required().fbo
    )
    _raw_readback_count += 1
    kernel_values = weight_tensor.detach().to(device="cpu").numpy()
    _status = "RUN"
    _latest = {
        "index": int(index),
        "input_shape": _shape(input_values.shape),
        "output_shape": _shape(output_values.shape),
        "kernel": kernel,
        "stride": stride,
        "padding": padding,
        "program": program,
        "prepared": "yes" if prepared else "no",
        "fused": fused,
        "input_physical": f"{input_tensor._owner.layout.texture_width}x{input_tensor._owner.layout.texture_height}",
        "output_physical": f"{output_tensor._owner.layout.texture_width}x{output_tensor._owner.layout.texture_height}",
        "input_values": input_values,
        "kernel_values": kernel_values,
        "output_values": output_values,
        "raw_texture_values": raw_texture_values,
    }
    _capture_count += 1
    _render_latest()
    _handle_keys()


def close() -> None:
    """Destroy the one diagnostics window and reset GUI state."""
    global _window_created, _paused, _live, _latest, _capture_count
    global _raw_readback_count, _status
    if _window_created and _cv2 is not None:
        try:
            _cv2.destroyWindow(WINDOW_NAME)
            _cv2.waitKey(1)
        except Exception:
            pass
    _window_created = False
    _paused = False
    _live = True
    _latest = None
    _capture_count = 0
    _raw_readback_count = 0
    _status = "RUN"


def set_context(input_shape, output_shape) -> None:
    global _input_shape, _output_shape
    _input_shape = _shape(input_shape)
    _output_shape = _shape(output_shape)


def set_status(status: str) -> None:
    global _status
    if status not in {"COMPILE", "READBACK", "RUN"}:
        raise ValueError(f"invalid Conv diagnostics status: {status!r}")
    _status = status
    if enabled():
        _ensure_window()
        _render_latest()
        _handle_keys()


def snapshot() -> dict:
    return {
        "enabled": enabled(),
        "window_created": _window_created,
        "window_create_count": _window_create_count,
        "capture_count": _capture_count,
        "raw_readback_count": _raw_readback_count,
        "raw_texture_shape": (
            tuple(int(value) for value in _latest["raw_texture_values"].shape)
            if _latest is not None else None
        ),
        "paused": _paused,
    }
