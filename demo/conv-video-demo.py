#!/usr/bin/env python
"""Live terminal visualization of the first YOLO Conv block through MatrixMan."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.yolo_helpers import preprocess_frame
from drivers import matrixman
from drivers.matrixman.backend import get_backend


PALETTE = " .:-=+*#%@"


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Watch video frames become first-layer YOLO activations in MatrixMan"
    )
    parser.add_argument(
        "video",
        nargs="?",
        type=Path,
        default=base / "videos/video0.mp4",
        help="video file (default: demo/videos/video0.mp4)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=base / "models/VisDrone-arm64-480/weights/best.pt",
        help="Ultralytics YOLO checkpoint",
    )
    parser.add_argument("--imgsz", type=int, default=320, help="square input size (default: 320)")
    channels = parser.add_mutually_exclusive_group()
    channels.add_argument(
        "--channel",
        type=int,
        default=None,
        help="starting channel for automatic one-channel-per-loop cycling (default: 0)",
    )
    channels.add_argument(
        "--channels",
        help="fixed comma-separated channels; does not advance between loops, e.g. 0,1,2,3",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="ASCII heatmap width (default scales from 72 for one channel)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="ASCII heatmap height (default scales from 28 for one channel)",
    )
    parser.add_argument("--fps-limit", type=float, default=0.0, help="optional display rate limit")
    parser.add_argument("--frames", type=int, default=0, help="stop after N frames; 0 means until EOF")
    parser.add_argument("--no-loop", action="store_true", help="stop at video EOF instead of replaying")
    parser.add_argument("--show-input", action="store_true", help="show a small grayscale source preview")
    parser.add_argument("--no-ansi", action="store_true", help="append frames instead of refreshing in place")
    return parser.parse_args()


def parse_channels(args: argparse.Namespace) -> list[int]:
    if args.channels is None:
        return [0 if args.channel is None else args.channel]
    try:
        values = [int(value.strip()) for value in args.channels.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError("--channels must be a comma-separated list of integers") from exc
    if not values:
        raise ValueError("--channels must contain at least one channel")
    return values


def default_heatmap_size(channel_count: int) -> tuple[int, int]:
    """Choose a readable default that keeps multi-channel layouts practical."""
    if channel_count == 1:
        return 72, 28
    if channel_count == 2:
        return 48, 22
    if channel_count <= 4:
        return 36, 18
    return 32, 16


def find_first_conv_block(model: nn.Module) -> tuple[str, nn.Module]:
    """Find the first Ultralytics Conv-like block, based on its actual structure."""
    for name, module in model.named_modules(remove_duplicate=False):
        conv = getattr(module, "conv", None)
        if (
            module.__class__.__name__ in {"Conv", "DWConv"}
            and isinstance(conv, nn.Conv2d)
        ):
            return name or "<root>", module
    raise RuntimeError("could not find a Conv/DWConv block in the loaded YOLO model")


def ascii_heatmap(feature_map: torch.Tensor, width: int, height: int) -> list[str]:
    values = feature_map.detach().cpu().float().numpy()
    small = cv2.resize(values, (width, height), interpolation=cv2.INTER_AREA)
    low, high = float(small.min()), float(small.max())
    if high > low:
        scaled = (small - low) / (high - low)
    else:
        scaled = small * 0.0
    indexes = (scaled * (len(PALETTE) - 1)).round().astype("int32")
    return ["".join(PALETTE[index] for index in row) for row in indexes]


def input_preview(frame: torch.Tensor, width: int = 28, height: int = 10) -> list[str]:
    values = frame.detach().cpu().float().mean(dim=0).numpy()
    return ascii_heatmap(torch.from_numpy(values), width, height)


def channel_stack_lines(channel_count: int, selected_channels: list[int]) -> list[str]:
    """Render metadata-only channel markers without reading additional data."""
    selected = sorted(set(selected_channels))
    if channel_count <= 16:
        ranges = [(0, channel_count)]
    else:
        # Keep a compact window around every explicitly selected channel so
        # multi-channel mode can still mark all of its requested channels.
        window = 9
        ranges = []
        for channel in selected:
            start = max(0, min(channel - window // 2, channel_count - window))
            end = min(channel_count, start + window)
            if ranges and start <= ranges[-1][1]:
                ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
            else:
                ranges.append((start, end))

    if len(ranges) == 1:
        start, end = ranges[0]
        title = f"Activation channels ({start}-{end - 1} / {channel_count})"
    else:
        title = f"Activation channels (selected windows / {channel_count})"
    lines = [title]
    for range_index, (start, end) in enumerate(ranges):
        if range_index > 0 or start:
            lines.append("  ...")
        for channel in range(start, end):
            marker = "  <---- viewing" if channel in selected else ""
            lines.append(f"  [ch {channel}]{marker}")
        if range_index == len(ranges) - 1 and end < channel_count:
            lines.append("  ...")
    return lines


def draw_screen(text: str, *, refresh: bool, previous_lines: int) -> int:
    lines = text.splitlines()
    if refresh:
        # The cursor is saved at the start of the demo UI on the first frame.
        # Restore that position for later frames so MatrixMan probe/startup
        # output above the UI is never overwritten.
        cursor = "\x1b[s" if previous_lines == 0 else "\x1b[u"
        sys.stdout.write(cursor + text + "\x1b[J")
        sys.stdout.flush()
    else:
        if previous_lines:
            sys.stdout.write("\n")
        sys.stdout.write(text + "\n")
        sys.stdout.flush()
    return len(lines)


def rewind_video(cap: cv2.VideoCapture, video_path: Path) -> cv2.VideoCapture:
    """Reopen the same stream at frame zero without touching model state."""
    # A few codecs report success from CAP_PROP_POS_FRAMES but still return
    # EOF on the next read. Reopening is slightly more conservative and keeps
    # the loop behavior reliable across OpenCV backends.
    cap.release()
    reopened = cv2.VideoCapture(str(video_path))
    if not reopened.isOpened():
        raise RuntimeError(f"could not reopen video after EOF: {video_path}")
    return reopened


def main() -> int:
    args = parse_args()
    if args.imgsz <= 0 or args.imgsz > 640 or args.imgsz % 32:
        raise ValueError("--imgsz must be a positive multiple of 32 and no greater than 640")
    channels = parse_channels(args)
    default_width, default_height = default_heatmap_size(len(channels))
    heatmap_width = default_width if args.width is None else args.width
    heatmap_height = default_height if args.height is None else args.height
    if heatmap_width <= 0 or heatmap_height <= 0 or args.fps_limit < 0:
        raise ValueError("--width, --height, and --fps-limit must be positive/non-negative")

    video_path = args.video.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if args.frames < 0:
        raise ValueError("--frames must be non-negative")
    # MATRIXMAN_BACKEND is the public backend selector used by the other demos.
    requested_backend = os.environ.get("MATRIXMAN_BACKEND", "auto").strip().lower() or "auto"
    if requested_backend not in {"auto", "cuda", "opengl"}:
        raise ValueError("MATRIXMAN_BACKEND must be auto, cuda, or opengl")
    matrixman.prefer(requested_backend)

    from ultralytics import YOLO

    matrixman.init()
    cap = None
    previous_lines = 0
    try:
        info = get_backend().device_info()
        backend_name = str(info.get("backend", get_backend().name)).lower()
        device_name = info.get("renderer") or info.get("device") or info.get("name") or "unknown"

        yolo = YOLO(str(model_path))
        model = yolo.model.eval()
        block_name, first_block = find_first_conv_block(model)
        # Preparation is intentionally scoped to the selected block, not the YOLO model.
        first_block = matrixman.prepare(first_block, backend=backend_name)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"could not open video: {video_path}")

        refresh = bool(sys.stdout.isatty() and not args.no_ansi)
        frame_number = 0
        total_frames = 0
        loop_number = 1
        automatic_channel = 0 if args.channel is None else args.channel
        fixed_channels = args.channels is not None
        channel_count: int | None = None
        started = time.perf_counter()
        while args.frames == 0 or total_frames < args.frames:
            ok, frame = cap.read()
            if not ok:
                if args.no_loop:
                    break
                if channel_count is None:
                    raise RuntimeError("video produced no frames; cannot determine activation channel count")
                loop_number += 1
                frame_number = 0
                if not fixed_channels:
                    automatic_channel = (automatic_channel + 1) % channel_count
                cap = rewind_video(cap, video_path)
                continue
            frame_started = time.perf_counter()

            # CPU preprocessing ends at the explicit MatrixMan upload boundary.
            cpu_input = preprocess_frame(frame, args.imgsz)
            frame_tensor = cpu_input.to(dtype=torch.float32)
            frame_tensor = matrixman.to_device(frame_tensor)
            with torch.no_grad():
                activation = first_block(frame_tensor)

            readback_started = time.perf_counter()
            activation_cpu = activation.cpu()
            readback_ms = (time.perf_counter() - readback_started) * 1000.0
            if activation_cpu.ndim != 4 or activation_cpu.shape[0] != 1:
                raise RuntimeError(f"unexpected first block output shape: {list(activation_cpu.shape)}")
            channel_count = int(activation_cpu.shape[1])
            display_channels = channels if fixed_channels else [automatic_channel]
            for channel in display_channels:
                if channel < 0 or channel >= channel_count:
                    raise ValueError(f"channel {channel} is outside [0, {channel_count - 1}]")

            elapsed = time.perf_counter() - frame_started
            total_elapsed = max(time.perf_counter() - started, 1e-9)
            lines = [
                "MatrixMan live Conv demo",
                f"backend: {backend_name.upper()}",
                f"device: {device_name}",
                f"model block: {block_name} ({first_block.__class__.__name__})",
                f"video loop: {loop_number}",
                f"video frame: {frame_number}",
                f"input:  {list(frame_tensor.shape)}",
                f"output: {list(activation_cpu.shape)}",
            ]
            if fixed_channels:
                lines.append(f"channels: {','.join(map(str, display_channels))} (fixed selection)")
            else:
                lines.append(f"visualizing channel: {automatic_channel} / {channel_count - 1}")
            lines += [
                f"Conv block: {elapsed * 1000.0:.1f} ms",
                f"readback:   {readback_ms:.1f} ms",
                f"FPS:        {(total_frames + 1) / total_elapsed:.2f}",
            ]
            if args.show_input:
                lines += ["", "ORIGINAL FRAME"] + input_preview(cpu_input[0], 28, 10)
            lines += [
                "",
                "INPUT",
                f"  {list(frame_tensor.shape)}",
                "    |",
                "    v",
                "FIRST CONV BLOCK",
                f"  {list(activation_cpu.shape)}",
                "    |",
                "    v",
            ]
            lines += channel_stack_lines(channel_count, display_channels)
            lines += ["    |", "    v", "ASCII ACTIVATION"]
            for channel in display_channels:
                lines += [f"", f"channel {channel}"] + ascii_heatmap(
                    activation_cpu[0, channel], heatmap_width, heatmap_height
                )

            previous_lines = draw_screen("\n".join(lines), refresh=refresh, previous_lines=previous_lines)
            frame_number += 1
            total_frames += 1
            if args.fps_limit:
                remaining = (1.0 / args.fps_limit) - (time.perf_counter() - frame_started)
                if remaining > 0:
                    time.sleep(remaining)
        if refresh:
            sys.stdout.write("\n")
        print(f"completed frames: {total_frames}")
        return 0
    finally:
        if cap is not None:
            cap.release()
        matrixman.shutdown()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        raise SystemExit(130)
