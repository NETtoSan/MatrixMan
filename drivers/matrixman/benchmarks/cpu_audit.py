"""Optional, cross-platform CPU-side measurements for MatrixMan benchmarks."""

from __future__ import annotations

import contextlib
import os
import time
import threading
from collections import defaultdict


_active = threading.local()


class FrameStages:
    """Accumulate wall and process-CPU time for named nested stages."""

    def __init__(self) -> None:
        self.values = defaultdict(lambda: {"wall_seconds": 0.0, "cpu_seconds": 0.0})

    def record(self, name: str, wall: float, cpu: float) -> None:
        self.values[name]["wall_seconds"] += wall
        self.values[name]["cpu_seconds"] += cpu


@contextlib.contextmanager
def frame_stages(enabled: bool = True):
    if not enabled:
        yield None
        return
    previous = getattr(_active, "collector", None)
    collector = FrameStages()
    _active.collector = collector
    try:
        yield collector
    finally:
        _active.collector = previous


@contextlib.contextmanager
def stage(name: str):
    collector = getattr(_active, "collector", None)
    if collector is None:
        yield
        return
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        yield
    finally:
        collector.record(name, time.perf_counter() - wall_started, time.process_time() - cpu_started)


def _rss_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return int(counters.WorkingSetSize)
        except Exception:
            return None
        return None
    try:
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:
        return None


def system_snapshot() -> dict:
    """Return inexpensive process/thread/runtime counters without psutil."""
    snapshot = {
        "rss_bytes": _rss_bytes(),
        "python_threads": threading.active_count(),
        "logical_cpus": os.cpu_count(),
        "torch_threads": None,
        "torch_interop_threads": None,
        "opencv_threads": None,
    }
    try:
        import torch
        snapshot["torch_threads"] = torch.get_num_threads()
        snapshot["torch_interop_threads"] = torch.get_num_interop_threads()
    except Exception:
        pass
    try:
        import cv2
        snapshot["opencv_threads"] = cv2.getNumThreads()
    except Exception:
        pass
    try:
        import psutil
        snapshot["os_threads"] = psutil.Process().num_threads()
    except Exception:
        snapshot["os_threads"] = None
    return snapshot


def memory_snapshot() -> dict:
    result = system_snapshot()
    try:
        from ..backends.opengl.tensor import live_textures
        result["live_texture_owners"] = len(live_textures)
    except Exception:
        result["live_texture_owners"] = None
    try:
        from ..backends.opengl import profiling
        result["texture_allocations"] = int(profiling.counters["texture_allocations"])
        result["parameter_cache_entries"] = len(profiling.parameters)
    except Exception:
        result["texture_allocations"] = None
        result["parameter_cache_entries"] = None
    return result
