"""Probe PrivateUse1 autograd/device plumbing without testing gradient math."""

from __future__ import annotations

import sys
import threading
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drivers import matrixman
from drivers.matrixman.tensor import MatrixManTensor


def _install_dispatch_probe() -> None:
    """Wrap only this diagnostic's dispatch entrypoint with failure context."""
    from drivers.matrixman.backends.opengl import dispatch as dispatch_module

    original = dispatch_module.handle_torch_dispatch
    if getattr(original, "_autograd_probe_wrapper", False):
        return

    def describe(value):
        if isinstance(value, MatrixManTensor):
            return {
                "shape": list(value.shape),
                "device": str(value.device),
                "dtype": str(value.dtype),
                "requires_grad": bool(value.requires_grad),
                "is_leaf": bool(value.is_leaf),
                "texture": int(value._owner.texture),
                "storage": value._owner.layout.kind,
                "storage_offset": int(value._storage_offset),
                "logical_strides": list(value._logical_strides),
            }
        if isinstance(value, torch.Tensor):
            return {
                "shape": list(value.shape),
                "device": str(value.device),
                "dtype": str(value.dtype),
                "requires_grad": bool(value.requires_grad),
                "is_leaf": bool(value.is_leaf),
            }
        if isinstance(value, (tuple, list)):
            return [describe(item) for item in value]
        if isinstance(value, dict):
            return {str(key): describe(item) for key, item in value.items()}
        return repr(value)

    def gl_identity():
        try:
            from drivers.matrixman.backends.opengl import gpumatrix as gm

            def text(value):
                return value.decode("utf-8", "replace") if value else None

            return {
                "version": text(gm.glGetString(0x1F02)),
                "renderer": text(gm.glGetString(0x1F01)),
            }
        except BaseException as error:
            return {"query_error": f"{type(error).__name__}: {error!r}"}

    def probed(cls, func, types, args=(), kwargs=None):
        print(
            f"[MatrixMan autograd probe] dispatch func={func} "
            f"thread_id={threading.get_ident()} thread_name={threading.current_thread().name}"
        )
        print(f"  gl_identity={gl_identity()}")
        print(f"  args={describe(args)}")
        print(f"  kwargs={describe(kwargs or {})}")
        try:
            return original(cls, func, types, args, kwargs)
        except BaseException as error:
            print(f"[MatrixMan autograd probe] dispatch failure func={func}")
            print(f"  exception_type={type(error).__name__}")
            print(f"  exception_str={str(error)}")
            print(f"  exception_repr={error!r}")
            traceback.print_exc()
            raise

    probed._autograd_probe_wrapper = True
    dispatch_module.handle_torch_dispatch = probed


def _describe_grad(grad) -> None:
    if grad is None:
        print("x.grad is None: yes")
        return
    print("x.grad is None: no")
    print("  grad_type:", type(grad).__name__)
    print("  grad_device:", grad.device)
    print("  grad_shape:", list(grad.shape))
    print("  grad_requires_grad:", bool(grad.requires_grad))
    print("  grad_is_leaf:", bool(grad.is_leaf))
    if hasattr(grad, "_owner"):
        print(
            "  grad_storage:", grad._owner.layout.kind,
            "texture=", int(grad._owner.texture),
            "storage_offset=", int(grad._storage_offset),
            "logical_strides=", list(grad._logical_strides),
        )
    values = grad.cpu()
    print("  grad_cpu_values:", values.tolist())


def main() -> None:
    matrixman.set_tracing(True)
    _install_dispatch_probe()
    print(
        "[MatrixMan autograd probe] initialization "
        f"thread_id={threading.get_ident()} thread_name={threading.current_thread().name}"
    )
    x = matrixman.to_device(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
    print("torch_version:", torch.__version__)
    print("device:", x.device)
    print("get_device:", x.get_device())
    try:
        x.requires_grad_(True)
        print("requires_grad: PASS", x.requires_grad)
    except Exception as error:
        print("requires_grad: BLOCKED", type(error).__name__, str(error))
        return

    y = None
    try:
        y = x + x
        print("graph_output_requires_grad:", y.requires_grad)
        print("graph_output_device:", y.device)
    except Exception as error:
        print("graph: BLOCKED", type(error).__name__, str(error))

    if y is None:
        print("backward: SKIPPED because graph construction failed")
        return

    try:
        print("[MatrixMan autograd probe] main-thread expand prewarm")
        prewarm_scalar = x.sum()
        prewarm = prewarm_scalar.expand((3,))
        print(
            "  prewarm result:", list(prewarm.shape),
            "thread_id=", threading.get_ident(),
            "gl_identity=", end="\n",
        )
        from drivers.matrixman.backends.opengl import gpumatrix as gm
        print({
            "version": gm.glGetString(0x1F02),
            "renderer": gm.glGetString(0x1F01),
        })
    except BaseException as error:
        print("  prewarm: FAILED", type(error).__name__, str(error), repr(error))
        traceback.print_exc()

    try:
        y.sum().backward()
        print("backward: PASS")
        _describe_grad(x.grad)
        if x.grad is None:
            print("autograd bootstrap: FAIL (x.grad is None)")
            return
        torch.testing.assert_close(
            x.grad.cpu(), torch.full_like(torch.tensor([1.0, 2.0, 3.0]), 2.0),
            rtol=0, atol=0,
        )
        print("first backward gradient: PASS [2.0, 2.0, 2.0]")

        accumulation_graph = (x + x).sum()
        accumulation_graph.backward()
        print("second independent backward: PASS")
        _describe_grad(x.grad)
        if x.grad is None:
            print("autograd accumulation: FAIL (x.grad is None)")
            return
        torch.testing.assert_close(
            x.grad.cpu(), torch.full_like(torch.tensor([1.0, 2.0, 3.0]), 4.0),
            rtol=0, atol=0,
        )
        print("gradient accumulation: PASS [4.0, 4.0, 4.0]")
        print("autograd bootstrap: PASS")
    except Exception as error:
        print("backward/gradient validation: FAIL")
        print("  exception_type:", type(error).__name__)
        print("  exception_str:", str(error))
        print("  exception_repr:", repr(error))
        print("  full_traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
