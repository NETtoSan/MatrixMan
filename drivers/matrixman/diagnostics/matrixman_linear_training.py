"""Phase 1 probe for normal PyTorch Linear/MSE/SGD training support.

This diagnostic intentionally does not provide a MatrixMan training fallback.
It reports the first boundary where the current training path stops so that
the smallest missing backend piece can be implemented next.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drivers import matrixman
from drivers.matrixman.tensor import MatrixManTensor
from drivers.matrixman.tensor import readback_tensor


def _install_sub_probe() -> None:
    """Log sub operands only for this diagnostic run."""
    from drivers.matrixman.backends.opengl import dispatch

    original = dispatch.handle_torch_dispatch
    if getattr(original, "_linear_training_sub_probe", False):
        return

    def describe(value):
        if isinstance(value, MatrixManTensor):
            return {
                "type": type(value).__name__,
                "device": str(value.device),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "storage": value._owner.layout.kind,
                "texture": int(value._owner.texture),
                "storage_offset": int(value._storage_offset),
                "logical_strides": list(value._logical_strides),
                "requires_grad": bool(value.requires_grad),
            }
        if isinstance(value, torch.Tensor):
            return {
                "type": type(value).__name__,
                "device": str(value.device),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "storage": "ordinary_torch_tensor",
                "requires_grad": bool(value.requires_grad),
            }
        return repr(value)

    def probed(cls, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.sub.Tensor:
            print("aten.sub.Tensor operands:")
            print("  left:", describe(args[0]))
            print("  right:", describe(args[1]))
        return original(cls, func, types, args, kwargs)

    probed._linear_training_sub_probe = True
    dispatch.handle_torch_dispatch = probed


def _parameter_report(model: nn.Module, label: str) -> None:
    print(label)
    for name, parameter in model.named_parameters():
        print(
            f"  {name}: type={type(parameter).__name__} device={parameter.device} "
            f"shape={list(parameter.shape)} dtype={parameter.dtype} "
            f"requires_grad={parameter.requires_grad}"
        )


def _report_failure(stage: str, error: BaseException) -> None:
    print(f"{stage}: FAIL")
    print(f"  exception_type={type(error).__name__}")
    print(f"  exception_str={str(error)}")
    print(f"  exception_repr={error!r}")
    traceback.print_exc()


def _initialise_linear(model: nn.Linear) -> None:
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.2, -0.3, 0.5], [0.7, 0.1, -0.4]]))
        model.bias.copy_(torch.tensor([0.05, -0.15]))


def _compare(label: str, actual, expected, rtol=2e-4, atol=2e-4) -> bool:
    # Read MatrixMan values through the explicit diagnostic readback API.  In
    # particular, do not use the backend's detach dispatch as a readback
    # mechanism: this diagnostic must leave the original autograd values
    # untouched for a later backward call.
    actual_cpu = readback_tensor(actual, audit_reason=f"Phase 1 comparison: {label}") if isinstance(actual, MatrixManTensor) else actual.detach()
    expected_cpu = expected.detach().cpu() if isinstance(expected, torch.Tensor) else expected
    difference = (actual_cpu - expected_cpu).abs()
    max_abs = float(difference.max().item()) if difference.numel() else 0.0
    mean_abs = float(difference.mean().item()) if difference.numel() else 0.0
    close = bool(torch.allclose(actual_cpu, expected_cpu, rtol=rtol, atol=atol))
    print(
        f"{label}: max_abs={max_abs:.9g} mean_abs={mean_abs:.9g} "
        f"allclose={close} rtol={rtol:g} atol={atol:g}"
    )
    return close


def _graph_report(label: str, value) -> None:
    print(
        f"{label}: type={type(value).__name__} device={value.device} "
        f"requires_grad={value.requires_grad} grad_fn={value.grad_fn!r} "
        f"is_leaf={value.is_leaf}"
    )


def main() -> int:
    torch.manual_seed(1234)
    matrixman.config.backend = "opengl"

    cpu_model = nn.Linear(3, 2, bias=True)
    _initialise_linear(cpu_model)
    model = nn.Linear(3, 2, bias=True)
    _initialise_linear(model)
    cpu_model.train()
    model.train()
    _parameter_report(model, "initial parameters")

    print("Phase 1: Linear(3,2) + MSE + SGD")
    print("requested API: matrixman.prepare(model, training=True)")
    _install_sub_probe()

    try:
        matrixman.init()
    except BaseException as error:
        _report_failure("OpenGL initialization", error)
        return 1

    try:
        prepared = matrixman.prepare(model, backend="opengl", training=True)
    except BaseException as error:
        _report_failure("training preparation", error)
        print("first blocker: training-aware matrixman.prepare is unavailable")
        print("forward Linear with training parameters: NOT REACHED")
        print("weight.grad/bias.grad: NOT REACHED")
        print("torch.optim.SGD update: NOT REACHED")
        matrixman.shutdown()
        return 0

    try:
        print("training preparation: PASS")
        _parameter_report(prepared, "prepared parameters")
        x_cpu = torch.tensor([[1.0, -2.0, 0.5], [-0.5, 0.25, 2.0]], dtype=torch.float32)
        target_cpu = torch.tensor([[0.3, -0.7], [1.1, 0.2]], dtype=torch.float32)
        x = matrixman.to_device(x_cpu)
        target = matrixman.to_device(target_cpu)
        x.requires_grad_(True)
        cpu_x = x_cpu.clone().requires_grad_(True)
        cpu_target = target_cpu.clone()
        cpu_optimizer = torch.optim.SGD(cpu_model.parameters(), lr=0.1)
        optimizer = torch.optim.SGD(prepared.parameters(), lr=0.1)
        cpu_optimizer.zero_grad()
        optimizer.zero_grad()
        cpu_output = cpu_model(cpu_x)
        cpu_loss = ((cpu_output - cpu_target) * (cpu_output - cpu_target)).mean()
        mm_output = prepared(x)
        mm_loss = ((mm_output - target) * (mm_output - target)).mean()
        print("forward Linear: PASS", list(mm_output.shape))
        print("loss: PASS", list(mm_loss.shape))
        _graph_report("mm_output after construction", mm_output)
        _graph_report("mm_loss after construction", mm_loss)
        comparisons_ok = _compare("forward output", mm_output, cpu_output)
        comparisons_ok &= _compare("scalar MSE loss", mm_loss, cpu_loss)
        _graph_report("mm_output before backward", mm_output)
        _graph_report("mm_loss before backward", mm_loss)
        mm_loss.backward()
        cpu_loss.backward()
        print("backward: PASS")
        print("x.grad:", x.grad)
        print("weight.grad:", prepared.weight.grad)
        print("bias.grad:", prepared.bias.grad)
        comparisons_ok &= _compare("x.grad", x.grad, cpu_x.grad)
        comparisons_ok &= _compare("weight.grad", prepared.weight.grad, cpu_model.weight.grad)
        comparisons_ok &= _compare("bias.grad", prepared.bias.grad, cpu_model.bias.grad)
        registered = list(prepared.parameters())
        optimizer_identity_ok = registered[0] is prepared.weight and registered[1] is prepared.bias
        print(f"optimizer parameter identity: {optimizer.param_groups[0]['params'][0] is prepared.weight and optimizer.param_groups[0]['params'][1] is prepared.bias}")
        optimizer.step()
        cpu_optimizer.step()
        print("optimizer.step: PASS")
        _parameter_report(prepared, "parameters after SGD")
        comparisons_ok &= _compare("weight after SGD", prepared.weight, cpu_model.weight)
        comparisons_ok &= _compare("bias after SGD", prepared.bias, cpu_model.bias)
        second_cpu_output = cpu_model(x_cpu)
        second_output = prepared(x)
        comparisons_ok &= _compare("second forward after SGD", second_output, second_cpu_output)
        print(f"registered parameter identity: {optimizer_identity_ok}")
        print(f"model.weight is MatrixManTensor: {isinstance(prepared.weight, MatrixManTensor)}")
        print(f"model.bias is MatrixManTensor: {isinstance(prepared.bias, MatrixManTensor)}")
        if comparisons_ok and optimizer_identity_ok:
            print("Phase 1 Linear/MSE/SGD: PASS")
            return 0
        print("Phase 1 Linear/MSE/SGD: FAIL (numerical or identity mismatch)")
        return 1
    except BaseException as error:
        _report_failure("Linear/MSE/SGD execution", error)
        return 1
    finally:
        matrixman.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
