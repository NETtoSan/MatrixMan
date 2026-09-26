"""Phase 2 probe: two Linear layers, ReLU, MSE, and normal SGD."""

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
from drivers.matrixman.tensor import MatrixManTensor, readback_tensor


def _initialise(model: nn.Sequential) -> None:
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([
            [0.20, -0.10, 0.30],
            [-0.40, 0.50, 0.10],
            [0.70, 0.20, -0.30],
            [0.05, -0.25, 0.45],
        ]))
        model[0].bias.copy_(torch.tensor([0.10, -0.20, 0.05, 0.30]))
        model[2].weight.copy_(torch.tensor([
            [0.30, -0.20, 0.40, 0.10],
            [-0.15, 0.25, -0.35, 0.45],
        ]))
        model[2].bias.copy_(torch.tensor([0.05, -0.10]))


def _compare(label, actual, expected, rtol=2e-4, atol=2e-4) -> bool:
    actual_cpu = (
        readback_tensor(actual, audit_reason=f"Phase 2 comparison: {label}")
        if isinstance(actual, MatrixManTensor) else actual.detach()
    )
    expected_cpu = expected.detach() if isinstance(expected, torch.Tensor) else expected
    difference = (actual_cpu - expected_cpu).abs()
    max_abs = float(difference.max().item()) if difference.numel() else 0.0
    mean_abs = float(difference.mean().item()) if difference.numel() else 0.0
    close = bool(torch.allclose(actual_cpu, expected_cpu, rtol=rtol, atol=atol))
    print(
        f"{label}: max_abs={max_abs:.9g} mean_abs={mean_abs:.9g} "
        f"allclose={close} rtol={rtol:g} atol={atol:g}"
    )
    return close


def _report_failure(stage: str, error: BaseException) -> None:
    print(f"{stage}: FAIL")
    print(f"  exception_type={type(error).__name__}")
    print(f"  exception_str={str(error)}")
    print(f"  exception_repr={error!r}")
    traceback.print_exc()


def _parameter_report(model: nn.Module, label: str) -> None:
    print(label)
    for name, parameter in model.named_parameters():
        print(
            f"  {name}: type={type(parameter).__name__} device={parameter.device} "
            f"shape={list(parameter.shape)} requires_grad={parameter.requires_grad}"
        )


def main() -> int:
    matrixman.config.backend = "opengl"
    cpu_model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    mm_model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    _initialise(cpu_model)
    _initialise(mm_model)

    x_cpu = torch.tensor([[1.0, -2.0, 0.5], [-0.5, 0.25, 2.0]], dtype=torch.float32)
    target_cpu = torch.tensor([[0.3, -0.7], [1.1, 0.2]], dtype=torch.float32)
    learning_rate = 0.1

    print("Phase 2: Linear(3,4) -> ReLU -> Linear(4,2) + MSE + SGD")
    try:
        matrixman.init()
    except BaseException as error:
        _report_failure("OpenGL initialization", error)
        return 1

    try:
        prepared = matrixman.prepare(mm_model, backend="opengl", training=True)
        print("training preparation: PASS")
        _parameter_report(prepared, "prepared parameters")
        names = [name for name, _ in prepared.named_parameters()]
        registered = list(prepared.parameters())
        registration_ok = names == ["0.weight", "0.bias", "2.weight", "2.bias"]
        registration_ok &= all(isinstance(parameter, MatrixManTensor) for parameter in registered)
        print(f"all four Linear parameters registered: {registration_ok}")

        x = matrixman.to_device(x_cpu)
        target = matrixman.to_device(target_cpu)
        x.requires_grad_(True)
        cpu_x = x_cpu.clone().requires_grad_(True)
        cpu_optimizer = torch.optim.SGD(cpu_model.parameters(), lr=learning_rate)
        optimizer = torch.optim.SGD(prepared.parameters(), lr=learning_rate)
        cpu_optimizer.zero_grad()
        optimizer.zero_grad()

        cpu_first = cpu_model[0](cpu_x)
        cpu_relu = cpu_model[1](cpu_first)
        cpu_final = cpu_model[2](cpu_relu)
        cpu_loss = ((cpu_final - target_cpu) * (cpu_final - target_cpu)).mean()

        mm_first = prepared[0](x)
        mm_relu = prepared[1](mm_first)
        mm_final = prepared[2](mm_relu)
        mm_loss = ((mm_final - target) * (mm_final - target)).mean()

        print("forward: PASS")
        comparisons_ok = registration_ok
        comparisons_ok &= _compare("first Linear output", mm_first, cpu_first)
        comparisons_ok &= _compare("ReLU output", mm_relu, cpu_relu)
        comparisons_ok &= _compare("final Linear output", mm_final, cpu_final)
        comparisons_ok &= _compare("scalar MSE loss", mm_loss, cpu_loss)

        try:
            mm_loss.backward()
        except BaseException as error:
            _report_failure("backward", error)
            print("first Phase 2 backward blocker reported above")
            return 0
        cpu_loss.backward()
        print("backward: PASS")
        comparisons_ok &= _compare("input gradient", x.grad, cpu_x.grad)
        comparisons_ok &= _compare("first Linear weight.grad", prepared[0].weight.grad, cpu_model[0].weight.grad)
        comparisons_ok &= _compare("first Linear bias.grad", prepared[0].bias.grad, cpu_model[0].bias.grad)
        comparisons_ok &= _compare("second Linear weight.grad", prepared[2].weight.grad, cpu_model[2].weight.grad)
        comparisons_ok &= _compare("second Linear bias.grad", prepared[2].bias.grad, cpu_model[2].bias.grad)

        optimizer.step()
        cpu_optimizer.step()
        print("optimizer.step: PASS")
        for name in ("0.weight", "0.bias", "2.weight", "2.bias"):
            comparisons_ok &= _compare(
                f"{name} after SGD",
                dict(prepared.named_parameters())[name],
                dict(cpu_model.named_parameters())[name],
            )

        cpu_second = cpu_model(cpu_x)
        mm_second = prepared(x)
        comparisons_ok &= _compare("second forward after SGD", mm_second, cpu_second)
        if comparisons_ok:
            print("Phase 2 two-layer MLP: PASS")
            return 0
        print("Phase 2 two-layer MLP: FAIL (numerical or registration mismatch)")
        return 1
    except BaseException as error:
        _report_failure("Phase 2 execution", error)
        return 1
    finally:
        matrixman.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
