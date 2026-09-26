"""Phase 3 probe: the MNIST MLP shape with MSE and normal SGD."""

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


def _make_model() -> nn.Sequential:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def _cpu_value(value, label):
    if isinstance(value, MatrixManTensor):
        result = readback_tensor(value, audit_reason=f"Phase 3 comparison: {label}")
    elif isinstance(value, torch.Tensor):
        result = value.detach().cpu()
    else:
        result = torch.as_tensor(value)
    assert isinstance(result, torch.Tensor)
    assert result.device.type == "cpu"
    assert not isinstance(result, MatrixManTensor)
    return result


def _compare(label, actual, expected, rtol=2e-3, atol=2e-3) -> bool:
    actual_cpu = _cpu_value(actual, f"{label} actual")
    expected_cpu = _cpu_value(expected, f"{label} expected")
    assert actual_cpu.device.type == "cpu"
    assert expected_cpu.device.type == "cpu"
    assert not isinstance(actual_cpu, MatrixManTensor)
    assert not isinstance(expected_cpu, MatrixManTensor)
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


def _tensor_storage(value):
    if not isinstance(value, MatrixManTensor):
        return f"type={type(value).__name__} device={value.device} shape={list(value.shape)}"
    return (
        f"type={type(value).__name__} texture={int(value._owner.texture)} "
        f"storage={value._owner.layout.kind} offset={int(value._storage_offset)} "
        f"shape={list(value.shape)} strides={list(value._logical_strides)} "
        f"requires_grad={value.requires_grad}"
    )


def _parameter_storage_report(model: nn.Module, label: str) -> None:
    print(label)
    for name, parameter in model.named_parameters():
        values = readback_tensor(parameter, audit_reason=f"Phase 3 parameter report: {name}")
        print(
            f"  {name}: id={id(parameter)} {_tensor_storage(parameter)} "
            f"sum={float(values.sum()):.9g} min={float(values.min()):.9g} max={float(values.max()):.9g}"
        )


def _log_linear_inputs(label, activation, module):
    weight = module.weight
    bias = module.bias
    weight_t = weight.t()
    print(f"{label} Linear input: {_tensor_storage(activation)}")
    print(f"{label} Linear weight: {_tensor_storage(weight)}")
    print(f"{label} Linear weight.t(): {_tensor_storage(weight_t)}")
    print(f"{label} Linear bias: {_tensor_storage(bias)}")


def _forward(model, x, trace=False):
    flat = model[0](x)
    if trace:
        _log_linear_inputs("first", flat, model[1])
    first = model[1](flat)
    first_relu = model[2](first)
    if trace:
        _log_linear_inputs("second", first_relu, model[3])
    second = model[3](first_relu)
    second_relu = model[4](second)
    if trace:
        _log_linear_inputs("third", second_relu, model[5])
    logits = model[5](second_relu)
    return flat, first, first_relu, second, second_relu, logits


def _loss(logits, target):
    error = logits - target
    return (error * error).mean()


def _parameter_report(model: nn.Module, label: str) -> None:
    print(label)
    for name, parameter in model.named_parameters():
        print(
            f"  {name}: type={type(parameter).__name__} device={parameter.device} "
            f"shape={list(parameter.shape)} requires_grad={parameter.requires_grad}"
        )


def main() -> int:
    matrixman.config.backend = "opengl"
    torch.manual_seed(20260926)
    cpu_model = _make_model()
    state = {name: value.detach().clone() for name, value in cpu_model.state_dict().items()}
    mm_model = _make_model()
    mm_model.load_state_dict(state)

    x_cpu = torch.linspace(-1.0, 1.0, 2 * 1 * 28 * 28, dtype=torch.float32).reshape(2, 1, 28, 28)
    target_cpu = torch.linspace(-0.5, 0.5, 20, dtype=torch.float32).reshape(2, 10)
    learning_rate = 0.05

    print("Phase 3: Flatten -> 784 -> 256 -> 128 -> 10 + MSE + SGD")
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
        expected_names = ["1.weight", "1.bias", "3.weight", "3.bias", "5.weight", "5.bias"]
        registration_ok = names == expected_names and len(registered) == 6
        registration_ok &= all(isinstance(parameter, MatrixManTensor) for parameter in registered)
        print(f"all six Linear parameters registered: {registration_ok}")

        x = matrixman.to_device(x_cpu)
        target = matrixman.to_device(target_cpu)
        x.requires_grad_(True)
        cpu_x = x_cpu.clone().requires_grad_(True)
        cpu_optimizer = torch.optim.SGD(cpu_model.parameters(), lr=learning_rate)
        optimizer = torch.optim.SGD(prepared.parameters(), lr=learning_rate)
        cpu_optimizer.zero_grad()
        optimizer.zero_grad()

        cpu_values = _forward(cpu_model, cpu_x)
        cpu_loss = _loss(cpu_values[-1], target_cpu)
        mm_values = _forward(prepared, x)
        mm_loss = _loss(mm_values[-1], target)

        print("forward: PASS")
        comparisons_ok = registration_ok
        labels = (
            "flattened activation", "first Linear output", "first ReLU output",
            "second Linear output", "second ReLU output", "final logits",
        )
        for label, actual, expected in zip(labels, mm_values, cpu_values):
            comparisons_ok &= _compare(label, actual, expected)
        comparisons_ok &= _compare("scalar MSE loss", mm_loss, cpu_loss)

        try:
            mm_loss.backward()
        except BaseException as error:
            _report_failure("backward", error)
            print("first Phase 3 backward blocker reported above")
            return 0
        cpu_loss.backward()
        print("backward: PASS")
        comparisons_ok &= _compare("input gradient", x.grad, cpu_x.grad)
        cpu_parameters = dict(cpu_model.named_parameters())
        mm_parameters = dict(prepared.named_parameters())
        for name in expected_names:
            comparisons_ok &= _compare(f"{name}.grad", mm_parameters[name].grad, cpu_parameters[name].grad)

        optimizer_identity_ok = all(
            left is right
            for left, right in zip(optimizer.param_groups[0]["params"], registered)
        )
        print(f"optimizer parameter identity: {optimizer_identity_ok}")
        _parameter_storage_report(prepared, "parameters before SGD")
        optimizer.step()
        cpu_optimizer.step()
        print("optimizer.step: PASS")
        _parameter_storage_report(prepared, "parameters immediately after SGD")
        for name in expected_names:
            comparisons_ok &= _compare(f"{name} after SGD", mm_parameters[name], cpu_parameters[name])
        print("updated weight transpose controls")
        for name in ("1.weight", "3.weight", "5.weight"):
            parameter = mm_parameters[name]
            transposed = parameter.t()
            transposed_cpu = readback_tensor(transposed, audit_reason=f"Phase 3 transpose control: {name}")
            expected_transposed = cpu_parameters[name].t()
            print(f"  {name}: {_tensor_storage(parameter)}")
            print(f"  {name}.t(): {_tensor_storage(transposed)}")
            comparisons_ok &= _compare(f"{name}.t() after SGD", transposed, expected_transposed)
            _ = transposed_cpu

        _parameter_storage_report(prepared, "parameters before second forward")
        second_mm_values = _forward(prepared, x, trace=True)
        second_cpu = cpu_model(cpu_x)[-1]
        second_cpu_values = _forward(cpu_model, cpu_x)
        print("second-forward intermediates")
        second_labels = (
            "second flattened activation", "second first Linear output", "second first ReLU output",
            "second second Linear output", "second second ReLU output", "second final Linear output",
        )
        for label, actual, expected in zip(second_labels, second_mm_values, second_cpu_values):
            comparisons_ok &= _compare(label, actual, expected)

        second_mm_repeat = _forward(prepared, x, trace=False)
        print("repeated second-forward stability")
        for label, actual, expected in zip(second_labels, second_mm_repeat, second_mm_values):
            comparisons_ok &= _compare(f"{label} repeat-vs-first", actual, expected, rtol=0, atol=0)

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            cpu_optimizer.zero_grad()
            mm_step_loss = _loss(_forward(prepared, x)[-1], target)
            cpu_step_loss = _loss(_forward(cpu_model, cpu_x)[-1], target_cpu)
            losses.append(float(readback_tensor(mm_step_loss, audit_reason="Phase 3 loss trend").item()))
            mm_step_loss.backward()
            cpu_step_loss.backward()
            optimizer.step()
            cpu_optimizer.step()
        print(f"five-step MatrixMan loss trend: {losses}")
        loss_decreased = losses[-1] < losses[0]
        print(f"five-step loss decreased: {loss_decreased}")
        if comparisons_ok and optimizer_identity_ok and loss_decreased:
            print("Phase 3 MNIST-shape MLP/MSE/SGD: PASS")
            return 0
        print("Phase 3 MNIST-shape MLP/MSE/SGD: FAIL")
        return 1
    except BaseException as error:
        _report_failure("Phase 3 execution", error)
        return 1
    finally:
        matrixman.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
