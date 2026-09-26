"""Phase 6 probe: tiny normal torch.optim.Adam training on MatrixMan."""

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


def _initialise(model):
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.2, -0.3, 0.5], [0.7, 0.1, -0.4]]))
        model.bias.copy_(torch.tensor([0.05, -0.15]))


def _cpu_value(value, reason):
    if isinstance(value, MatrixManTensor):
        return readback_tensor(value, audit_reason=reason)
    return value.detach().cpu()


def _compare(label, actual, expected, rtol=2e-3, atol=2e-3):
    actual_cpu = _cpu_value(actual, f"Adam diagnostic: {label} actual")
    expected_cpu = _cpu_value(expected, f"Adam diagnostic: {label} expected")
    difference = (actual_cpu - expected_cpu).abs()
    max_abs = float(difference.max().item()) if difference.numel() else 0.0
    mean_abs = float(difference.mean().item()) if difference.numel() else 0.0
    close = bool(torch.allclose(actual_cpu, expected_cpu, rtol=rtol, atol=atol))
    print(f"{label}: max_abs={max_abs:.9g} mean_abs={mean_abs:.9g} allclose={close}")
    return close


def _loss(model, x, target):
    prediction = model(x)
    error = prediction - target
    return prediction, (error * error).mean()


def _report_failure(stage, error):
    print(f"{stage}: FAIL")
    print(f"  exception_type={type(error).__name__}")
    print(f"  exception_str={str(error)}")
    print(f"  exception_repr={error!r}")
    traceback.print_exc()


def _report_state(label, optimizer, model):
    print(label)
    for parameter_name, parameter in model.named_parameters():
        state = optimizer.state.get(parameter, {})
        print(
            f"  {parameter_name}: parameter_type={type(parameter).__name__} "
            f"device={parameter.device} texture={getattr(parameter._owner, 'texture', None)} "
            f"state_keys={list(state.keys())}"
        )
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                print(
                    f"    state[{key!r}]: type={type(value).__name__} device={value.device} "
                    f"shape={list(value.shape)} dtype={value.dtype}"
                )
            else:
                print(f"    state[{key!r}]: {value!r}")


def main() -> int:
    torch.manual_seed(20260927)
    cpu_model = nn.Linear(3, 2)
    mm_model = nn.Linear(3, 2)
    _initialise(cpu_model)
    mm_model.load_state_dict({name: value.detach().clone() for name, value in cpu_model.state_dict().items()})
    x_cpu = torch.tensor([[1.0, -2.0, 0.5], [-0.5, 0.25, 2.0]], dtype=torch.float32)
    target_cpu = torch.tensor([[0.3, -0.7], [1.1, 0.2]], dtype=torch.float32)
    learning_rate = 1e-3

    matrixman.config.backend = "opengl"
    initialized = False
    try:
        matrixman.init()
        initialized = True
        prepared = matrixman.prepare(mm_model, backend="opengl", training=True)
        x = matrixman.to_device(x_cpu)
        target = matrixman.to_device(target_cpu)
        x.requires_grad_(True)
        cpu_x = x_cpu.clone().requires_grad_(True)
        cpu_optimizer = torch.optim.Adam(cpu_model.parameters(), lr=learning_rate)
        optimizer = torch.optim.Adam(prepared.parameters(), lr=learning_rate)
        parameter_ids = {name: id(value) for name, value in prepared.named_parameters()}
        print("training preparation: PASS")
        print(f"parameter identities stable initially: {all(id(value) == parameter_ids[name] for name, value in prepared.named_parameters())}")

        all_close = True
        for step in range(4):
            cpu_optimizer.zero_grad()
            optimizer.zero_grad()
            cpu_prediction, cpu_loss = _loss(cpu_model, cpu_x, target_cpu)
            mm_prediction, mm_loss = _loss(prepared, x, target)
            all_close &= _compare(f"step {step} loss", mm_loss, cpu_loss)
            if step == 0:
                all_close &= _compare("step 0 prediction", mm_prediction, cpu_prediction)
            mm_loss.backward()
            cpu_loss.backward()
            if step == 0:
                for name, parameter in prepared.named_parameters():
                    all_close &= _compare(f"step 0 {name}.grad", parameter.grad, dict(cpu_model.named_parameters())[name].grad)
            try:
                optimizer.step()
            except BaseException as error:
                _report_failure(f"Adam optimizer.step step={step}", error)
                _report_state("MatrixMan Adam state at failure", optimizer, prepared)
                print("first Adam blocker reported above")
                return 0
            cpu_optimizer.step()
            for name, parameter in prepared.named_parameters():
                all_close &= _compare(f"step {step} {name} after Adam", parameter, dict(cpu_model.named_parameters())[name])
            if {name: id(value) for name, value in prepared.named_parameters()} != parameter_ids:
                raise RuntimeError("MatrixMan parameter identity changed during Adam")
            _report_state(f"Adam state after step {step}", optimizer, prepared)

        print(f"parameter identity stable: {all(id(value) == parameter_ids[name] for name, value in prepared.named_parameters())}")
        print(f"Phase 6 tiny Adam: {'PASS' if all_close else 'FAIL (numerical mismatch)'}")
        return 0 if all_close else 1
    except BaseException as error:
        _report_failure("Adam diagnostic", error)
        return 1
    finally:
        if initialized:
            matrixman.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
