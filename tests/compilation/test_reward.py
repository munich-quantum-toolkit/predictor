# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for different reward functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from mqt.bench import BenchmarkLevel, get_benchmark
from mqt.bench.targets import get_device
from qiskit import QuantumCircuit, transpile

if TYPE_CHECKING:
    from qiskit.transpiler import Target


from qiskit.circuit.library import CXGate, Measure, XGate
from qiskit.transpiler import InstructionProperties, Target

from mqt.predictor import reward
from mqt.predictor.reward import esp_data_available


@pytest.fixture
def device() -> Target:
    """Return the ibm_falcon_27 device."""
    return get_device("ibm_falcon_27")


@pytest.fixture
def compiled_qc(device: Target) -> QuantumCircuit:
    """Return a compiled quantum circuit."""
    qc = get_benchmark("ghz", BenchmarkLevel.ALG, 3)
    return transpile(qc, target=device)


def test_rewards_functions(compiled_qc: QuantumCircuit, device: Target) -> None:
    """Test all reward function."""
    reward_expected_fidelity = reward.expected_fidelity(compiled_qc, device)
    assert 0 <= reward_expected_fidelity <= 1
    reward_critical_depth = reward.crit_depth(compiled_qc)
    assert 0 <= reward_critical_depth <= 1
    assert reward.esp_data_available(device)
    reward_estimated_success_probability = reward.estimated_success_probability(compiled_qc, device)
    assert 0 <= reward_estimated_success_probability <= 1


def make_target(
    *,
    t1: float = 1e-6,
    t2: float = 1e-6,
    meas_err: float = 0.01,
    meas_dur: float = 1e-6,
    x_err: float = 0.01,
    x_dur: float = 1e-6,
    cx_err: float = 0.01,
    cx_dur: float = 1e-6,
    no_qubit_props: bool = False,
) -> Target:
    """Create a Qiskit Target object with customizable error/duration parameters."""
    t = Target(num_qubits=2)

    t.add_instruction(
        Measure(),
        {
            (0,): InstructionProperties(error=meas_err, duration=meas_dur),
            (1,): InstructionProperties(error=meas_err, duration=meas_dur),
        },
    )
    t.add_instruction(
        XGate(),
        {
            (0,): InstructionProperties(error=x_err, duration=x_dur),
            (1,): InstructionProperties(error=x_err, duration=x_dur),
        },
    )
    t.add_instruction(
        CXGate(),
        {
            (0, 1): InstructionProperties(error=cx_err, duration=cx_dur),
            (1, 0): InstructionProperties(error=cx_err, duration=cx_dur),
        },
    )

    if no_qubit_props:
        t.qubit_properties = None
    else:
        t.qubit_properties = [
            type("QubitProps", (), {"t1": t1, "t2": t2}),
            type("QubitProps", (), {"t1": t1, "t2": t2}),
        ]
    return t


@pytest.mark.parametrize(
    "kwargs",
    [
        {"t1": -1.0},
        {"t2": -1.0},
        {"meas_err": -0.1},
        {"meas_err": 1.1},
        {"meas_dur": -1.0},
        {"x_err": -0.1},
        {"x_err": 1.1},
        {"x_dur": -1.0},
        {"cx_err": -0.1},
        {"cx_err": 1.1},
        {"cx_dur": -1.0},
        {"no_qubit_props": True},
    ],
)
def test_esp_data_available_invalid_kwargs(kwargs: dict[str, float | bool]) -> None:
    """Test that `esp_data_available` returns False for invalid device configurations."""
    target = make_target(**kwargs)
    assert not esp_data_available(target)
