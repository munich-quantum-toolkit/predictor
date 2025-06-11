# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for different reward functions."""

from __future__ import annotations

import pytest
from mqt.bench import get_benchmark
from mqt.bench.devices import Device, get_device_by_name
from qiskit import QuantumCircuit, transpile

from mqt.predictor import reward


@pytest.fixture
def device() -> Device:
    """Return the IonQ Harmony device."""
    return get_device_by_name("ionq_harmony")


@pytest.fixture
def compiled_qc(device: Device) -> QuantumCircuit:
    """Return a compiled quantum circuit."""
    qc = get_benchmark("ghz", 1, 3)
    return transpile(qc, basis_gates=device.basis_gates, coupling_map=device.coupling_map)


def test_rewards_functions(compiled_qc: QuantumCircuit, device: Device) -> None:
    """Test all reward function."""
    reward_expected_fidelity = reward.expected_fidelity(compiled_qc, device)
    assert 0 <= reward_expected_fidelity <= 1
    reward_critical_depth = reward.crit_depth(compiled_qc)
    assert 0 <= reward_critical_depth <= 1
    reward_estimated_success_probability = reward.estimated_success_probability(compiled_qc, device)
    assert 0 <= reward_estimated_success_probability <= 1
