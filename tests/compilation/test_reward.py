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

from mqt.predictor import reward


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
    reward_estimated_success_probability = reward.estimated_success_probability(compiled_qc, device)
    assert 0 <= reward_estimated_success_probability <= 1
