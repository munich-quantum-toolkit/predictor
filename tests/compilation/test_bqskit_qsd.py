# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the BQSKit quantum Shannon decomposition action."""

from __future__ import annotations

import asyncio

import pytest
from bqskit.compiler.passdata import PassData
from bqskit.ir.gates import VariableUnitaryGate
from qiskit import QuantumCircuit

from mqt.predictor.rl.actions import bqskit_actions


@pytest.mark.filterwarnings("ignore:__array__ implementation doesn't accept a copy keyword:DeprecationWarning")
def test_qsd_unitary_synthesis_pass_applies_one_qsd_level() -> None:
    """The QSD adapter performs one equivalent decomposition of a reachable partition target."""
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.ry(0.21, 2)
    circuit.cx(1, 2)
    bqskit_circuit = bqskit_actions.qiskit_to_bqskit(circuit)
    unitary = bqskit_circuit.get_unitary()
    synthesis_pass = bqskit_actions._QSDUnitarySynthesisPass()  # ruff: ignore[private-member-access]

    decomposed = asyncio.run(synthesis_pass.synthesize(unitary, PassData(bqskit_circuit)))

    assert decomposed.get_unitary().get_distance_from(unitary) < 1e-7
    assert sum(count for gate, count in decomposed.gate_counts.items() if isinstance(gate, VariableUnitaryGate)) == 4
    assert all(gate.num_qudits == 2 for gate in decomposed.gate_set if isinstance(gate, VariableUnitaryGate))


@pytest.mark.filterwarnings("ignore:__array__ implementation doesn't accept a copy keyword:DeprecationWarning")
def test_qsd_unitary_synthesis_pass_preserves_single_qubit_target() -> None:
    """A single-qubit partition retains its unitary without QSD decomposition."""
    circuit = QuantumCircuit(1)
    circuit.ry(0.37, 0)
    bqskit_circuit = bqskit_actions.qiskit_to_bqskit(circuit)
    unitary = bqskit_circuit.get_unitary()
    synthesis_pass = bqskit_actions._QSDUnitarySynthesisPass()  # ruff: ignore[private-member-access]

    decomposed = asyncio.run(synthesis_pass.synthesize(unitary, PassData(bqskit_circuit)))

    assert decomposed.num_qudits == 1
    assert decomposed.get_unitary().get_distance_from(unitary) < 1e-7
