# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Boundary conditions for KIT's optimization-only Qiskit baseline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.transpiler import PassManager
from qiskit.transpiler.optimization_metric import OptimizationMetric
from qiskit.transpiler.passes import BasisTranslator, HighLevelSynthesis, UnitarySynthesis

if TYPE_CHECKING:
    from collections.abc import Iterable

    from qiskit import QuantumCircuit


TARGET_BASIS: tuple[str, ...] = (
    "rz",
    "sx",
    "x",
    "cx",
    "id",
    "measure",
    "reset",
    "barrier",
    "delay",
)


def build_translation_pass_manager(target_basis: Iterable[str] = TARGET_BASIS) -> PassManager:
    """Builds the fixed post-optimization translation PassManager.

    This manager synthesizes high-level objects and translates all remaining
    gates into the canonical target basis. It is applied to both baseline
    and optimized circuits to ensure an apples-to-apples comparison.

    Order (matches Qiskit preset common.py: synthesis first, then basis translation):
      1. HighLevelSynthesis   - collapses Clifford / LinearFunction / abstract objects
      2. UnitarySynthesis     - synthesizes UnitaryGate objects using the target basis
      3. BasisTranslator      - translates remaining gates into target_basis

    We use exact synthesis (approximation_degree=1.0 means NO approximation per
    Qiskit docs).

    Args:
        target_basis: An iterable of allowed target hardware operations. Defaults to TARGET_BASIS.

    Returns:
        A Qiskit PassManager configured for exact synthesis and basis translation.
    """
    target_basis_list = list(target_basis)
    return PassManager([
        HighLevelSynthesis(optimization_metric=OptimizationMetric.COUNT_2Q),
        UnitarySynthesis(basis_gates=target_basis_list, approximation_degree=1.0),
        BasisTranslator(SessionEquivalenceLibrary, target_basis=target_basis_list),
    ])


def count_two_qubit_gates(circuit: QuantumCircuit) -> int:
    """Counts the number of 2-qubit gates in a circuit.

    This explicitly ignores directives (like barriers) and non-gate operations
    (like measures). After translation, this should practically just be the CX
    gate count. We keep the logic generic.

    Args:
        circuit: The quantum circuit to analyze.

    Returns:
        The total count of valid 2-qubit gates.
    """
    non_gate_operations_to_be_skipped = {"barrier", "snapshot", "measure", "reset", "delay"}
    two_qubit_gate_count = 0
    for circuit_instruction in circuit.data:
        if circuit_instruction.operation.name in non_gate_operations_to_be_skipped:
            continue
        if len(circuit_instruction.qubits) == 2:
            two_qubit_gate_count += 1
    return two_qubit_gate_count
