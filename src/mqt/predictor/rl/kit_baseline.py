# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Boundary conditions for KIT's optimization-only Qiskit baseline."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import TYPE_CHECKING

from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.passmanager import ConditionalController
from qiskit.transpiler import PassManager
from qiskit.transpiler.optimization_metric import OptimizationMetric
from qiskit.transpiler.passes import (
    BasisTranslator,
    GatesInBasis,
    HighLevelSynthesis,
    UnitarySynthesis,
)

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

QISKIT_BASELINE_O0_COLUMN = "baseline_cx_O0"


def load_qiskit_baseline_lookup(
    path: Path | str,
    baseline_column: str = QISKIT_BASELINE_O0_COLUMN,
) -> dict[str, float]:
    """Load precomputed KIT Qiskit baseline counts from a CSV file.

    Args:
        path: Path to KIT's ``qiskit_baselines.csv`` file.
        baseline_column: Baseline count column to read. Defaults to Qiskit's
            optimization level 0 column.

    Returns:
        A mapping from circuit stem to the precomputed baseline count.

    Raises:
        ValueError: If the CSV file does not contain the required columns.
    """
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        required_columns = {"circuit_id", baseline_column}
        missing_columns = required_columns - fieldnames
        if missing_columns:
            msg = f"{csv_path} is missing required columns: {sorted(missing_columns)}."
            raise ValueError(msg)

        baseline_lookup: dict[str, float] = {}
        for row in reader:
            if "status" in fieldnames and row.get("status") != "ok":
                continue

            circuit_id = row.get("circuit_id", "")
            baseline_value = row.get(baseline_column, "")
            if not circuit_id or not baseline_value:
                continue

            baseline_cx = float(baseline_value)
            if math.isfinite(baseline_cx):
                baseline_lookup[Path(circuit_id).stem] = baseline_cx

    return baseline_lookup


def build_translation_pass_manager(target_basis: Iterable[str] = TARGET_BASIS) -> PassManager:
    """Builds the fixed post-optimization translation PassManager.

    This manager synthesizes high-level objects and translates all remaining
    gates into the canonical target basis. It is applied to both baseline
    and optimized circuits to ensure an apples-to-apples comparison.

    Order (matches Qiskit preset common.py: synthesis first, then basis translation):
      1. HighLevelSynthesis   - collapses Clifford / LinearFunction / abstract objects
      2. UnitarySynthesis     - synthesizes UnitaryGate objects using the target basis
      3. BasisTranslator      - translates remaining gates into target_basis

    The synthesis and translation tasks are guarded by ``GatesInBasis`` and skipped
    when the circuit is already in the target basis.

    We use exact synthesis (approximation_degree=1.0 means NO approximation per
    Qiskit docs).

    Args:
        target_basis: An iterable of allowed target hardware operations. Defaults to TARGET_BASIS.

    Returns:
        A Qiskit PassManager configured for exact synthesis and basis translation.
    """
    target_basis_list = list(target_basis)
    translation_tasks = [
        HighLevelSynthesis(optimization_metric=OptimizationMetric.COUNT_2Q),
        UnitarySynthesis(basis_gates=target_basis_list, approximation_degree=1.0),
        BasisTranslator(SessionEquivalenceLibrary, target_basis=target_basis_list),
    ]

    def needs_translation(property_set: dict[str, bool]) -> bool:
        return not property_set["all_gates_in_basis"]

    translation_pass_manager = PassManager()
    translation_pass_manager.append(GatesInBasis(basis_gates=target_basis_list))
    translation_pass_manager.append(ConditionalController(translation_tasks, condition=needs_translation))
    return translation_pass_manager


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
