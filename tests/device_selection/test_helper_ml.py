# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the helper functions in the ml module."""

from __future__ import annotations

from qiskit import QuantumCircuit

from mqt.predictor.ml.helper import (
    create_feature_vector,
    get_openqasm_gates,
    get_path_training_circuits,
    get_path_training_circuits_compiled,
    get_path_training_data,
)


def test_create_feature_vector() -> None:
    """Keep the input width and gate positions used by trained device-selection models."""
    qc = QuantumCircuit(2, 1)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(1)
    qc.r(0.1, 0.2, 1)
    qc.measure(1, 0)

    feature_vector = create_feature_vector(qc)

    gate_counts = [0.0] * 42
    gates = get_openqasm_gates()
    assert (gates[3], gates[11]) == ("cx", "h")
    gate_counts[3], gate_counts[11] = 1.0, 2.0
    assert len(feature_vector) == 49
    assert feature_vector[:42] == gate_counts
    assert feature_vector[42:44] == [qc.num_qubits, qc.depth()]


def test_get_path_training_circuits() -> None:
    """Test the retrieval of the path to the training circuits."""
    path = get_path_training_circuits()
    assert path.exists()


def test_get_path_training_circuits_compiled() -> None:
    """Test the retrieval of the path to the compiled training circuits."""
    path = get_path_training_circuits_compiled()
    assert path.exists()


def test_get_path_training_data() -> None:
    """Test the retrieval of the path to the training data."""
    path = get_path_training_data()
    assert path.exists()
