# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the helper functions in the ml module."""

from __future__ import annotations

import torch
from mqt.bench import BenchmarkLevel, get_benchmark
from qiskit import QuantumCircuit

from mqt.predictor.ml.helper import (
    create_dag,
    create_feature_vector,
    get_openqasm_gates,
    get_openqasm3_gates,
    get_path_training_circuits,
    get_path_training_circuits_compiled,
    get_path_training_data,
)


def test_create_feature_vector() -> None:
    """Test the creation of a feature dictionary."""
    qc = get_benchmark("dj", BenchmarkLevel.ALG, 3)
    feature_vector = create_feature_vector(qc)
    assert feature_vector is not None


def test_create_dag() -> None:
    """Test the creation of a DAG."""
    qc = get_benchmark("dj", BenchmarkLevel.INDEP, 3).decompose()
    node_vector, edge_index, number_nodes = create_dag(qc)
    assert isinstance(node_vector, torch.Tensor)
    assert isinstance(edge_index, torch.Tensor)
    assert number_nodes > 0
    assert node_vector.shape[0] == number_nodes
    assert edge_index.dtype == torch.long
    assert edge_index.ndim == 2
    assert edge_index.shape[0] == 2


def test_empty_circuit_dag() -> None:
    """Test the creation of a DAG from an empty quantum circuit."""
    qc = QuantumCircuit(2)

    node_vector, edge_index, number_nodes = create_dag(qc)

    num_gates = len(get_openqasm3_gates()) + 1
    expected_feature_size = num_gates + 12
    # No nodes
    assert number_nodes == 0

    # node_vector empty with shape
    assert isinstance(node_vector, torch.Tensor)
    assert node_vector.ndim == 2
    assert node_vector.shape[0] == 0
    assert node_vector.shape[1] == expected_feature_size

    # edge_index empty (2, 0) and dtype long as in the code
    assert isinstance(edge_index, torch.Tensor)
    assert edge_index.dtype == torch.long
    assert tuple(edge_index.shape) == (2, 0)


def test_get_openqasm_gates() -> None:
    """Test the retrieval of the OpenQASM gates."""
    assert get_openqasm_gates() is not None


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
