# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the helper functions in the ml module."""

from __future__ import annotations

from mqt.bench import BenchmarkLevel, get_benchmark

from mqt.predictor.ml.helper import (
    create_feature_vector,
    get_openqasm_gates,
    get_path_training_circuits,
    get_path_training_circuits_compiled,
    get_path_training_data,
)


def test_create_feature_vector() -> None:
    """Test the creation of a feature dictionary."""
    qc = get_benchmark("dj", BenchmarkLevel.ALG, 3)
    feature_vector = create_feature_vector(qc)
    assert feature_vector is not None


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
