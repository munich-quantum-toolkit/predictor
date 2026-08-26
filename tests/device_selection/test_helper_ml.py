# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the helper functions in the ml module."""

from __future__ import annotations

import numpy as np
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

    expected_operations = dict.fromkeys(get_openqasm_gates(), 0.0)
    expected_operations.update({"x": 1.0, "h": 5.0, "measure": 2.0})
    expected_features = [*expected_operations.values(), 3.0, 5.0, 0.0, 0.0, 0.0, 1 / 5, 11 / 15]

    np.testing.assert_allclose(feature_vector, expected_features)


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
