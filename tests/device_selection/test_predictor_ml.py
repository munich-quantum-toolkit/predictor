# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the machine learning device selection predictor module."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from mqt.bench import BenchmarkLevel, get_benchmark
from mqt.bench.targets import get_device
from qiskit.qasm2 import dump

from mqt.predictor.ml import predict_device_for_figure_of_merit, setup_device_predictor
from mqt.predictor.ml.helper import get_path_training_data
from mqt.predictor.ml.predictor import Predictor


@pytest.fixture
def path_uncompiled_circuits() -> Path:
    """Return the source path."""
    return Path("./test_uncompiled_circuits")


@pytest.fixture
def path_compiled_circuits() -> Path:
    """Return the target path."""
    return Path("./test_compiled_circuits")


@pytest.mark.parametrize("gnn,verbose", [(False, False), (True, False), (True, True)], ids=["rf", "gnn", "gnn_verbose"])
def test_setup_device_predictor_with_prediction(
    path_uncompiled_circuits: Path, path_compiled_circuits: Path, gnn: bool, verbose: bool
) -> None:
    """Test the full training pipeline and prediction using a mock device."""
    if not path_uncompiled_circuits.exists():
        path_uncompiled_circuits.mkdir()
    if not path_compiled_circuits.exists():
        path_compiled_circuits.mkdir()

    for i in range(2, 8):
        qc = get_benchmark("ghz", BenchmarkLevel.ALG, i)
        path = path_uncompiled_circuits / f"qc{i}.qasm"
        with path.open("w", encoding="utf-8") as f:
            dump(qc, f)

    device = get_device("ibm_falcon_127")
    success = setup_device_predictor(
        devices=[device],
        figure_of_merit="expected_fidelity",
        path_uncompiled_circuits=path_uncompiled_circuits,
        path_compiled_circuits=path_compiled_circuits,
        gnn=gnn,
    )
    assert success

    data_path = get_path_training_data() / "training_data_aggregated"
    if gnn:
        assert (data_path / "graph_dataset_expected_fidelity.pt").exists()
    else:
        assert (data_path / "training_data_expected_fidelity.npy").exists()
        assert (data_path / "names_list_expected_fidelity.npy").exists()
        assert (data_path / "scores_list_expected_fidelity.npy").exists()

    test_qc = get_benchmark("ghz", BenchmarkLevel.ALG, 3)
    predicted = predict_device_for_figure_of_merit(test_qc, figure_of_merit="expected_fidelity")

    assert predicted.description == "ibm_falcon_127"


def test_remove_files(path_uncompiled_circuits: Path, path_compiled_circuits: Path) -> None:
    """Remove files created during testing."""
    if path_uncompiled_circuits.exists():
        for file in path_uncompiled_circuits.iterdir():
            if file.suffix == ".qasm":
                file.unlink()
        path_uncompiled_circuits.rmdir()

    if path_compiled_circuits.exists():
        for file in path_compiled_circuits.iterdir():
            if file.suffix == ".qasm":
                file.unlink()
        path_compiled_circuits.rmdir()

    data_path = get_path_training_data() / "training_data_aggregated"
    if data_path.exists():
        for file in data_path.iterdir():
            if file.suffix == ".npy" or file.suffix == ".pt":
                file.unlink()


def test_predict_device_for_figure_of_merit_no_suitable_device() -> None:
    """Test the prediction of the device for a given figure of merit with a wrong device name."""
    num_qubits = 130
    qc = get_benchmark("ghz", BenchmarkLevel.ALG, num_qubits)
    with pytest.raises(
        ValueError, match=re.escape(f"No suitable device found for the given quantum circuit with {num_qubits} qubits.")
    ):
        predict_device_for_figure_of_merit(qc)


@pytest.mark.parametrize("gnn", [False, True], ids=["rf", "gnn"])
def test_get_prepared_training_data_false_input(gnn: bool) -> None:
    """Test the retrieval of prepared training data."""
    pred = Predictor(devices=[], figure_of_merit="expected_fidelity", gnn=gnn)
    with pytest.raises(FileNotFoundError, match=re.escape("Training data not found.")):
        pred._get_prepared_training_data()  # noqa: SLF001
