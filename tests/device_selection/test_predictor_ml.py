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
import sys
from pathlib import Path

import pytest
from mqt.bench import BenchmarkLevel, get_benchmark
from mqt.bench.targets import get_available_device_names, get_device
from qiskit.qasm2 import dump

from mqt.predictor import ml


def test_predictor_initialization_with_all_devices() -> None:
    """Test the predictor initialization for a given figure of merit with all available devices."""
    predictor = ml.Predictor(figure_of_merit="expected_fidelity", devices=None)
    assert len(predictor.devices) > 0


# create fixture to get the predictor
@pytest.fixture
def predictor() -> ml.Predictor:
    """Return the predictor."""
    return ml.Predictor(figure_of_merit="expected_fidelity", devices=[get_device("ibm_falcon_127")])


@pytest.fixture
def source_path() -> Path:
    """Return the source path."""
    return Path("./test_uncompiled_circuits")


@pytest.fixture
def target_path() -> Path:
    """Return the target path."""
    return Path("./test_compiled_circuits")


def test_load_training_data_not_found(predictor: ml.Predictor) -> None:
    """Test the loading of the training data."""
    msg = "Training data not found. Please run the training script first as described in the documentation that can be found at https://mqt.readthedocs.io/projects/predictor/en/latest/Usage.html."
    with pytest.raises(FileNotFoundError, match=re.escape(msg)):
        predictor.load_training_data()


def test_generate_training_data(predictor: ml.Predictor, source_path: Path, target_path: Path) -> None:
    """Test the generation of the training data."""
    if not source_path.exists():
        source_path.mkdir()
    if not target_path.exists():
        target_path.mkdir()

    for i in range(2, 8):
        qc = get_benchmark("ghz", BenchmarkLevel.ALG, i)
        path = source_path / f"qc{i}.qasm"
        with path.open("w", encoding="utf-8") as f:
            dump(qc, f)

    # generate compiled circuits using trained RL model
    if sys.platform == "win32":
        with pytest.warns(RuntimeWarning, match=re.escape("Timeout is not supported on Windows.")):
            predictor.generate_compiled_circuits(
                timeout=600, target_path=target_path, source_path=source_path, num_workers=1
            )
    else:
        predictor.generate_compiled_circuits(
            timeout=600, target_path=target_path, source_path=source_path, num_workers=1
        )


def test_save_training_data(predictor: ml.Predictor, source_path: Path, target_path: Path) -> None:
    """Test the saving of the training data."""
    training_data, names_list, scores_list = predictor.generate_trainingdata_from_qasm_files(
        path_uncompiled_circuits=source_path, path_compiled_circuits=target_path, num_workers=1
    )
    assert len(training_data) > 0
    assert len(names_list) > 0
    assert len(scores_list) > 0

    # save the training data and delete it afterwards
    predictor.save_training_data(training_data, names_list, scores_list)
    for file in [
        "training_data_expected_fidelity.npy",
        "names_list_expected_fidelity.npy",
        "scores_list_expected_fidelity.npy",
    ]:
        path = ml.helper.get_path_training_data() / "training_data_aggregated" / file
        assert path.exists()


def test_train_random_forest_classifier_and_predict(predictor: ml.Predictor, source_path: Path) -> None:
    """Test the training of the random forest classifier."""
    predictor.train_random_forest_classifier(save_classifier=True)
    qc = get_benchmark("ghz", BenchmarkLevel.ALG, 3)
    predicted_dev = ml.predict_device_for_figure_of_merit(qc)
    assert predicted_dev.description in get_available_device_names()

    file = source_path / "qc1.qasm"
    with file.open("w", encoding="utf-8") as f:
        dump(qc, f)

    predicted_dev = ml.predict_device_for_figure_of_merit(
        file,
    )
    assert predicted_dev.description in get_available_device_names()

    with pytest.raises(
        FileNotFoundError, match=re.escape("The ML model is not trained yet. Please train the model before using it.")
    ):
        ml.predict_device_for_figure_of_merit(qc=qc, figure_of_merit="false_input")  # type: ignore[arg-type]


def test_remove_files(source_path: Path, target_path: Path) -> None:
    """Remove files created during testing."""
    if source_path.exists():
        for file in source_path.iterdir():
            if file.suffix == ".qasm":
                file.unlink()
        source_path.rmdir()

    if target_path.exists():
        for file in target_path.iterdir():
            if file.suffix == ".qasm":
                file.unlink()
        target_path.rmdir()

    data_path = ml.helper.get_path_training_data() / "training_data_aggregated"
    if data_path.exists():
        for file in data_path.iterdir():
            if file.suffix == ".npy":
                file.unlink()


def test_predict_device_for_figure_of_merit_no_suitable_device() -> None:
    """Test the prediction of the device for a given figure of merit with a wrong device name."""
    num_qubits = 130
    qc = get_benchmark("ghz", BenchmarkLevel.ALG, num_qubits)
    with pytest.raises(
        ValueError, match=re.escape(f"No suitable device found for the given quantum circuit with {num_qubits} qubits.")
    ):
        ml.predict_device_for_figure_of_merit(qc)
