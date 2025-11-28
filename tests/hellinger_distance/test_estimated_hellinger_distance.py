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
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from mqt.bench import BenchmarkLevel, get_benchmark
from mqt.bench.targets import get_available_device_names, get_device
from qiskit import QuantumCircuit
from qiskit.qasm2 import dump
from torch_geometric.data import Batch, Data

from mqt.predictor.hellinger import calc_device_specific_features, hellinger_distance
from mqt.predictor.ml import Predictor as ml_Predictor
from mqt.predictor.ml import predict_device_for_figure_of_merit
from mqt.predictor.ml.helper import TrainingData, create_dag, get_path_training_data
from mqt.predictor.rl import Predictor as rl_Predictor

if TYPE_CHECKING:
    from qiskit.transpiler import Target


@pytest.fixture
def source_path() -> Path:
    """Return the source path."""
    return Path("./test_uncompiled_circuits")


@pytest.fixture
def target_path() -> Path:
    """Return the target path."""
    return Path("./test_compiled_circuits")


@pytest.fixture
def device() -> Path:
    """Return the target device."""
    return get_device("quantinuum_h2_56")


def test_create_device_specific_feature_dict(device: Target) -> None:
    """Test the creation of a device-specific feature vector."""
    qc = QuantumCircuit(device.num_qubits)
    for i in range(1, device.num_qubits):
        qc.cz(0, i)

    feature_vector = calc_device_specific_features(qc, device)
    expected_feat_vec = np.array([
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        1.00000000e00,
        5.50000000e01,
        5.60000000e01,
        1.00000000e00,
        1.00000000e00,
        0.00000000e00,
        3.57142857e-02,
        1.78571429e-02,
        0.00000000e00,
        3.57142857e-02,
    ])

    assert np.allclose(feature_vector, expected_feat_vec)


def test_hellinger_distance() -> None:
    """Test the calculation of the Hellinger distance."""
    p = [0.0, 1.0]
    q = [1.0, 0.0]
    distance = hellinger_distance(p, q)
    assert distance == 1


def test_hellinger_distance_error() -> None:
    """Test error during Hellinger distance calculation."""
    valid = [0.5, 0.5]
    invalid = [0.5, 0.4]

    with pytest.raises(AssertionError, match="q is not a probability distribution"):
        hellinger_distance(p=valid, q=invalid)
    with pytest.raises(AssertionError, match="p is not a probability distribution"):
        hellinger_distance(p=invalid, q=valid)


@pytest.mark.parametrize("gnn", [False, True], ids=["rf", "gnn"])
def test_train_model_and_predict(device: Target, gnn: bool) -> None:
    """Test the training of the random forest regressor. The trained model is saved and used in the following tests."""
    n_circuits = 20

    qc = QuantumCircuit(device.num_qubits)
    for i in range(1, device.num_qubits):
        qc.cz(0, i)

    if not gnn:
        feature_vector = calc_device_specific_features(qc, device)
        feature_vector_list = [feature_vector] * n_circuits
    else:
        x, edge_index, number_of_gates = create_dag(qc)
        training_sample = [(x, edge_index, number_of_gates)] * n_circuits

    rng = np.random.default_rng()
    noisy = rng.random(device.num_qubits)
    noisy /= np.sum(noisy)
    noiseless = np.zeros_like(noisy)
    noiseless[0] = 1.0
    distance_label = hellinger_distance(noisy, noiseless)
    labels_list = [distance_label] * n_circuits
    if not gnn:
        training_data = TrainingData(X_train=feature_vector_list, y_train=labels_list)
    else:
        training_data_list = []
        for i in range(n_circuits):
            x, edge_idx, n_nodes = training_sample[i]
            gnn_training_sample = Data(
                x=x,
                y=torch.tensor(labels_list[i], dtype=torch.float32),
                edge_index=edge_idx,
                num_nodes=n_nodes,
            )
            training_data_list.append(gnn_training_sample)
        training_data = TrainingData(X_train=training_data_list, y_train=labels_list)

    # 3. Model Training
    pred = ml_Predictor(figure_of_merit="hellinger_distance", devices=[device], gnn=gnn)
    if gnn:
        trained_model = pred.train_gnn_model(training_data, num_epochs=100, patience=20)
    else:
        trained_model = pred.train_random_forest_model(training_data)

    if not gnn:
        assert np.isclose(trained_model.predict([feature_vector]), distance_label)
    else:
        trained_model.eval()
        model_device = next(trained_model.parameters()).device
        with torch.no_grad():
            batch = Batch.from_data_list(training_data.X_train).to(model_device)
            out = trained_model(batch)
            out = out.squeeze(-1)
            predicted_values = out.cpu().numpy()
            labels = np.asarray(labels_list, dtype=np.float32)
        # it is set a tolerance value of 2e-1 just because of the small number of training samples
        assert np.allclose(predicted_values, labels, atol=2e-1)


@pytest.mark.parametrize("gnn", [False, True], ids=["rf", "gnn"])
def test_train_and_qcompile_with_hellinger_model(
    source_path: Path, target_path: Path, device: Target, gnn: bool
) -> None:
    """Test the entire predictor toolchain for estimating the Hellinger distance with both RF and GNN."""
    figure_of_merit = "estimated_hellinger_distance"

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=f"The connectivity of the device '{device.description}' is uni-directional and MQT Predictor might return a compiled circuit that assumes bi-directionality.",
        )

        # 1. Train the reinforcement learning model for circuit compilation
        rl_predictor = rl_Predictor(device=device, figure_of_merit=figure_of_merit)

        rl_predictor.train_model(
            timesteps=5,
            test=True,
        )

        # 2. Setup and train the machine learning model for device selection
        ml_predictor = ml_Predictor(devices=[device], figure_of_merit=figure_of_merit, gnn=gnn)

        # Prepare uncompiled circuits
        if not source_path.exists():
            source_path.mkdir()
        if not target_path.exists():
            target_path.mkdir()

        for i in range(2, 5):
            qc = get_benchmark("ghz", BenchmarkLevel.ALG, i)
            path = source_path / f"qc{i}.qasm"
            with path.open("w", encoding="utf-8") as f:
                dump(qc, f)

        # Generate compiled circuits (using trained RL model)
        if sys.platform == "win32" and not gnn:
            with pytest.warns(RuntimeWarning, match=re.escape("Timeout is not supported on Windows.")):
                ml_predictor.compile_training_circuits(
                    timeout=600,
                    path_compiled_circuits=target_path,
                    path_uncompiled_circuits=source_path,
                    num_workers=1,
                )
        else:
            ml_predictor.compile_training_circuits(
                timeout=600, path_compiled_circuits=target_path, path_uncompiled_circuits=source_path, num_workers=1
            )

        # Generate training data from the compiled circuits
        ml_predictor.generate_training_data(
            path_uncompiled_circuits=source_path, path_compiled_circuits=target_path, num_workers=1
        )
        if gnn:
            assert (
                get_path_training_data() / "training_data_aggregated" / "graph_dataset_estimated_hellinger_distance.pt"
            ).exists()
        else:
            for file in [
                "training_data_estimated_hellinger_distance.npy",
                "names_list_estimated_hellinger_distance.npy",
                "scores_list_estimated_hellinger_distance.npy",
            ]:
                path = get_path_training_data() / "training_data_aggregated" / file
                assert path.exists()

        # Train the ML model
        ml_predictor.train_gnn_model() if gnn else ml_predictor.train_random_forest_model()
        qc = get_benchmark("ghz", BenchmarkLevel.ALG, 3)

        # Test the prediction
        predicted_dev = predict_device_for_figure_of_merit(qc, figure_of_merit, gnn=gnn)
        assert predicted_dev.description in get_available_device_names()


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

    data_path = get_path_training_data() / "training_data_aggregated"
    if data_path.exists():
        for file in data_path.iterdir():
            if file.suffix in (".npy", ".pt"):
                file.unlink()


    model_path = get_path_training_data() / "trained_model"
    if model_path.exists():
        for file in model_path.iterdir():
            if file.suffix in (".joblib", ".pth", ".json"):
                file.unlink()


def test_predict_device_for_estimated_hellinger_distance_no_device_provided() -> None:
    """Test the error handling of the device selection predictor when no device is provided for the Hellinger distance model."""
    rng = np.random.default_rng()
    random_int = rng.integers(0, 10)

    feature_vector = rng.random(random_int)
    feature_vector_list = [feature_vector]

    distance_label = rng.random(random_int)
    labels_list = [distance_label]
    training_data = TrainingData(X_train=feature_vector_list, y_train=labels_list)

    pred = ml_Predictor(
        figure_of_merit="hellinger_distance", devices=[get_device("ibm_falcon_27"), get_device("ibm_falcon_127")]
    )
    with pytest.raises(
        ValueError, match=re.escape("A single device must be provided for Hellinger distance model training.")
    ):
        pred.train_random_forest_model(training_data)
