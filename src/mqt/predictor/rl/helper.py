# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Helper functions of the reinforcement learning compilation predictor."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from qiskit import QuantumCircuit

from mqt.predictor.utils import calc_supermarq_features, get_openqasm_gates_without_u

if TYPE_CHECKING:
    from numpy.random import Generator
    from numpy.typing import NDArray

import zipfile
from importlib import resources

logger = logging.getLogger("mqt-predictor")


def get_state_sample(max_qubits: int, path_training_circuits: Path, rng: Generator) -> tuple[QuantumCircuit, str]:
    """Returns a random quantum circuit from the training circuits folder.

    Arguments:
        max_qubits: The maximum number of qubits the returned quantum circuit may have. If no limit is set, it defaults to None.
        path_training_circuits: The path to the training circuits folder.
        rng: A random number generator to select a random quantum circuit.

    Returns:
        A tuple containing the random quantum circuit and the path to the file from which it was read.

    Raises:
        RuntimeError: If no quantum circuit could be read from the training circuits folder.
    """
    file_list = list(path_training_circuits.glob("*.qasm"))

    path_zip = path_training_circuits / "training_data_compilation.zip"
    if len(file_list) == 0 and path_zip.exists():
        with zipfile.ZipFile(str(path_zip), "r") as zip_ref:
            zip_ref.extractall(path_training_circuits)

        file_list = list(path_training_circuits.glob("*.qasm"))
        assert len(file_list) > 0

    found_suitable_qc = False
    while not found_suitable_qc:
        random_index = rng.integers(len(file_list))
        num_qubits = int(str(file_list[random_index]).split("_")[-1].split(".")[0])
        if max_qubits and num_qubits > max_qubits:
            continue
        found_suitable_qc = True

    try:
        qc = QuantumCircuit.from_qasm_file(str(file_list[random_index]))
    except Exception:
        raise RuntimeError("Could not read QuantumCircuit from: " + str(file_list[random_index])) from None

    return qc, str(file_list[random_index])


def dict_to_featurevector(gate_dict: dict[str, int]) -> dict[str, float]:
    """Calculates and returns a normalized feature vector of a given quantum circuit gate dictionary."""
    res_dct = dict.fromkeys(get_openqasm_gates_without_u(), 0.0)
    exclude_from_total = {"barrier"}
    total = sum(val for key, val in gate_dict.items() if key not in exclude_from_total)

    for key, val in gate_dict.items():
        if key in res_dct:
            res_dct[key] = val / total if total > 0 else 0.0
    return res_dct


def create_feature_dict(qc: QuantumCircuit) -> dict[str, int | NDArray[np.float64]]:
    """Creates a feature dictionary for a given quantum circuit.

    Arguments:
        qc: The quantum circuit for which the feature dictionary is created.

    Returns:
        The feature dictionary for the given quantum circuit.
    """
    ops_list = qc.count_ops()
    total = sum(val for key, val in ops_list.items() if key != "barrier")
    ops_list_dict = dict_to_featurevector(ops_list)

    feature_dict = {
        **{key: np.array([val], dtype=np.float32) for key, val in ops_list_dict.items()},
        "measure": np.array([ops_list.get("measure", 0) / total if total > 0 else 0.0], dtype=np.float32),
        "num_qubits": qc.num_qubits,
        "depth": qc.depth(),
    }
    supermarq_features = calc_supermarq_features(qc)
    # for all dict values, put them in a list each
    feature_dict["program_communication"] = np.array([supermarq_features.program_communication], dtype=np.float32)
    feature_dict["critical_depth"] = np.array([supermarq_features.critical_depth], dtype=np.float32)
    feature_dict["entanglement_ratio"] = np.array([supermarq_features.entanglement_ratio], dtype=np.float32)
    feature_dict["parallelism"] = np.array([supermarq_features.parallelism], dtype=np.float32)
    feature_dict["liveness"] = np.array([supermarq_features.liveness], dtype=np.float32)
    return feature_dict


def get_path_training_data() -> Path:
    """Returns the path to the training data folder used for RL training."""
    return Path(str(resources.files("mqt.predictor"))) / "rl" / "training_data"


def get_path_trained_model() -> Path:
    """Returns the path to the trained model folder used for RL training."""
    return get_path_training_data() / "trained_model"


def get_path_training_circuits() -> Path:
    """Returns the path to the training circuits folder used for RL training."""
    return get_path_training_data() / "training_circuits"
