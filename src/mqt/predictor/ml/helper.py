# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Helper functions for the machine learning device selection predictor."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

from mqt.predictor.utils import calc_supermarq_features

if TYPE_CHECKING:
    import numpy as np
    from numpy._typing import NDArray
    from qiskit import QuantumCircuit


def get_path_training_data() -> Path:
    """Returns the path to the training data folder."""
    return Path(str(resources.files("mqt.predictor"))) / "ml" / "training_data"


def get_path_results(ghz_results: bool = False) -> Path:
    """Returns the path to the results file."""
    if ghz_results:
        return get_path_training_data() / "trained_model" / "res_GHZ.csv"
    return get_path_training_data() / "trained_model" / "res.csv"


def get_path_trained_model(figure_of_merit: str) -> Path:
    """Returns the path to the trained model folder resulting from the machine learning training."""
    return get_path_training_data() / "trained_model" / ("trained_clf_" + figure_of_merit + ".joblib")


def get_path_training_circuits() -> Path:
    """Returns the path to the training circuits folder."""
    return get_path_training_data() / "training_circuits"


def get_path_training_circuits_compiled() -> Path:
    """Returns the path to the compiled training circuits folder."""
    return get_path_training_data() / "training_circuits_compiled"


def get_openqasm_gates() -> list[str]:
    """Returns a list of all quantum gates within the openQASM 2.0 standard header."""
    # according to https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/qasm/libs/qelib1.inc
    return [
        "u3",
        "u2",
        "u1",
        "cx",
        "id",
        "u0",
        "u",
        "p",
        "x",
        "y",
        "z",
        "h",
        "s",
        "sdg",
        "t",
        "tdg",
        "rx",
        "ry",
        "rz",
        "sx",
        "sxdg",
        "cz",
        "cy",
        "swap",
        "ch",
        "ccx",
        "cswap",
        "crx",
        "cry",
        "crz",
        "cu1",
        "cp",
        "cu3",
        "csx",
        "cu",
        "rxx",
        "rzz",
        "rccx",
        "rc3x",
        "c3x",
        "c3sqrtx",
        "c4x",
    ]


def dict_to_featurevector(gate_dict: dict[str, int]) -> dict[str, int]:
    """Calculates and returns the feature vector of a given quantum circuit gate dictionary."""
    res_dct = dict.fromkeys(get_openqasm_gates(), 0)
    for key, val in dict(gate_dict).items():
        if key in res_dct:
            res_dct[key] = val

    return res_dct


def create_feature_vector(qc: QuantumCircuit) -> list[int | float]:
    """Creates and returns a feature dictionary for a given quantum circuit.

    Arguments:
        qc: The quantum circuit to be compiled.

    Returns:
        The feature dictionary of the given quantum circuit.
    """
    ops_list = qc.count_ops()
    ops_list_dict = dict_to_featurevector(ops_list)

    feature_dict = {}
    for key in ops_list_dict:
        feature_dict[key] = float(ops_list_dict[key])

    feature_dict["num_qubits"] = float(qc.num_qubits)
    feature_dict["depth"] = float(qc.depth())

    supermarq_features = calc_supermarq_features(qc)
    feature_dict["program_communication"] = supermarq_features.program_communication
    feature_dict["critical_depth"] = supermarq_features.critical_depth
    feature_dict["entanglement_ratio"] = supermarq_features.entanglement_ratio
    feature_dict["parallelism"] = supermarq_features.parallelism
    feature_dict["liveness"] = supermarq_features.liveness
    return list(feature_dict.values())


@dataclass
class TrainingData:
    """Dataclass for the training data."""

    X_train: NDArray[np.float64]
    y_train: NDArray[np.float64]
    X_test: NDArray[np.float64] | None = None
    y_test: NDArray[np.float64] | None = None
    indices_train: list[int] | None = None
    indices_test: list[int] | None = None
    names_list: list[str] | None = None
    scores_list: list[list[float]] | None = None
