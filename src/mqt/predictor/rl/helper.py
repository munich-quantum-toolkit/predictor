# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
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

from mqt.predictor.utils import calc_supermarq_features, get_openqasm_gates_for_rl

if TYPE_CHECKING:
    from numpy.random import Generator
    from numpy.typing import NDArray

import zipfile
from importlib import resources

logger = logging.getLogger("mqt-predictor")


def predicted_action_to_index(action: object) -> int:
    """Normalize an SB3 action prediction to a scalar action index.

    SB3 may return either a scalar value or a single-element array depending on
    the environment/policy path. RL compilation uses a non-vectorized single
    environment, so anything other than one predicted action is invalid here.
    """
    action_array = np.asarray(action)
    if action_array.ndim == 0:
        return int(action_array.item())

    flattened_action = action_array.reshape(-1)
    if flattened_action.size != 1:
        msg = f"Expected a scalar action prediction, received shape {action_array.shape}."
        raise ValueError(msg)
    return int(flattened_action[0])


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
    file_list = get_training_circuit_files(path_training_circuits)
    if not file_list:
        msg = f"No training circuits found in '{path_training_circuits}'."
        raise RuntimeError(msg)

    suitable_files = [
        file_path
        for file_path in file_list
        if not max_qubits or get_num_qubits_for_training_circuit(file_path) <= max_qubits
    ]
    if not suitable_files:
        msg = f"No training circuits with at most {max_qubits} qubits found in '{path_training_circuits}'."
        raise RuntimeError(msg)

    random_index = int(rng.integers(len(suitable_files)))
    selected_path = suitable_files[random_index]

    try:
        qc = QuantumCircuit.from_qasm_file(str(selected_path))
    except Exception as e:
        msg = f"Could not read QuantumCircuit from: {selected_path}"
        raise RuntimeError(msg) from e

    return qc, str(selected_path)


def get_num_qubits_for_training_circuit(path: Path) -> int:
    """Return the qubit count for one training circuit.

    The historical RL datasets often end in ``_<num>.qasm``, while newer
    split/test layouts can also use names like ``qft_5_indep.qasm`` where the
    numeric token is not the final filename segment. When no numeric token is
    present at all, fall back to parsing the QASM file.
    """
    qubit_count_from_name = get_num_qubits_from_filename(path)
    if qubit_count_from_name is not None:
        return qubit_count_from_name

    try:
        return int(QuantumCircuit.from_qasm_file(str(path)).num_qubits)
    except Exception as e:
        msg = f"Could not determine the number of qubits for training circuit '{path}'."
        raise RuntimeError(msg) from e


def get_num_qubits_from_filename(path: Path) -> int | None:
    """Extract an encoded qubit count from a training-circuit filename."""
    for part in reversed(path.stem.split("_")):
        if part.isdigit():
            return int(part)
    return None


def count_ops_by_name(qc: QuantumCircuit) -> dict[str, int]:
    """Return count_ops with string keys (gate names)."""
    raw = qc.count_ops()
    out: dict[str, int] = {}
    for k, v in raw.items():
        name = k if isinstance(k, str) else k.name  # Instruction -> gate name
        out[name] = out.get(name, 0) + int(v)
    return out


def dict_to_featurevector(gate_dict: dict[str, int]) -> dict[str, float]:
    """Calculates and returns a normalized feature vector of a given quantum circuit gate dictionary."""
    res_dct = dict.fromkeys(get_openqasm_gates_for_rl(), 0.0)
    exclude_from_total = {"barrier"}
    total = sum(val for key, val in gate_dict.items() if key not in exclude_from_total)

    for key, val in gate_dict.items():
        if key in res_dct:
            res_dct[key] = val / total if total > 0 else 0.0
    return res_dct


def create_feature_dict(qc: QuantumCircuit) -> dict[str, int | NDArray[np.float32]]:
    """Creates a feature dictionary for a given quantum circuit."""
    ops = count_ops_by_name(qc)
    total = sum(v for k, v in ops.items() if k != "barrier")
    ops_list_dict = dict_to_featurevector(ops)

    feature_dict: dict[str, int | NDArray[np.float32]] = {
        **{key: np.array([val], dtype=np.float32) for key, val in ops_list_dict.items()},
        "measure": np.array([ops.get("measure", 0) / total if total > 0 else 0.0], dtype=np.float32),
        "num_qubits": int(qc.num_qubits),
        "depth": int(qc.depth()),
    }

    supermarq_features = calc_supermarq_features(qc)
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


def get_path_training_circuits_train() -> Path:
    """Returns the path to the RL training split directory."""
    return get_path_training_circuits() / "train"


def get_path_training_circuits_test() -> Path:
    """Returns the path to the RL test split directory."""
    return get_path_training_circuits() / "test"


def ensure_training_circuit_directories() -> tuple[Path, Path]:
    """Create the RL train/test split directories if they do not exist."""
    train_path = get_path_training_circuits_train()
    test_path = get_path_training_circuits_test()
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)
    return train_path, test_path


def get_training_circuit_files(path_training_circuits: Path) -> list[Path]:
    """Return training-circuit files for both split and legacy layouts."""
    candidate_directories = [path_training_circuits]
    if path_training_circuits.name == "train":
        candidate_directories.append(path_training_circuits.parent)
    else:
        train_directory = path_training_circuits / "train"
        if train_directory.exists():
            candidate_directories.insert(0, train_directory)

    file_list: list[Path] = []
    for candidate_directory in candidate_directories:
        ensure_legacy_training_data_extracted(candidate_directory)
        file_list.extend(sorted(candidate_directory.glob("*.qasm")))

    deduplicated_files: dict[Path, None] = {}
    for file_path in file_list:
        deduplicated_files[file_path] = None
    return list(deduplicated_files)


def ensure_legacy_training_data_extracted(path_training_circuits: Path) -> None:
    """Extract legacy packaged RL circuits when only the zip archive is present."""
    path_zip = path_training_circuits / "training_data_compilation.zip"
    if any(path_training_circuits.glob("*.qasm")) or not path_zip.exists():
        return

    with zipfile.ZipFile(str(path_zip), "r") as zip_ref:
        zip_ref.extractall(path_training_circuits)
