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
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager, Target

from mqt.predictor.utils import calc_supermarq_features

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.random import Generator
    from numpy.typing import NDArray

    from mqt.predictor.rl.actions import Action

    # from mqt.predictor.rl.actions import Action

import zipfile
from importlib import resources

logger = logging.getLogger("mqt-predictor")


def best_of_n_passmanager(
    action: Action,
    device: Target,
    qc: QuantumCircuit,
    max_iteration: tuple[int, int] = (20, 20),
    metric_fn: Callable[[QuantumCircuit], float] | None = None,
) -> tuple[QuantumCircuit, dict[str, Any]]:
    """Runs the given transpile_pass multiple times and keeps the best result.

    Args:
        action: The action dictionary with a 'transpile_pass' key
            (lambda device -> [passes]).
        device: The target backend or device.
        qc: The input quantum circuit.
        max_iteration: A tuple (layout_trials, routing_trials) specifying
            how many times to try.
        metric_fn: Optional function to score circuits; defaults to circuit depth.

    Returns:
        A tuple containing the best transpiled circuit and its corresponding
        property set.
    """
    best_val = None
    best_result = None
    best_property_set = None

    if callable(action.transpile_pass):
        try:
            if action.name == "SabreLayout+AIRouting":
                all_passes = action.transpile_pass(device, max_iteration)
            else:
                all_passes = action.transpile_pass(device)
        except TypeError as e:
            msg = f"Error calling transpile_pass for {action.name}: {e}"
            raise ValueError(msg) from e
    else:
        all_passes = action.transpile_pass

    if not isinstance(all_passes, list):
        msg = f"Expected list of passes, got {type(all_passes)}"
        raise TypeError(msg)

    layout_passes = all_passes[:-1]
    routing_pass = all_passes[-1:]

    # Run layout once
    layout_pm = PassManager(layout_passes)
    try:
        layouted_qc = layout_pm.run(qc)
        layout_props = dict(layout_pm.property_set)
    except Exception:
        return qc, {}

    # Run routing multiple times and optimize for the given metric
    for i in range(max_iteration[1]):
        pm = PassManager(routing_pass)
        pm.property_set.update(layout_props)
        try:
            out_circ = pm.run(layouted_qc)
            prop_set = dict(pm.property_set)

            val = metric_fn(out_circ) if metric_fn else out_circ.depth()
            if best_val is None or val < best_val:
                best_val = val
                best_result = out_circ
                best_property_set = prop_set
                if best_val == 0:
                    break
        except Exception as e:
            print(f"[Routing] Trial {i + 1} failed: {e}")
            continue
    if best_result is not None:
        if best_property_set is None:
            best_property_set = {}
        return best_result, best_property_set
    print("All mapping attempts failed; returning original circuit.")
    return qc, {}


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


def create_feature_dict(qc: QuantumCircuit) -> dict[str, int | NDArray[np.float64]]:
    """Creates a feature dictionary for a given quantum circuit.

    Arguments:
        qc: The quantum circuit for which the feature dictionary is created.

    Returns:
        The feature dictionary for the given quantum circuit.
    """
    feature_dict = {
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
