# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""This module contains the functions to calculate the reward of a quantum circuit on a given device."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from mqt.bench.utils import calc_supermarq_features
from qiskit.converters import circuit_to_dag

if TYPE_CHECKING:
    from mqt.bench.devices import Device
    from numpy.typing import NDArray
    from qiskit import QuantumCircuit


def hellinger_distance(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """Calculates the Hellinger distance between two probability distributions."""
    assert np.isclose(np.sum(p), 1, 0.05), "p is not a probability distribution"
    assert np.isclose(np.sum(q), 1, 0.05), "q is not a probability distribution"

    return float((1 / np.sqrt(2)) * np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def calc_device_specific_features(
    qc: QuantumCircuit, device: Device, ignore_gates: list[str] | None = None
) -> NDArray[np.float64]:
    """Creates and returns a device-specific feature vector for a given quantum circuit and device.

    Arguments:
        qc: The quantum circuit for which the features are calculated.
        device: The device for which the features are calculated.
        ignore_gates: A list of gates to be ignored when calculating the features. Defaults to ["barrier", "id"].

    Returns:
        The device-specific feature vector of the given quantum circuit consisting of:
        - The number of operations for each native gate (excluding `ignore_gates`)
        - The active qubits (one-hot encoded e.g. {Qubit1: 1, Qubit2: 0, Qubit3: 0})
        - The depth of the quantum circuit
        - The number of qubits active in the computations
        - The supermarq features (wo program communication)
        - The directed program communication
        - The single and multi qubit gate ratio
    """
    if ignore_gates is None:
        ignore_gates = ["barrier", "id", "measure"]

    # Create a dictionary with all native gates
    native_gate_dict = {gate: 0.0 for gate in device.basis_gates if gate not in ignore_gates}
    # Add the number of operations for each native gate
    for key, val in qc.count_ops().items():
        if key in native_gate_dict:
            native_gate_dict[key] = val
    feature_dict = native_gate_dict

    # Create a list of zeros for the one-hot vector
    active_qubits_dict = dict.fromkeys(qc.qubits, 0)
    # Iterate over the operations in the quantum circuit
    for op in qc.data:
        if op.operation.name in ignore_gates:
            continue
        # Mark the qubits that are used in the operation as active
        for qubit in op.qubits:
            active_qubits_dict[qubit] = 1
    feature_dict.update(active_qubits_dict)

    # Add the depth of the quantum circuit to the feature dictionary
    feature_dict["depth"] = qc.depth()

    # Add the number of active qubits to the feature dictionary
    num_active_qubits = sum(active_qubits_dict.values())
    feature_dict["num_qubits"] = num_active_qubits  # NOTE: not used in feature vector

    # Calculate supermarq features, which uses circ.num_qubits
    assert device.num_qubits == qc.num_qubits
    supermarq_features = calc_supermarq_features(qc)
    feature_dict["critical_depth"] = supermarq_features.critical_depth
    feature_dict["entanglement_ratio"] = supermarq_features.entanglement_ratio
    feature_dict["parallelism"] = supermarq_features.parallelism
    feature_dict["liveness"] = supermarq_features.liveness

    # NOTE: renormalize using the number of active qubits (not the device's num_qubits)
    feature_dict["parallelism"] = (
        feature_dict["parallelism"] * (device.num_qubits - 1) / (num_active_qubits - 1) if num_active_qubits >= 2 else 0
    )
    feature_dict["liveness"] = (
        feature_dict["liveness"] * device.num_qubits / num_active_qubits if num_active_qubits >= 1 else 0
    )

    # Calculate additional features based on DAG
    dag = circuit_to_dag(qc)
    dag.remove_all_ops_named("barrier")
    dag.remove_all_ops_named("measure")

    # Directed program communication = circuit's average directed qubit degree / degree of a complete directed graph.
    di_graph = nx.DiGraph()
    for op in dag.two_qubit_ops():
        q1, q2 = op.qargs
        di_graph.add_edge(qc.find_bit(q1).index, qc.find_bit(q2).index)
    degree_sum = sum(di_graph.degree(n) for n in di_graph.nodes)
    directed_program_communication = (
        degree_sum / (2 * num_active_qubits * (num_active_qubits - 1)) if num_active_qubits >= 2 else 0
    )

    # Average number of 1q gates per layer = num of 1-qubit gates in the circuit / depth
    single_qubit_gates_per_layer = (
        (len(dag.gate_nodes()) - len(dag.two_qubit_ops())) / dag.depth() if dag.depth() > 0 else 0
    )
    # Average number of 2q gates per layer = num of 2-qubit gates in the circuit / depth
    multi_qubit_gates_per_layer = len(dag.two_qubit_ops()) / dag.depth() if dag.depth() > 0 else 0

    # Normalize both values using the available number of qubits
    single_qubit_gates_per_layer = single_qubit_gates_per_layer / num_active_qubits if num_active_qubits > 1 else 0
    multi_qubit_gates_per_layer = multi_qubit_gates_per_layer / (num_active_qubits // 2) if num_active_qubits > 2 else 0

    feature_dict["directed_program_communication"] = directed_program_communication
    feature_dict["single_qubit_gates_per_layer"] = single_qubit_gates_per_layer
    feature_dict["multi_qubit_gates_per_layer"] = multi_qubit_gates_per_layer

    return np.array(list(feature_dict.values()))


def get_hellinger_model_path(device: Device) -> Path:
    """Returns the path to the trained model folder resulting from the machine learning training."""
    training_data_path = Path(str(resources.files("mqt.predictor"))) / "ml" / "training_data"
    model_path = (
        training_data_path / "trained_model" / ("trained_hellinger_distance_regressor_" + device.name + ".joblib")
    )
    return Path(model_path)
