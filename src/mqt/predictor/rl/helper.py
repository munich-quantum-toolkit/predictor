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
from qiskit.circuit import ClassicalRegister, QuantumRegister
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import PassManager
from qiskit_ibm_transpiler.ai.routing import AIRouting

from mqt.predictor.utils import calc_supermarq_features

if TYPE_CHECKING:
    from numpy.random import Generator
    from numpy.typing import NDArray

import zipfile
from importlib import resources

logger = logging.getLogger("mqt-predictor")


def extract_cregs_and_measurements(qc):
    cregs = [ClassicalRegister(cr.size, name=cr.name) for cr in qc.cregs]
    measurements = [(item.operation, item.qubits, item.clbits) for item in qc.data if item.operation.name == "measure"]
    return cregs, measurements


def remove_cregs(qc):
    qregs = [QuantumRegister(qr.size, name=qr.name) for qr in qc.qregs]
    new_qc = QuantumCircuit(*qregs)
    old_to_new = {}
    for orig_qr, new_qr in zip(qc.qregs, new_qc.qregs, strict=False):
        for idx in range(orig_qr.size):
            old_to_new[orig_qr[idx]] = new_qr[idx]
    for item in qc.data:
        instr = item.operation
        qargs = [old_to_new[q] for q in item.qubits]
        if instr.name not in ("measure", "barrier"):
            new_qc.append(instr, qargs)
    return new_qc


def add_cregs_and_measurements(qc, cregs, measurements, qubit_map=None):
    for cr in cregs:
        qc.add_register(cr)
    for instr, qargs, cargs in measurements:
        new_qargs = [qubit_map[q] for q in qargs] if qubit_map else qargs
        qc.append(instr, new_qargs, cargs)
    return qc


class SafeAIRouting(AIRouting):
    """Remove cregs before AIRouting and add them back afterwards
    Necessary because there are cases AIRouting can't handle.
    """

    def run(self, dag):
        # 1. Convert input dag to circuit
        qc_orig = dag_to_circuit(dag)

        # 2. Extract classical registers and measurement instructions
        cregs, measurements = extract_cregs_and_measurements(qc_orig)

        # 3. Remove cregs and measurements
        qc_noclassical = remove_cregs(qc_orig)

        # 4. Convert back to dag and run routing (AIRouting)
        dag_noclassical = circuit_to_dag(qc_noclassical)
        dag_routed = super().run(dag_noclassical)

        # 5. Convert routed dag to circuit for restoration
        qc_routed = dag_to_circuit(dag_routed)

        # 6. Build mapping from original qubits to qubits in routed circuit
        final_layout = getattr(self, "property_set", {}).get("final_layout", None)
        if final_layout is None and hasattr(dag_routed, "property_set"):
            final_layout = dag_routed.property_set.get("final_layout", None)

        qubit_map = {}
        for virt in qc_orig.qubits:
            phys = final_layout[virt]
            if isinstance(phys, int):
                qubit_map[virt] = qc_routed.qubits[phys]
            else:
                try:
                    idx = qc_routed.qubits.index(phys)
                except ValueError:
                    msg = f"Physical qubit {phys} not found in output circuit!"
                    raise RuntimeError(msg)
                qubit_map[virt] = qc_routed.qubits[idx]
                # 7. Restore classical registers and measurement instructions
        qc_final = add_cregs_and_measurements(qc_routed, cregs, measurements, qubit_map)
        # 8. Return as dag
        return circuit_to_dag(qc_final)


def best_of_n_passmanager(
    action,
    device,
    qc,
    max_iteration=(20, 20),
    metric_fn=None,
):
    """Runs the given transpile_pass multiple times and keeps the best result.
    action: the action dict with a 'transpile_pass' key (lambda/device->[passes])
    device: the backend or device
    qc: input circuit
    max_iteration: number of times to try
    metric_fn: function(circ) -> float for scoring
    require_layout: skip outputs with missing layouts.
    """
    best_val = None
    best_result = None
    best_property_set = None

    if action["name"] == "SabreLayout+AIRouting":
        all_passes = action.transpile_pass(device, max_iteration)
    else:
        all_passes = action.transpile_pass(device)

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


def get_openqasm_gates() -> list[str]:
    """Returns a list of all quantum gates within the openQASM 2.0 standard header.

    According to https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/qasm/libs/qelib1.inc
    Removes generic single qubit gates u1, u2, u3 since they are no meaningful features for RL

    """
    return [
        "cx",
        "id",
        "u",
        "p",
        "x",
        "y",
        "z",
        "h",
        "r",
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
        "csx",
        "cu",
        "rxx",
        "rzz",
        "rccx",
    ]


def create_feature_dict(
    qc: QuantumCircuit, basis_gates: list[str], coupling_map
) -> dict[str, int | NDArray[np.float64]]:
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
