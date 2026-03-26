# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Helper functions of the reinforcement learning compilation predictor."""

from __future__ import annotations

import math
import zipfile
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers
from torch_geometric.data import Data

from mqt.predictor.utils import calc_supermarq_features, get_openqasm_gates_for_rl

if TYPE_CHECKING:
    from numpy.random import Generator
    from numpy.typing import NDArray


# Number of circuit-level global features appended to each PyG Data object when
# graph=True.  Features (in order):
#   0. log-normalized num_qubits
#   1. log-normalized depth
#   2. program_communication
#   3. critical_depth
#   4. entanglement_ratio
#   5. parallelism
#   6. liveness
GLOBAL_FEATURE_DIM: int = 7


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
        qc = QuantumCircuit.from_qasm_file(file_list[random_index])
    except Exception as e:
        msg = f"Could not read QuantumCircuit from: {file_list[random_index]}"
        raise RuntimeError(msg) from e

    return qc, str(file_list[random_index])


def get_bqskit_gates() -> list[str]:
    """Returns a list of gate names matching the OpenQASM 3.0 standard library gates supported in BQSKit."""
    return [
        # --- 1-qubit gates ---
        "id",
        "x",
        "y",
        "z",
        "h",
        "s",
        "sdg",
        "t",
        "tdg",
        "sx",
        "rx",
        "ry",
        "rz",
        "u",
        "u1",
        "u2",
        "u3",
        # --- Controlled 1-qubit gates ---
        "cx",
        "cy",
        "cz",
        "ch",
        "crx",
        "cry",
        "crz",
        "cp",
        "cu",
        "cu1",
        "cu2",
        "cu3",
        # --- 2-qubit gates ---
        "swap",
        "iswap",
        "ecr",
        "rzz",
        "rxx",
        "ryy",
        "zz",
        # --- 3-qubit gates ---
        "ccx",
        # --- Others ---
        "reset",
    ]


def create_dag(qc: QuantumCircuit) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Create a feature-annotated DAG representation of a quantum circuit for GNN models.

    Each node corresponds to a gate operation and is annotated with:
    one-hot gate encoding, sin/cos of parameters (up to 3), arity, number of
    control qubits, number of parameters, critical-path flag, fan-in, fan-out.

    Args:
        qc: The quantum circuit to convert.

    Returns:
        node_vector: Float tensor of shape (N, F) with node features.
        edge_index: Long tensor of shape (2, E) with directed edges.
        number_of_gates: Number of operation nodes N.
    """
    pm = PassManager(RemoveBarriers())
    qc = pm.run(qc)
    dag = circuit_to_dag(qc)

    unique_gates = [*get_bqskit_gates(), "measure", "other"]
    gate2idx = {g: i for i, g in enumerate(unique_gates)}
    number_gates = len(unique_gates)

    def _safe_float(val: object, default: float = 0.0) -> float:
        try:
            return float(val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default

    def param_vector(node: DAGOpNode, dim: int = 3) -> list[float]:
        raw_params = getattr(node.op, "params", [])
        params = [_safe_float(v) for v in raw_params[:dim]]
        params += [0.0] * (dim - len(params))
        out: list[float] = []
        for p in params:
            out.extend([np.sin(p), np.cos(p)])
        return out  # length = 2 * dim

    nodes = list(dag.op_nodes())
    number_nodes = len(nodes)

    onehots = torch.zeros((number_nodes, number_gates), dtype=torch.float32)
    num_params = torch.zeros((number_nodes, 1), dtype=torch.float32)
    params = torch.zeros((number_nodes, 6), dtype=torch.float32)
    arity = torch.zeros((number_nodes, 1), dtype=torch.float32)
    controls = torch.zeros((number_nodes, 1), dtype=torch.float32)
    fan_in = torch.zeros((number_nodes, 1), dtype=torch.float32)
    fan_out = torch.zeros((number_nodes, 1), dtype=torch.float32)

    for i, node in enumerate(nodes):
        gate_name = node.op.name
        idx = gate2idx.get(gate_name, gate2idx["other"])
        onehots[i, idx] = 1.0
        params[i] = torch.tensor(param_vector(node), dtype=torch.float32)
        arity[i] = float(len(node.qargs))
        controls[i] = float(getattr(node.op, "num_ctrl_qubits", 0))
        num_params[i] = float(len(getattr(node.op, "params", [])))
        preds = [p for p in dag.predecessors(node) if isinstance(p, DAGOpNode)]
        succs = [s for s in dag.successors(node) if isinstance(s, DAGOpNode)]
        fan_in[i] = len(preds)
        fan_out[i] = len(succs)

    idx_map = {node: i for i, node in enumerate(nodes)}
    edges: list[list[int]] = []
    for src, dst, _ in dag.edges():
        if src in idx_map and dst in idx_map:
            edges.append([idx_map[src], idx_map[dst]])
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    topo_nodes = list(dag.topological_op_nodes())
    if not topo_nodes:
        critical_flag = torch.zeros((number_nodes, 1), dtype=torch.float32)
        node_vector = torch.cat([onehots, params, arity, controls, num_params, critical_flag, fan_in, fan_out], dim=1)
        return node_vector, edge_index, number_nodes

    dist_in: dict[DAGOpNode, int] = dict.fromkeys(topo_nodes, 0)
    for node in topo_nodes:
        preds = [p for p in dag.predecessors(node) if isinstance(p, DAGOpNode)]
        if preds:
            dist_in[node] = max(dist_in.get(p, 0) + 1 for p in preds)

    dist_out: dict[DAGOpNode, int] = dict.fromkeys(topo_nodes, 0)
    for node in reversed(topo_nodes):
        succs = [s for s in dag.successors(node) if isinstance(s, DAGOpNode)]
        if succs:
            dist_out[node] = max(dist_out.get(s, 0) + 1 for s in succs)

    critical_len = max(dist_in.get(n, 0) + dist_out.get(n, 0) for n in topo_nodes)
    critical_flag = torch.zeros((number_nodes, 1), dtype=torch.float32)
    for i, node in enumerate(nodes):
        if dist_in.get(node, 0) + dist_out.get(node, 0) == critical_len:
            critical_flag[i] = 1.0

    node_vector = torch.cat([onehots, params, arity, controls, num_params, critical_flag, fan_in, fan_out], dim=1)
    return node_vector, edge_index, number_nodes


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


def create_feature_dict(qc: QuantumCircuit, *, graph: bool = False) -> dict[str, int | NDArray[np.float32]] | Data:
    """Creates a feature representation for a given quantum circuit.

    Args:
        qc: The quantum circuit to represent.
        graph: If True, returns a PyG ``Data`` object suitable for GNN models.
               If False (default), returns the flat feature dictionary used by MaskablePPO.

    Returns:
        A PyG ``Data`` object when ``graph=True``, otherwise a feature dictionary.
    """
    if graph:
        node_vector, edge_index, number_nodes = create_dag(qc)
        # Attach circuit-level global features (same information used by the
        # non-GNN MaskablePPO policy) so the GNN can also reason about
        # high-level circuit properties that are not captured in the DAG structure.
        supermarq = calc_supermarq_features(qc)
        global_feature_vector = torch.tensor(
            [
                [
                    math.log1p(qc.num_qubits) / math.log1p(128),
                    math.log1p(qc.depth()) / math.log1p(10_000),
                    float(supermarq.program_communication),
                    float(supermarq.critical_depth),
                    float(supermarq.entanglement_ratio),
                    float(supermarq.parallelism),
                    float(supermarq.liveness),
                ]
            ],
            dtype=torch.float32,
        )  # shape [1, GLOBAL_FEATURE_DIM] — batched to [B, GLOBAL_FEATURE_DIM] by PyG
        return Data(x=node_vector, edge_index=edge_index, num_nodes=number_nodes, global_features=global_feature_vector)

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
