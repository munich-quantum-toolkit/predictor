# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
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
from typing import TYPE_CHECKING, SupportsFloat, SupportsIndex, cast

import numpy as np
import torch
from qiskit import transpile
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score

from mqt.predictor.utils import calc_supermarq_features

if TYPE_CHECKING:
    import torch_geometric
    from numpy._typing import NDArray
    from qiskit import QuantumCircuit
    from torch import nn


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


def get_path_trained_model_gnn(figure_of_merit: str) -> Path:
    """Return the path to the GNN checkpoint file for the given figure of merit."""
    base_dir = get_path_training_data() / "trained_model"
    filename = f"trained_gnn_{figure_of_merit}.pth"
    return base_dir / filename


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


def get_openqasm3_gates() -> list[str]:
    """Returns a list of all quantum gates within the openQASM 3.0 standard header."""
    # according to https://openqasm.com/language/standard_library.html#standard-library
    # Snapshot from OpenQASM 3.0 specification (version 3.0)
    # Verify against latest spec when Qiskit or OpenQASM updates
    return [
        "x",
        "y",
        "z",
        "h",
        "s",
        "sdg",
        "t",
        "tdg",
        "sx",
        "p",
        "rx",
        "ry",
        "rz",
        "u",
        "cx",
        "cy",
        "cz",
        "ch",
        "cp",
        "crx",
        "cry",
        "crz",
        "cu",
        "swap",
        "ccx",
        "cswap",
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


def create_dag(qc: QuantumCircuit) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Creates and returns the feature-annotated DAG of the quantum circuit.

    This is particularly useful for GNN-based models. It is needed to convert the quantum circuit
    into a graph representation. Then, each node is annotated with features such as one-hot gate encoding,
    sin/cos of parameters, arity, number of controls, number of parameters, critical path flag, fan-in and fan-out.

    Arguments:
        qc: The quantum circuit to be converted to a DAG.

    Returns:
        node_vector: features per node = [one-hot gate, sin/cos params, arity, controls,
        num_params, critical_flag, fan_in, fan_out]
        edge_index: 2 for E tensor of edges (src, dst)
        number_of_gates: number of nodes in the DAG.
    """
    # 0) cleanup & DAG
    pm = PassManager(RemoveBarriers())
    qc = pm.run(qc)
    try:
        qc = transpile(qc, optimization_level=0, basis_gates=get_openqasm3_gates())
    except Exception as e:
        msg = f"Transpilation failed: {e}"
        raise ValueError(msg) from e
    dag = circuit_to_dag(qc)

    unique_gates = [*get_openqasm3_gates(), "measure"]
    gate2idx = {g: i for i, g in enumerate(unique_gates)}
    number_gates = len(unique_gates)

    def _safe_float(val: object, default: float = 0.0) -> float:
        try:
            return float(val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default

    # --- parameters sin/cos (max 3 param) ---
    def param_vector(node: DAGOpNode, dim: int = 3) -> list[float]:
        """Return [sin(p1), cos(p1), sin(p2), cos(p2), sin(p3), cos(p3)]."""
        raw_params = getattr(node.op, "params", [])
        params = [_safe_float(v) for v in raw_params[:dim]]
        params += [0.0] * (dim - len(params))

        out: list[float] = []
        for p in params:
            out.extend([np.sin(p), np.cos(p)])
        return out  # len = 2*dim

    nodes = list(dag.op_nodes())
    number_nodes = len(nodes)

    # prealloc
    onehots = torch.zeros((number_nodes, number_gates), dtype=torch.float32)
    num_params = torch.zeros((number_nodes, 1), dtype=torch.float32)
    params = torch.zeros((number_nodes, 6), dtype=torch.float32)
    arity = torch.zeros((number_nodes, 1), dtype=torch.float32)
    controls = torch.zeros((number_nodes, 1), dtype=torch.float32)
    fan_in = torch.zeros((number_nodes, 1), dtype=torch.float32)
    fan_out = torch.zeros((number_nodes, 1), dtype=torch.float32)

    for i, node in enumerate(nodes):
        name = node.op.name
        if name not in unique_gates:
            msg = f"Unknown gate: {name}"
            raise ValueError(msg)
        onehots[i, gate2idx[name]] = 1.0
        params[i] = torch.tensor(param_vector(node), dtype=torch.float32)
        arity[i] = float(len(node.qargs))
        controls[i] = float(getattr(node.op, "num_ctrl_qubits", 0))
        num_params[i] = float(len(getattr(node.op, "params", [])))
        preds = [p for p in dag.predecessors(node) if isinstance(p, DAGOpNode)]
        succs = [s for s in dag.successors(node) if isinstance(s, DAGOpNode)]
        fan_in[i] = len(preds)
        fan_out[i] = len(succs)

    # edges DAG
    idx_map = {node: i for i, node in enumerate(nodes)}
    edges: list[list[int]] = []
    for src, dst, _ in dag.edges():
        if src in idx_map and dst in idx_map:
            edges.append([idx_map[src], idx_map[dst]])
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # --- critical path detection ---
    topo_nodes = list(dag.topological_op_nodes())
    if not topo_nodes:
        # No operation nodes: return node features with zero critical flags
        critical_flag = torch.zeros((number_nodes, 1), dtype=torch.float32)
        node_vector = torch.cat([onehots, params, arity, controls, num_params, critical_flag, fan_in, fan_out], dim=1)
        return node_vector, edge_index, number_nodes

    dist_in = dict.fromkeys(topo_nodes, 0)
    for node in topo_nodes:
        preds = [p for p in dag.predecessors(node) if isinstance(p, DAGOpNode)]
        if preds:
            dist_in[node] = max(dist_in.get(p, 0) + 1 for p in preds)

    dist_out = dict.fromkeys(topo_nodes, 0)
    for node in reversed(topo_nodes):
        succs = [s for s in dag.successors(node) if isinstance(s, DAGOpNode)]
        if succs:
            dist_out[node] = max(dist_out.get(s, 0) + 1 for s in succs)

    critical_len = max(dist_in.get(n, 0) + dist_out.get(n, 0) for n in topo_nodes)

    critical_flag = torch.zeros((number_nodes, 1), dtype=torch.float32)
    for i, node in enumerate(nodes):
        # set critical flag to 1 if on critical path
        if dist_in.get(node, 0) + dist_out.get(node, 0) == critical_len:
            critical_flag[i] = 1.0

    # final concat of features
    node_vector = torch.cat([onehots, params, arity, controls, num_params, critical_flag, fan_in, fan_out], dim=1)

    return node_vector, edge_index, number_nodes


def get_gnn_input_features() -> int:
    """Calculate GNN input feature dimension.

    Components:
    - len(get_openqasm3_gates()) + 1: one-hot gate encoding (including 'measure')
    - 6: sin/cos for up to 3 parameters
    - 1: arity
    - 1: controls
    - 1: num_params
    - 1: critical_flag
    - 2: fan_in, fan_out
    """
    return len(get_openqasm3_gates()) + 1 + 6 + 1 + 1 + 1 + 1 + 2


def get_results_classes(preds: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return predicted and target class indices.

    Arguments:
        preds: model predictions
        targets: ground truth targets (one-hot encoded or integer class labels)

    Returns:
        pred_idx: predicted class indices
        targets_idx: target class indices
    """
    pred_idx = torch.argmax(preds, dim=1)
    if targets.dim() == 1:
        targets_idx = targets.long()
    elif targets.size(1) == 1:
        targets_idx = targets.view(-1).long()
    else:
        targets_idx = torch.argmax(targets, dim=1)
    return pred_idx, targets_idx


# ---------------------------------------------------
# Evaluation
# ---------------------------------------------------
def evaluate_classification_model(
    model: nn.Module,
    loader: torch_geometric.loader.DataLoader,
    loss_fn: nn.Module,
    device: str | torch.device,
    *,
    return_arrays: bool = False,
    verbose: bool = False,
) -> tuple[float, dict[str, float | str], tuple[np.ndarray, np.ndarray] | None]:
    """Evaluate a classification model with the given loss function and compute accuracy metrics.

    Arguments:
        model: classification model to be evaluated
        loader: data loader for the evaluation dataset
        loss_fn: loss function for evaluation
        device: device to be used for evaluation (cuda or cpu)
        return_arrays: whether to return prediction and target arrays
        verbose: whether to print the metrics results.

    Returns:
        avg_loss: average loss over the loader
        metrics:  {"custom_accuracy": ..., "classification_report": ..., "mse": ..., "rmse": ..., "mae": ..., "r2": ...}
        arrays:   (preds, y_true) if return_arrays=True, else None.
    """
    if isinstance(device, str):
        device = torch.device(device)
    metrics: dict[str, float | str] = {}
    model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in loader:
            batch_device = batch.to(device)
            preds = model(batch_device)
            preds = torch.clamp(preds, 0.0, 1.0)
            targets = batch_device.y.float()

            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            if preds.shape != targets.shape:
                msg = f"Shape mismatch: preds {preds.shape} vs targets {targets.shape}"
                raise ValueError(msg)

            bs = targets.size(0)
            loss = loss_fn(preds, targets)
            total_loss += loss.item() * bs
            total += bs

            all_preds.append(preds.detach().cpu())
            all_targets.append(targets.detach().cpu())

    avg_loss = total_loss / max(1, total)
    metrics = {"loss": float(avg_loss)}
    if not all_preds or not all_targets:
        arrays = (np.array([]), np.array([])) if return_arrays else None
        return avg_loss, metrics, arrays

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # --- compute accuracy ---
    pred_classes, target_classes = get_results_classes(preds, targets)
    acc = accuracy_score(target_classes, pred_classes)
    classification_report_res = classification_report(target_classes, pred_classes)
    metrics["custom_accuracy"] = float(acc)
    metrics["classification_report"] = classification_report_res

    if verbose:
        mse = mean_squared_error(targets.numpy().reshape(-1), preds.numpy().reshape(-1))
        mae = mean_absolute_error(targets.numpy().reshape(-1), preds.numpy().reshape(-1))
        rmse = float(np.sqrt(mse))
        if targets.size(0) < 2 or torch.all(targets == targets[0]):
            r2 = float("nan")
        else:
            r2 = float(r2_score(targets.numpy().reshape(-1), preds.numpy().reshape(-1)))
        metrics.update({"mse": float(mse), "rmse": float(rmse), "mae": float(mae), "r2": float(r2)})

    arrays = (preds.numpy(), targets.numpy()) if return_arrays else None
    return avg_loss, metrics, arrays


def evaluate_regression_model(
    model: nn.Module,
    loader: torch_geometric.loader.DataLoader,
    loss_fn: nn.Module,
    device: str | torch.device,
    *,
    return_arrays: bool = False,
    verbose: bool = False,
) -> tuple[float, dict[str, float | str], tuple[np.ndarray, np.ndarray] | None]:
    """Evaluate a regression model (logits = scalar predictions).

    Arguments:
        model: regression model to be evaluated
        loader: data loader for the evaluation dataset
        loss_fn: loss function for evaluation
        device: device to be used for evaluation (cuda or cpu)
        return_arrays: whether to return prediction and target arrays
        verbose: whether to print the metrics results.

    Returns:
        avg_loss: average loss over the loader
        metrics:  {"rmse": ..., "mae": ..., "r2": ...}
        arrays:   (preds, y_true) if return_arrays=True, else None
    """
    if isinstance(device, str):
        device = torch.device(device)
    model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in loader:
            batch_device = batch.to(device)
            logits = model(batch_device)
            y = batch_device.y.float().view_as(logits)

            loss = loss_fn(logits, y)
            bs = y.numel()
            total_loss += loss.item() * bs
            total += bs

            preds_1d = logits.view(-1).detach().cpu().numpy()
            y_1d = y.view(-1).detach().cpu().numpy()
            all_preds.append(preds_1d)
            all_targets.append(y_1d)

    avg_loss = total_loss / max(1, total)
    preds = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
    y_true = np.concatenate(all_targets, axis=0) if all_targets else np.array([])

    metrics: dict[str, float | str] = {"loss": float(avg_loss)}
    if preds.size > 0:
        rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
        mae = float(mean_absolute_error(y_true, preds))
        r2 = float("nan") if y_true.shape[0] < 2 or np.all(y_true == y_true[0]) else float(r2_score(y_true, preds))
        metrics.update({"rmse": rmse, "mae": mae, "r2": r2})

        if verbose:
            print(f"[Eval] loss={avg_loss:.6f} | rmse={rmse:.6f} | mae={mae:.6f} | r2={metrics['r2']:.6f}")

    arrays = (preds, y_true) if return_arrays else None
    return avg_loss, metrics, arrays


def train_model(
    model: nn.Module,
    train_loader: torch_geometric.loader.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    num_epochs: int,
    task: str,
    *,
    device: str | torch.device | None = None,
    verbose: bool = True,
    val_loader: torch_geometric.loader.DataLoader | None = None,
    patience: int = 10,
    min_delta: float = 0.0,
    restore_best: bool = True,
) -> None:
    """Trains a GNN model with optional early stopping based on validation loss (MSE Loss).

    Arguments:
        model: regression model to be trained
        train_loader: training set split into mini-batch
        optimizer: optimizer for model training
        loss_fn: loss function for training
        num_epochs: number of training epochs
        task: either "classification" or "regression"
        device: device to be used for training (gpu or cpu)
        verbose: whether to print progress messages
        val_loader: validation set split into mini-batch (optional)
        patience: number of epochs with no improvement after which training will be stopped
        min_delta: minimum change in the monitored quantity to qualify as an improvement
        restore_best: whether to restore model weights from the epoch with the best validation loss
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif isinstance(device, str):
        device = torch.device(device)
    model.to(device)

    best_state, best_metric = None, float("inf")
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss, total = 0.0, 0

        for batch in train_loader:
            batch_device = batch.to(device)
            preds = model(batch_device)

            targets = batch_device.y
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = targets.size(0)
            running_loss += loss.item() * bs
            total += bs

        train_loss = running_loss / max(1, total)

        if val_loader is not None:
            if task == "classification":
                # evaluate on validation set the loss function and metrics connected to classification
                # Passed the model, val_loader, loss function, device, and
                # verbose (leaving the user decide if also wants to print metrics during evaluation)
                val_loss, val_metrics, _ = evaluate_classification_model(
                    model, val_loader, loss_fn, device=str(device), verbose=verbose
                )
            elif task == "regression":
                # evaluate on validation set the loss function and metrics connected to regression
                # Passed the model, val_loader, loss function, device, and
                # verbose (leaving the user decide if also wants to print metrics during evaluation)
                val_loss, val_metrics, _ = evaluate_regression_model(
                    model, val_loader, loss_fn, device=str(device), verbose=verbose
                )
            else:
                # raise an error if task not classification or regression
                msg = "Task variable not regression or classification"
                raise ValueError(msg)
            improved = (best_metric - val_loss) > min_delta
            if improved:
                best_metric = val_loss
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if verbose:
                if task == "classification":
                    print(
                        f"Epoch {epoch:03d}/{num_epochs} | train_loss={train_loss:.6f} | "
                        f"val_loss={val_loss:.6f} | acc={val_metrics.get('custom_accuracy', 0):.4f} | patience={epochs_no_improve}/{patience}"
                    )
                else:
                    r2 = val_metrics.get("r2", 0)

                    print(
                        f"Epoch {epoch:03d}/{num_epochs} | train_loss={train_loss:.6f} | "
                        f"val_loss={val_loss:.6f} | r2={r2:.4f} | patience={epochs_no_improve}/{patience}"
                    )

            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}.")
                break
        else:
            if verbose:
                print(f"Epoch {epoch:03d}/{num_epochs} | train_loss={train_loss:.6f}")

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)


@dataclass
class TrainingData:
    """Container for training/test data for both classical (numpy) and GNN (graph) models."""

    X_train: NDArray[np.float64] | list[torch_geometric.data.Data]
    y_train: NDArray[np.float64] | torch.Tensor
    X_test: NDArray[np.float64] | list[torch_geometric.data.Data] | None = None
    y_test: NDArray[np.float64] | torch.Tensor | None = None
    indices_train: list[int] | None = None
    indices_test: list[int] | None = None
    names_list: list[str] | None = None
    scores_list: list[list[float]] | None = None
