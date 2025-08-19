# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Helper functions for the machine learning device selection predictor."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from qiskit.converters import circuit_to_dag
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from torch import nn

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers
from mqt.predictor.utils import calc_supermarq_features

if TYPE_CHECKING:
    import torch_geometric
    from numpy._typing import NDArray
    from qiskit import QuantumCircuit
    from qiskit.dagcircuit import DAGOpNode


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


def create_dag(qc: QuantumCircuit) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Creates and returns the associate DAG of the quantum circuit.

    Arguments:
        qc: the quantum circuit to be compiled

    Returns:
        node_vector: node vectors, each element of the vector contains a vector
                    which describes the type of operation, the qubits involved
                    and the associated parameters
        edge_index: edge_matrix describing the associated graph
        number_of_gates: the number of nodes, and so the operations applied
    """
    # Get the number of qubits
    num_qubits = qc.num_qubits
    # remove barriers
    pm = PassManager(RemoveBarriers())
    qc = pm.run(qc)
    # Transform the circuit into a DAG
    dag = circuit_to_dag(qc)

    unique_gates = [*get_openqasm_gates(), "measure"]
    gate2idx = {g: i for i, g in enumerate(unique_gates)}
    number_unique_gates = len(unique_gates)

    def qubit_vector(node: DAGOpNode) -> list[int]:
        """Return [target, ctrl1, ctrl2], fill -1 if missing."""
        qinds = [qc.find_bit(q).index for q in node.qargs]
        # from the node get the number of control qubits (if field missing, set 0)
        n_ctrl = getattr(node.op, "num_ctrl_qubits", 0)
        # assume controls appear first, then target:
        ctrls = qinds[:n_ctrl]

        tgt = qinds[-1] if qinds else -1
        # pad to 2 controls
        ctrls = ctrls + [-1] * (2 - len(ctrls))
        return [tgt, ctrls[0], ctrls[1]]

    # helper to extract up to 3 real-valued params
    def param_vector(node: DAGOpNode, dim: int = 3) -> list[float]:
        p = [float(val) for val in node.op.params]
        p = p[:dim]  # truncate if more than dim
        return p + [0.0] * (dim - len(p))  # pad with zeros

    nodes = list(dag.op_nodes())
    number_of_gates = len(nodes)

    # preallocate feature arrays
    onehots = torch.zeros((number_of_gates, number_unique_gates), dtype=torch.float)
    qubits = torch.full((number_of_gates, 3), -1, dtype=torch.long)
    params = torch.zeros((number_of_gates, 3), dtype=torch.float)

    for i, node in enumerate(nodes):
        # 2a) one-hot gate
        # check if name gate in unique_gates
        if node.op.name not in unique_gates:
            # otherwise raise an error
            msg = f"Unknown gate: {node.op.name}"
            raise ValueError(msg)
        onehots[i, gate2idx[node.op.name]] = 1.0

        # 2b) [target, ctrl1, ctrl2]
        qubits[i] = torch.tensor(qubit_vector(node), dtype=torch.long) / num_qubits

        # 2c) up to 3 angle params

        params[i] = torch.tensor(param_vector(node), dtype=torch.float) % (2 * math.pi)

        node_vector = torch.cat([onehots, qubits.float(), params], dim=1)

    # build edges
    idx_map = {node: i for i, node in enumerate(nodes)}
    edges = []
    for src, dst, _ in dag.edges():
        if src in idx_map and dst in idx_map:
            edges.append([idx_map[src], idx_map[dst]])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return node_vector, edge_index, number_of_gates


def evaluate_classification_model(
    model: nn.Module,
    loader: torch_geometric.loader.DataLoader,
    loss_fn: nn.Module,
    *,
    task: str = "binary",
    device: str | None = None,
    return_arrays: bool = False,
    verbose: bool = False,
) -> tuple[float, dict[str, float], tuple[np.ndarray, np.ndarray] | None]:
    """Evaluate the models.

    Arguments:
        model: the model to be evaluated, model's output must be logits
        loader: contain the set in a minibatch approach
        loss_fn: is the loss function used
        task: describe which kind of classification is done
        device: where to run the evaluation (gpu or cpu)
        return_arrays: decide if return the probability and targets.
        verbose: set as True if you want also the metrics results
    Returns:
        avg_loss: average loss measured
        metrics: dictionary containing the metrics of the model
        arrays: an array containing the probabilities of the targets and the actual value
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model.eval()
    total_loss, total = 0.0, 0
    all_logits, all_targets = [], []

    # --- no decorator; use context manager instead ---
    with torch.no_grad():
        for batch in loader:
            batch_device = batch.to(device)
            logits = model(batch_device)  # [B,1] or [B,K]
            y = batch_device.y

            # unify shapes for loss computation
            if task == "multiclass":
                if y.dim() > 1:
                    y = y.squeeze(-1)
                y_loss = y.long()
                bs = y_loss.size(0)
            elif task == "binary":
                y_loss = y.float().view(-1, 1)
                bs = y_loss.size(0)
            else:
                msg = f"Unknown task: {task}"
                raise ValueError(msg)

            loss = loss_fn(logits, y_loss)
            total_loss += loss.item() * bs
            total += bs

            all_logits.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu())

    avg_loss = total_loss / max(1, total)
    logits = torch.cat(all_logits, dim=0)
    y_true = torch.cat(all_targets, dim=0)

    metrics: dict[str, float] = {"loss": float(avg_loss)}

    # ---- Convert logits -> probs / preds & compute sklearn metrics ----
    if verbose:
        if task == "binary":
            probs = torch.sigmoid(logits).squeeze(-1).numpy()  # [N]
            y_bin = y_true.view(-1).numpy().astype(int)  # [N]
            preds = (probs >= 0.5).astype(int)

            metrics["accuracy"] = accuracy_score(y_bin, preds)
            metrics["precision"] = precision_score(y_bin, preds, zero_division=0)
            metrics["recall"] = recall_score(y_bin, preds, zero_division=0)
            metrics["f1"] = f1_score(y_bin, preds, zero_division=0)
            if np.unique(y_bin).size > 1:
                metrics["roc_auc"] = roc_auc_score(y_bin, probs)
                metrics["avg_prec"] = average_precision_score(y_bin, probs)

            arrays = (probs, y_bin)

        elif task == "multiclass":
            probs = torch.softmax(logits, dim=1).numpy()  # [N,K]
            preds = probs.argmax(axis=1)  # [N]
            y_mc = y_true.view(-1).numpy().astype(int)

            metrics["accuracy"] = accuracy_score(y_mc, preds)
            metrics["f1_macro"] = f1_score(y_mc, preds, average="macro", zero_division=0)
            metrics["f1_micro"] = f1_score(y_mc, preds, average="micro", zero_division=0)

            arrays = (probs, y_mc)
    if return_arrays:
        return avg_loss, metrics, arrays
    return avg_loss, metrics, None


def train_classification_model(
    model: nn.Module,
    train_loader: torch_geometric.loader.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    num_epochs: int,
    *,
    task: str = "binary",
    device: str | None = None,
    verbose: bool = True,
    val_loader: torch_geometric.loader.DataLoader = None,
    patience: int = 10,
    min_delta: float = 0.0,
    restore_best: bool = True,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
) -> None:
    """Trains the model with optional early stopping on validation loss.

    Arguments:
        model: the model to be trained
        train_loader: training set split in mini-batch
        optimizer: the optimizer chosen
        loss_fn: loss function adopted
        num_epochs: number of epochs set for training
        task: type of classification (binary, multiclass)
        device: if the code is run on a cpu or a gpu
        verbose: if set true print the results obtained during the training
        val_loader: validation set which allows also to understand if apply early-stopping methods
        patience: variable used for saying how many epochs waiting for the early-stopping
        min_delta: if the loss is lower that delta, patience is incremented; otherwise reset it
        restore_best: allows to restore the best model found during training
        scheduler: scheduler used for training (optionally)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model.to(device)

    best_state = None
    best_metric = float("inf")
    best_metrics_dict: dict[str, float] = {}
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss, total = 0.0, 0

        for batch in train_loader:
            batch_device = batch.to(device)
            logits = model(batch_device)
            y = batch_device.y

            if task == "multiclass":
                if y.dim() > 1:
                    y = y.squeeze(-1)
                y_loss = y.long()
                bs = y_loss.size(0)
            elif task == "binary":
                y_loss = y.float().view(-1, 1)
                bs = y_loss.size(0)
            else:
                msg = f"Unknown task: {task}"
                raise ValueError(msg)

            loss = loss_fn(logits, y_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * bs
            total += bs

        train_loss = running_loss / max(1, total)
        if scheduler is not None:
            scheduler.step()

        if val_loader is not None:
            val_loss, val_metrics, _ = evaluate_classification_model(
                model, val_loader, loss_fn, task=task, device=str(device)
            )

            improved = (best_metric - val_loss) > min_delta
            if improved:
                best_metric = val_loss
                best_state = deepcopy(model.state_dict())  # freeze best weights
                best_metrics_dict = {"val_" + k: v for k, v in val_metrics.items()}
                best_metrics_dict["train_loss_at_best"] = float(train_loss)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if verbose:
                metrics_str = " | ".join(f"{k}={v:.6f}" for k, v in val_metrics.items())
                print(
                    f"Epoch {epoch:03d}/{num_epochs} | train_loss={train_loss:.6f} | {metrics_str} | "
                    f"no_improve={epochs_no_improve}/{patience}"
                )

            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (best val_loss={best_metric:.6f}).")
                break
        else:
            # Optional early stopping on training loss only
            improved = (best_metric - train_loss) > min_delta
            if improved:
                best_metric = train_loss
                best_state = deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if verbose:
                print(
                    f"Epoch {epoch:03d}/{num_epochs} | train_loss={train_loss:.6f} | "
                    f"no_improve={epochs_no_improve}/{patience}"
                )
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping on training loss at epoch {epoch} (best train_loss={best_metric:.6f}).")
                break

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)


def evaluate_regression_model(
    model: nn.Module,
    loader: torch_geometric.loader.DataLoader,
    loss_fn: nn.Module,
    *,
    device: str | None = None,
    return_arrays: bool = False,
    verbose: bool = False,
) -> tuple[float, dict[str, float], tuple[np.ndarray, np.ndarray] | None]:
    """Evaluate a regression model (logits = scalar predictions).

    Arguments:
        model: regression model to be evaluated
        loader: data loader for the evaluation dataset
        loss_fn: loss function for evaluation
        device: device to be used for evaluation (gpu or cpu)
        return_arrays: whether to return prediction and target arrays
        verbose: whether to print the metrics results.

    Returns:
        avg_loss: average loss over the loader
        metrics:  {"rmse": ..., "mae": ..., "r2": ...}
        arrays:   (preds, y_true) if return_arrays=True, else None
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
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

            # porta a 1D per metriche
            preds_1d = logits.view(-1).detach().cpu().numpy()
            y_1d = y.view(-1).detach().cpu().numpy()
            all_preds.append(preds_1d)
            all_targets.append(y_1d)

    avg_loss = total_loss / max(1, total)
    preds = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
    y_true = np.concatenate(all_targets, axis=0) if all_targets else np.array([])

    metrics: dict[str, float] = {"loss": float(avg_loss)}
    if preds.size > 0:
        rmse = float(math.sqrt(mean_squared_error(y_true, preds)))
        mae = float(mean_absolute_error(y_true, preds))
        r2 = float(r2_score(y_true, preds)) if np.var(y_true) > 0 else float("nan")
        metrics.update({"rmse": rmse, "mae": mae, "r2": r2})

        if verbose:
            print(f"[Eval] loss={avg_loss:.6f} | rmse={rmse:.6f} | mae={mae:.6f} | r2={metrics['r2']:.6f}")

    arrays = (preds, y_true) if return_arrays else None
    return avg_loss, metrics, arrays


def train_regression_model(
    model: nn.Module,
    train_loader: torch_geometric.loader.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    num_epochs: int,
    *,
    device: str | None = None,
    verbose: bool = True,
    val_loader: torch_geometric.loader.DataLoader | None = None,
    patience: int = 10,
    min_delta: float = 0.0,
    restore_best: bool = True,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
) -> None:
    """Train a regression model with optional early stopping on validation loss.

    Arguments:
        model: regression model to be trained
        train_loader: training set split into mini-batch
        optimizer: optimizer for model training
        loss_fn: loss function for training
        num_epochs: number of training epochs
        device: device to be used for training (gpu or cpu)
        verbose: whether to print progress messages
        val_loader: validation set split into mini-batch (optional)
        patience: number of epochs with no improvement after which training will be stopped
        min_delta: minimum change in the monitored quantity to qualify as an improvement
        restore_best: whether to restore model weights from the epoch with the best validation loss
        scheduler: learning rate scheduler (optional)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model.to(device)

    best_state = None
    best_metric = float("inf")
    best_metrics_dict: dict[str, float] = {}
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss, total = 0.0, 0

        for batch in train_loader:
            batch_device = batch.to(device)
            preds = model(batch_device)  # [B] o [B,1]
            # align y
            y = batch_device.y.float().view_as(preds)

            loss = loss_fn(preds, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * y.numel()
            total += y.numel()

        train_loss = running_loss / max(1, total)
        if scheduler is not None:
            scheduler.step()

        if val_loader is not None:
            val_loss, val_metrics, _ = evaluate_regression_model(
                model, val_loader, loss_fn, device=str(device), return_arrays=False, verbose=False
            )

            improved = (best_metric - val_loss) > min_delta
            if improved:
                best_metric = val_loss
                best_state = deepcopy(model.state_dict())
                best_metrics_dict = {"val_" + k: float(v) for k, v in val_metrics.items()}
                best_metrics_dict["train_loss_at_best"] = float(train_loss)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if verbose:
                msg_metrics = " | ".join(f"{k}={v:.6f}" for k, v in val_metrics.items())
                print(
                    f"Epoch {epoch:03d}/{num_epochs} | train_loss={train_loss:.6f} | {msg_metrics} | "
                    f"no_improve={epochs_no_improve}/{patience}"
                )

            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (best val_loss={best_metric:.6f}).")
                break
        else:
            # early stopping opzionale on training loss
            improved = (best_metric - train_loss) > min_delta
            if improved:
                best_metric = train_loss
                best_state = deepcopy(model.state_dict())
                best_metrics_dict = {"train_loss_at_best": float(train_loss)}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if verbose:
                print(
                    f"Epoch {epoch:03d}/{num_epochs} | train_loss={train_loss:.6f} | "
                    f"no_improve={epochs_no_improve}/{patience}"
                )
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping on training loss at epoch {epoch} (best train_loss={best_metric:.6f}).")
                break

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)


@dataclass
class TrainingData:
    """Dataclass for the training data."""

    X_train: NDArray[np.float64] | list[torch_geometric.data.Data]
    y_train: NDArray[np.float64] | torch.Tensor
    X_test: NDArray[np.float64] | list[torch_geometric.data.Data] | None = None
    y_test: NDArray[np.float64] | torch.Tensor | None = None
    indices_train: list[int] | None = None
    indices_test: list[int] | None = None
    names_list: list[str] | None = None
    scores_list: list[list[float]] | None = None
