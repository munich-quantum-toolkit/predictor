# ──────────────────────────────────────────────────────────────────────────────
# Training & evaluation hooks — use YOUR existing helpers
# ──────────────────────────────────────────────────────────────────────────────

# Replace `user_train_eval` with the actual module/file where your functions live,
# or paste those definitions above this block and delete the import.
from sklearn.model_selection import KFold
from mqt.predictor.ml.helper import (
    evaluate_classification_model,
    train_classification_model,
    evaluate_regression_model,
    train_regression_model,
)
import optuna
from torch import nn
import torch
import numpy as np
from mqt.predictor.ml.helper import (
    TrainingData,
    create_dag,
    create_feature_vector,
    get_path_trained_model,
    get_path_trained_model_gnn,
    get_path_training_circuits,
    get_path_training_circuits_compiled,
    get_path_training_data,
    train_classification_model,
    train_regression_model,
)
from mqt.predictor.ml.gnn import GNN
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


# ──────────────────────────────────────────────────────────────────────────────
# Objective with k-fold CV
# ──────────────────────────────────────────────────────────────────────────────
def objective(trial: optuna.Trial,
              dataset: TrainingData,
              task: str,
              in_feats: int,
              num_outputs: int,
              device: str,
              loss_fn: nn.Module,
              optimizer: torch.optim.Optimizer,
              k_folds: int,
              classes:list[str] | None = None,
              batch_size:int=32,
              num_epochs:int=100,
              patience:int=10) -> float:

    """
    Objective function for Optuna hyperparameter optimization.

    Arguments:
        trial: The Optuna trial object.
        in_feats: number of input features.
        num_outputs: number of output features.
        device: device to use for training.
        loss_fn: loss function to use.
        optimizer: optimizer to use.
        k_folds: number of folds for cross-validation.
        classes: list of class names (for classification tasks).
        batch_size: batch size for training.
        num_epochs: number of epochs for training.
        patience: patience for early stopping.
    Returns:
        mean_val: The mean value in validation considering the k-folds.
    """
    # Type of device used
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)

    # Hyperparameter spaces
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    num_resnet_layers = trial.suggest_int("num_resnet_layers", 1, 10)
    mlp_depth = trial.suggest_int("mlp_depth", 1, 3)
    mlp_choices = [32, 64, 128, 256, 512, 1024]
    mlp_units = [trial.suggest_categorical(f"mlp_units_{i}", mlp_choices) for i in range(mlp_depth)]


    # Split into k-folds
    kf = KFold(n_splits=k_folds, shuffle=True)
    fold_val_best_losses: list[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset)))):
        train_subset = [dataset[i] for i in train_idx]
        val_subset = [dataset[i] for i in val_idx]
        # Transform the data into loaders
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        # Define the GNN
        model = GNN(
            in_feats=in_feats,
            hidden_dim=hidden_dim,
            num_resnet_layers=num_resnet_layers,
            mlp_units=mlp_units,
            num_outputs=num_outputs,
            classes=classes).to(device_obj)


        # Based on the task, do a training and evaluation for regression or classification
        if task == "regression":
            train_regression_model(
                model,
                train_loader,
                optimizer,
                loss_fn,
                num_epochs=num_epochs,
                device=device,
                verbose=False,
                val_loader=val_loader,
                patience=patience,
                min_delta=0.0,
                restore_best=True,
                scheduler=None,
            )
            val_loss, _, _ = evaluate_regression_model(
                model, val_loader, loss_fn, device=device, return_arrays=False, verbose=False
            )
        else:
            train_classification_model(
                model,
                train_loader,
                optimizer,
                loss_fn,
                num_epochs=num_epochs,
                task=task,
                device=device,
                verbose=False,
                val_loader=val_loader,
                patience=patience,
                min_delta=0.0,
                restore_best=True,
                scheduler=None,
            )
            val_loss, _, _ = evaluate_classification_model(
                model, val_loader, loss_fn, task=task, device=device, return_arrays=False, verbose=False
            )

        fold_val_best_losses.append(float(val_loss))
    # Take the mean value
    mean_val = float(np.mean(fold_val_best_losses))
    trial.set_user_attr("fold_val_best_losses", fold_val_best_losses)
    trial.set_user_attr(
        "best_hparams",
        {
            "in_feats": in_feats,
            "hidden_dim": hidden_dim,
            "num_resnet_layers": num_resnet_layers,
            "mlp_units": mlp_units,
            "num_outputs": num_outputs,     
        },
    )
    return mean_val
