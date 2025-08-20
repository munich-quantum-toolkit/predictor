# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""This module contains the Predictor class, which is used to predict the most suitable quantum device for a given quantum circuit."""

from __future__ import annotations

import logging
import sys
import zipfile
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

from joblib import dump as joblib_dump
from torch import nn
from torch_geometric.loader import DataLoader

from mqt.predictor.ml.gnn import GNN

if sys.version_info >= (3, 11) and TYPE_CHECKING:  # pragma: no cover
    from typing import assert_never
else:
    from typing_extensions import assert_never

import matplotlib.pyplot as plt
import numpy as np
import torch
from joblib import Parallel, delayed, load
from mqt.bench.targets import get_device
from qiskit import QuantumCircuit
from qiskit.qasm2 import dump
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from torch_geometric.data import Data

from mqt.predictor.hellinger import get_hellinger_model_path, get_hellinger_model_path_gnn
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
from mqt.predictor.reward import (
    crit_depth,
    estimated_hellinger_distance,
    estimated_success_probability,
    expected_fidelity,
)
from mqt.predictor.rl import Predictor as rl_Predictor
from mqt.predictor.rl import rl_compile
from mqt.predictor.utils import timeout_watcher

if TYPE_CHECKING:
    from qiskit.transpiler import Target

    from mqt.predictor.reward import figure_of_merit

import json
import warnings

# ─────────────────────────────────────────────────────────────────────────
# Suppress torch-geometric "plugin" import warnings (torch-scatter, etc.)
warnings.filterwarnings(
    "ignore",
    message=r"An issue occurred while importing 'torch-scatter'.*",
    category=UserWarning,
    module=r"torch_geometric.typing",
)
warnings.filterwarnings(
    "ignore",
    message=r"An issue occurred while importing 'torch-spline-conv'.*",
    category=UserWarning,
    module=r"torch_geometric.typing",
)
warnings.filterwarnings(
    "ignore",
    message=r"An issue occurred while importing 'torch-sparse'.*",
    category=UserWarning,
    module=r"torch_geometric.typing",
)
warnings.filterwarnings(
    "ignore",
    message=r"An issue occurred while importing 'torch-geometric'.*",
    category=UserWarning,
)


warnings.filterwarnings(
    "ignore",
    message=r".*'type_params' parameter of 'typing\._eval_type'.*",
    category=DeprecationWarning,
)

# ─────────────────────────────────────────────────────────────────────────

plt.rcParams["font.family"] = "Times New Roman"

logger = logging.getLogger("mqt-predictor")


def setup_device_predictor(
    devices: list[Target],
    figure_of_merit: figure_of_merit = "expected_fidelity",
    path_uncompiled_circuits: Path | None = None,
    path_compiled_circuits: Path | None = None,
    path_training_data: Path | None = None,
    timeout: int = 600,
    gnn: bool = False,
) -> bool:
    """Sets up the device predictor for the given figure of merit.

    Arguments:
        devices: The devices to be used for training.
        figure_of_merit: The figure of merit to be used for training. Defaults to "expected_fidelity".
        path_uncompiled_circuits: The path to the directory containing the circuits to be compiled. Defaults to None.
        path_compiled_circuits: The path to the directory where the compiled circuits should be saved. Defaults to None.
        path_training_data: The path to the directory where the generated training data should be saved. Defaults to None.
        timeout: The timeout in seconds for the compilation of a single circuit. Defaults to 600.
        gnn: Whether to use a GNN for training. Defaults to False.

    Returns:
        True if the setup was successful, False otherwise.
    """
    predictor = Predictor(figure_of_merit=figure_of_merit, devices=devices, gnn=gnn)
    try:
        logger.info(f"Start the training for the figure of merit: {figure_of_merit}")
        # Step 1: Generate compiled circuits for all devices
        predictor.compile_training_circuits(
            path_uncompiled_circuits=path_uncompiled_circuits,
            path_compiled_circuits=path_compiled_circuits,
            timeout=timeout,
        )
        logger.info(f"Generated compiled circuit for {figure_of_merit}")
        # Step 2: Generate training data from the compiled circuits
        predictor.generate_training_data(
            path_uncompiled_circuits=path_uncompiled_circuits,
            path_compiled_circuits=path_compiled_circuits,
            path_training_data=path_training_data,
        )
        logger.info(f"Generated training data for {figure_of_merit}")

        # Step 3: Train the random forest classifier
        if not predictor.gnn:
            predictor.train_random_forest_model()
            logger.info(f"Trained random forest classifier for {figure_of_merit}")
        else:
            predictor.train_gnn_model()
            logger.info(f"Trained random GNN for {figure_of_merit}")

    except FileNotFoundError:
        logger.exception("File not found during setup.")
        return False

    except TimeoutError:
        logger.exception("Timeout occurred during setup.")
        return False

    except Exception:
        logger.exception("An unexpected error occurred.")
        return False

    return True


class Predictor:
    """The Predictor class is used to predict the most suitable quantum device for a given quantum circuit."""

    def __init__(
        self,
        devices: list[Target],
        figure_of_merit: figure_of_merit = "expected_fidelity",
        gnn: bool = False,
        logger_level: int = logging.INFO,
    ) -> None:
        """Initializes the Predictor class.

        Arguments:
            figure_of_merit: The figure of merit to be used for training.
            devices: The devices to be used for training.
            logger_level: The level of the logger. Defaults to logging.INFO.
            gnn: Decide if using GNN or other models
        """
        logger.setLevel(logger_level)

        self.figure_of_merit = figure_of_merit
        self.devices = devices
        self.gnn = gnn
        self.devices.sort(
            key=lambda x: x.description
        )  # sorting is necessary to determine the ground truth label later on when generating the training data

    def _compile_all_circuits_devicewise(
        self,
        device: Target,
        timeout: int,
        path_uncompiled_circuits: Path | None = None,
        path_compiled_circuits: Path | None = None,
        logger_level: int = logging.INFO,
    ) -> None:
        """Compiles all circuits in the given directory with the given timeout and saves them in the given directory.

        Arguments:
            device: The device to be used for compilation.
            timeout: The timeout in seconds for the compilation of a single circuit.
            path_uncompiled_circuits: The path to the directory containing the circuits to be compiled. Defaults to None.
            path_compiled_circuits: The path to the directory where the compiled circuits should be saved. Defaults to None.
            logger_level: The level of the logger. Defaults to logging.INFO.

        Raises:
            RuntimeError: If an error occurs during compilation.
        """
        logger.setLevel(logger_level)

        logger.info("Processing: " + device.description + " for " + self.figure_of_merit)
        rl_pred = rl_Predictor(figure_of_merit=self.figure_of_merit, device=device)

        dev_max_qubits = device.num_qubits

        if path_uncompiled_circuits is None:
            path_uncompiled_circuits = get_path_training_circuits()

        if path_compiled_circuits is None:
            path_compiled_circuits = get_path_training_circuits_compiled()

        for filename in path_uncompiled_circuits.iterdir():
            if filename.suffix != ".qasm":
                continue
            qc = QuantumCircuit.from_qasm_file(filename)
            if qc.num_qubits > dev_max_qubits:
                continue

            target_filename = Path(filename).stem + "_" + self.figure_of_merit + "-" + device.description
            if (path_compiled_circuits / (target_filename + ".qasm")).exists():
                continue
            try:
                res = timeout_watcher(rl_compile, [qc, device, self.figure_of_merit, rl_pred], timeout)
                if isinstance(res, tuple):
                    compiled_qc = res[0]
                    with Path(path_compiled_circuits / (target_filename + ".qasm")).open("w", encoding="utf-8") as f:
                        dump(compiled_qc, f)

            except Exception as e:
                print(e, filename, device.description)
                raise RuntimeError("Error during compilation: " + str(e)) from e

    def compile_training_circuits(
        self,
        path_uncompiled_circuits: Path | None = None,
        path_compiled_circuits: Path | None = None,
        timeout: int = 600,
        num_workers: int = -1,
    ) -> None:
        """Compiles all circuits in the given directory with the given timeout and saves them in the given directory.

        Arguments:
            path_uncompiled_circuits: The path to the directory containing the circuits to be compiled. Defaults to None.
            path_compiled_circuits: The path to the directory where the compiled circuits should be saved. Defaults to None.
            timeout: The timeout in seconds for the compilation of a single circuit. Defaults to 600.
            num_workers: The number of workers to be used for parallelization. Defaults to -1.
        """
        if path_uncompiled_circuits is None:
            path_uncompiled_circuits = get_path_training_circuits()

        if path_compiled_circuits is None:
            path_compiled_circuits = get_path_training_circuits_compiled()

        path_zip = path_uncompiled_circuits / "training_data_device_selection.zip"
        if not any(file.suffix == ".qasm" for file in path_uncompiled_circuits.iterdir()) and path_zip.exists():
            with zipfile.ZipFile(str(path_zip), "r") as zip_ref:
                zip_ref.extractall(path_uncompiled_circuits)

        Parallel(n_jobs=num_workers, verbose=100)(
            delayed(self._compile_all_circuits_devicewise)(
                device, timeout, path_uncompiled_circuits, path_compiled_circuits, logger.level
            )
            for device in self.devices
        )

    def generate_training_data(
        self,
        path_uncompiled_circuits: Path | None = None,
        path_compiled_circuits: Path | None = None,
        path_training_data: Path | None = None,
        num_workers: int = -1,
    ) -> None:
        """Creates and saves training data from all generated training samples.

        Arguments:
            path_uncompiled_circuits: The path to the directory containing the uncompiled circuits. Defaults to None.
            path_compiled_circuits: The path to the directory containing the compiled circuits. Defaults to None.
            path_training_data: The path to the directory where the generated training data should be saved. Defaults to None.
            num_workers: The number of workers to be used for parallelization. Defaults to -1.

        Returns:
            The training data, consisting of training_data, name_list, scores_list

        """
        if not path_uncompiled_circuits:
            path_uncompiled_circuits = get_path_training_circuits()

        if not path_compiled_circuits:
            path_compiled_circuits = get_path_training_circuits_compiled()

        if not path_training_data:
            path_training_data = get_path_training_data() / "training_data_aggregated"

        # init resulting list (feature vector, name, scores)
        training_data = []
        names_list = []
        scores_list = []

        results = Parallel(n_jobs=num_workers, verbose=100)(
            delayed(self._generate_training_sample)(
                filename.name,
                path_uncompiled_circuits,
                path_compiled_circuits,
                logger.level,
            )
            for filename in path_uncompiled_circuits.glob("*.qasm")
        )
        if not self.gnn:
            for sample in results:
                training_sample, circuit_name, scores = sample
                if all(score == -1 for score in scores):
                    continue
                training_data.append(training_sample)
                names_list.append(circuit_name)
                scores_list.append(scores)

            with resources.as_file(path_training_data) as path:
                data = np.asarray(training_data, dtype=object)
                np.save(str(path / ("training_data_" + self.figure_of_merit + ".npy")), data)
                data = np.asarray(names_list, dtype=str)
                np.save(str(path / ("names_list_" + self.figure_of_merit + ".npy")), data)
                data = np.asarray(scores_list, dtype=object)
                np.save(str(path / ("scores_list_" + self.figure_of_merit + ".npy")), data)
        else:
            dataset = []
            for sample in results:
                if all(score == -1 for score in sample.scores_list):
                    continue
                dataset.append(sample)
            with resources.as_file(path_training_data) as path:
                torch.save(dataset, str(path / ("graph_dataset_" + self.figure_of_merit + ".pt")))

    def _generate_training_sample(
        self,
        file: Path,
        path_uncompiled_circuit: Path,
        path_compiled_circuits: Path,
        logger_level: int = logging.INFO,
    ) -> tuple[tuple[list[Any], Any], str, list[float]] | Data:
        """Handles to create a training sample from a given file.

        Arguments:
            file: The name of the file to be used for training.
            path_uncompiled_circuit: The path to the directory containing the uncompiled circuits. Defaults to None.
            path_compiled_circuits: The path to the directory containing the compiled circuits. Defaults to None.
            logger_level: The level of the logger. Defaults to logging.INFO.

        Returns:
            Training_sample, circuit_name, scores

        Raises:
            RuntimeError: If the file is not a qasm file or if no compiled circuits are found for the given file.
        """
        logger.setLevel(logger_level)

        if ".qasm" not in str(file):
            raise RuntimeError("File is not a qasm file: " + str(file))

        logger.debug("Checking " + str(file))
        scores = {dev.description: -1.0 for dev in self.devices}
        all_relevant_files = path_compiled_circuits.glob(str(file).split(".")[0] + "*")

        for filename in all_relevant_files:
            filename_str = str(filename)
            if (str(file).split(".")[0] + "_" + self.figure_of_merit) not in filename_str and filename_str.endswith(
                ".qasm"
            ):
                continue
            dev_name = filename_str.rsplit("-", maxsplit=1)[-1].split(".", maxsplit=1)[0]
            if dev_name not in [dev.description for dev in self.devices]:
                continue
            device = get_device(dev_name)
            qc = QuantumCircuit.from_qasm_file(filename_str)
            if self.figure_of_merit == "critical_depth":
                score = crit_depth(qc)
            elif self.figure_of_merit == "expected_fidelity":
                score = expected_fidelity(qc, device)
            elif self.figure_of_merit == "estimated_success_probability":
                score = estimated_success_probability(qc, device)
            elif self.figure_of_merit == "estimated_hellinger_distance":
                score = estimated_hellinger_distance(qc, device)
            elif self.figure_of_merit == "hellinger_distance":
                msg = "Hellinger distance should not be used for training data generation. Use 'estimated_hellinger_distance' instead."
                raise RuntimeError(msg)
            else:
                assert_never(self.figure_of_merit)
            scores[dev_name] = score

        num_not_empty_entries = 0
        for dev in self.devices:
            if scores[dev.description] != -1.0:
                num_not_empty_entries += 1

        if num_not_empty_entries == 0:
            logger.warning("no compiled circuits found for:" + str(file))

        scores_list = list(scores.values())
        target_label = max(scores, key=lambda k: scores[k])

        qc = QuantumCircuit.from_qasm_file(path_uncompiled_circuit / file)
        if not self.gnn:
            feature_vec = create_feature_vector(qc)
            training_sample = (feature_vec, target_label)
            circuit_name = str(file).split(".")[0]
            return training_sample, circuit_name, scores_list
        x, edge_index, number_of_gates = create_dag(qc)
        circuit_name = str(file).split(".")[0]
        self.devices_description = [dev.description for dev in self.devices]
        y = self.devices_description.index(target_label)
        print(target_label)
        return Data(
            x=x,
            y=torch.tensor([y], dtype=torch.float),
            circuit_name=circuit_name,
            edge_index=edge_index,
            target_label=target_label,  # torch.tensor([target_label], dtype=torch.float),
            scores_list=scores_list,
            num_nodes=number_of_gates,
        )

    def train_gnn_model(self, training_data: TrainingData | None = None) -> nn.Module:
        """Train the GNN model(s) and return the trained model."""
        # Figure out outputs and save path
        if self.figure_of_merit == "hellinger_distance":
            if len(self.devices) != 1:
                msg = "A single device must be provided for Hellinger distance model training."
                raise ValueError(msg)
            num_outputs = 1
            save_mdl_path = str(get_hellinger_model_path_gnn(self.devices[0]))
        else:
            num_outputs = max(1, len(self.devices))
            save_mdl_path = str(get_path_trained_model_gnn(self.figure_of_merit))

        # Prepare data
        if training_data is None:
            training_data = self._get_prepared_training_graphs()
        # number_in_features = int(len(get_openqasm_gates()) + 1 + 3 + 3)
        # Build model (ensure final layer outputs raw logits/no activation)
        if self.figure_of_merit != "hellinger_distance":
            model = GNN(
                in_feats=49,
                hidden_dim=128,
                num_resnet_layers=5,
                mlp_units=[1024, 128, 64],
                output_dim=num_outputs,
                classes=[dev.description for dev in self.devices],
            )
            json_dict = {
                "in_feats": 49,
                "hidden_dim": 128,
                "num_resnet_layers": 5,
                "mlp_units": [1024, 128, 64],
                "output_dim": num_outputs,
                "classes": [dev.description for dev in self.devices],
            }
        else:
            model = GNN(
                in_feats=49,
                hidden_dim=128,
                num_resnet_layers=5,
                mlp_units=[1024, 128, 64],
                output_dim=num_outputs,
            )

            # create a json with the characteristics of the model
            json_dict = {
                "in_feats": 49,
                "hidden_dim": 128,
                "num_resnet_layers": 5,
                "mlp_units": [1024, 128, 64],
                "output_dim": num_outputs,
            }

        json_path = Path(save_mdl_path).with_suffix(".json")  # works whether save_mdl_path is str or Path
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(json_dict, f, indent=4)

        # Device handling
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Dataloader
        train_loader = DataLoader(training_data.X_train, batch_size=64, shuffle=True)

        # Choose task + loss
        if self.figure_of_merit == "hellinger_distance":
            loss_fn = nn.MSELoss()
            task = "regression"
        else:
            if num_outputs == 1:
                loss_fn = nn.BCEWithLogitsLoss()
                task = "binary"
            else:
                loss_fn = nn.CrossEntropyLoss()
                task = "multiclass"

        # Train (your helpers should handle model.train(), moving batches to device, etc.)
        if task == "regression":
            train_regression_model(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                num_epochs=100,
                device=device,
            )
        elif task == "binary":
            train_classification_model(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                num_epochs=100,
                task="binary",
                device=device,
            )
        else:  # multiclass
            train_classification_model(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                num_epochs=100,
                task="multiclass",
                device=device,
            )
        # Save the model
        torch.save(model.state_dict(), save_mdl_path)
        return model

    def train_random_forest_model(
        self, training_data: TrainingData | None = None
    ) -> RandomForestRegressor | RandomForestClassifier:
        """Trains a random forest model for the given figure of merit.

        Arguments:
            training_data: The training data to be used for training the model. If None, the training data is loaded from the pre-prepared training data files.

        Returns:
            Either a trained RandomForestRegressor to estimate the Hellinger distance for a single device,
            or a trained RandomForestClassifier to score multiple devices according to a specific figure of merit.

        Raises:
            ValueError: If the figure of merit is 'hellinger_distance' and more than one device is provided.
        """
        tree_param = [
            {
                "n_estimators": [100, 200, 500],
                "max_depth": list(range(8, 30, 6)),
                "min_samples_split": list(range(2, 20, 6)),
                "min_samples_leaf": list(range(2, 20, 6)),
                "bootstrap": [True, False],
            },
        ]
        # Device-specific regression model for Hellinger distance
        if self.figure_of_merit == "hellinger_distance":
            if len(self.devices) != 1:
                msg = "A single device must be provided for Hellinger distance model training."
                raise ValueError(msg)

            mdl = RandomForestRegressor(random_state=0)
            save_mdl_path = str(get_hellinger_model_path(self.devices[0]))

        else:  # Default classification model to score all devices
            mdl = RandomForestClassifier(random_state=0)
            save_mdl_path = str(get_path_trained_model(self.figure_of_merit))

        if not training_data:
            training_data = self._get_prepared_training_data()
        num_cv = min(len(training_data.y_train), 5)
        mdl = GridSearchCV(mdl, tree_param, cv=num_cv, n_jobs=8).fit(training_data.X_train, training_data.y_train)

        joblib_dump(mdl, save_mdl_path)
        logger.info("Random Forest model is trained and saved.")

        return mdl.best_estimator_

    def _get_prepared_training_graphs(self) -> TrainingData:
        """Returns the training graphs for the given figure of merit."""
        with resources.as_file(get_path_training_data() / "training_data_aggregated") as path:
            prefix = f"{self.figure_of_merit}.pt"
            file_data = path / f"graph_dataset_{prefix}"

            if file_data.is_file():
                training_data = torch.load(file_data, weights_only=False)
            else:
                msg = "Training data not found."
                raise FileNotFoundError(msg)
        indices = np.arange(len(training_data), dtype=np.int64)
        score_list = [el.scores_list for el in training_data]
        names_list = [el.circuit_name for el in training_data]
        y = [el.target_label for el in training_data]
        # split data graph
        training_data, train_y, test_data, test_y, train_indices, test_indices = train_test_split(
            training_data, y, indices, test_size=0.3, random_state=5
        )

        return TrainingData(
            X_train=training_data,
            y_train=train_y,
            X_test=test_data,
            y_test=test_y,
            indices_train=train_indices,
            indices_test=test_indices,
            names_list=names_list,
            scores_list=score_list,
        )

    def _get_prepared_training_data(self) -> TrainingData:
        """Returns the training data for the given figure of merit.

        Raises:
            FileNotFoundError: If the training data files are not found.
        """
        with resources.as_file(get_path_training_data() / "training_data_aggregated") as path:
            prefix = f"{self.figure_of_merit}.npy"
            file_data = path / f"training_data_{prefix}"
            file_names = path / f"names_list_{prefix}"
            file_scores = path / f"scores_list_{prefix}"

            if file_data.is_file() and file_names.is_file() and file_scores.is_file():
                training_data = np.load(file_data, allow_pickle=True)
                names_list = list(np.load(file_names, allow_pickle=True))
                scores_list = [list(scores) for scores in np.load(file_scores, allow_pickle=True)]
            else:
                msg = "Training data not found."
                raise FileNotFoundError(msg)

        x_list, y_list = zip(*training_data, strict=False)
        x = np.array(x_list, dtype=np.float64)
        y = np.array(y_list, dtype=str)
        indices = np.arange(len(y), dtype=np.int64)

        x_train, x_test, y_train, y_test, indices_train, indices_test = train_test_split(
            x, y, indices, test_size=0.3, random_state=5
        )

        return TrainingData(
            X_train=x_train,
            y_train=y_train,
            X_test=x_test,
            y_test=y_test,
            indices_train=indices_train.tolist(),
            indices_test=indices_test.tolist(),
            names_list=names_list,
            scores_list=scores_list,
        )


def predict_device_for_figure_of_merit(
    qc: Path | QuantumCircuit, figure_of_merit: figure_of_merit = "expected_fidelity", gnn: bool = False
) -> Target:
    """Returns the probabilities for all supported quantum devices to be the most suitable one for the given quantum circuit.

    Arguments:
        qc: The QuantumCircuit or Path to the respective qasm file.
        figure_of_merit: The figure of merit to be used for compilation.
        gnn: Whether to use a GNN for prediction. Defaults to False.

    Returns:
        The probabilities for all supported quantum devices to be the most suitable one for the given quantum circuit.

    Raises:
        FileNotFoundError: If the ML model is not trained yet.
        ValueError: If no suitable device is found for the given quantum circuit.
    """
    if isinstance(qc, Path) and qc.exists():
        qc = QuantumCircuit.from_qasm_file(qc)
    assert isinstance(qc, QuantumCircuit)
    path = get_path_trained_model(figure_of_merit) if not gnn else get_path_trained_model_gnn(figure_of_merit)
    if not path.exists():
        error_msg = "The ML model is not trained yet. Please train the model before using it."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    if not gnn:
        clf = load(path)

        feature_vector = create_feature_vector(qc)

        probabilities = clf.predict_proba([feature_vector])[0]
        class_labels = clf.classes_
        # sort all devices with decreasing probabilities
        sorted_devices = np.array([
            label for _, label in sorted(zip(probabilities, class_labels, strict=False), reverse=True)
        ])
    else:
        # Open the json file save_mdl_path[:-4] + ".json"
        with Path.open(path.with_suffix(".json"), encoding="utf-8") as f:
            json_dict = json.load(f)

        gnn_model = GNN(
            in_feats=json_dict["in_feats"],
            hidden_dim=json_dict["hidden_dim"],
            num_resnet_layers=json_dict["num_resnet_layers"],
            mlp_units=json_dict["mlp_units"],
            output_dim=json_dict["output_dim"],
            classes=json_dict["classes"],
        )
        gnn_model.load_state_dict(torch.load(path))
        x, edge_index, number_of_gates = create_dag(qc)
        feature_vector = Data(x=x, edge_index=edge_index, num_gates=number_of_gates)
        gnn_model.eval()
        class_labels = gnn_model.classes
        with torch.no_grad():
            probabilities = torch.softmax(gnn_model(feature_vector), dim=1)
        assert class_labels is not None
        if len(class_labels) != len(probabilities):
            msg = "probabilities and class_labels must be same length"
            raise ValueError(msg)

        pairs = sorted(zip(probabilities.tolist(), class_labels, strict=False), reverse=True)
        sorted_devices = np.array([label for _, label in pairs])

    for dev_name in sorted_devices:
        dev = get_device(dev_name)
        if dev.num_qubits >= qc.num_qubits:
            return dev
    msg = f"No suitable device found for the given quantum circuit with {qc.num_qubits} qubits."
    raise ValueError(msg)
