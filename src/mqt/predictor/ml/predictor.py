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

if sys.version_info >= (3, 11) and TYPE_CHECKING:  # pragma: no cover
    from typing import assert_never
else:
    from typing_extensions import assert_never

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed, load
from mqt.bench.targets import get_available_device_names, get_device
from qiskit import QuantumCircuit
from qiskit.qasm2 import dump
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

from mqt.predictor import ml, reward, rl, utils
from mqt.predictor.hellinger import get_hellinger_model_path

if TYPE_CHECKING:
    from numpy._typing import NDArray
    from qiskit.transpiler import Target

plt.rcParams["font.family"] = "Times New Roman"

logger = logging.getLogger("mqt-predictor")


class Predictor:
    """The Predictor class is used to predict the most suitable quantum device for a given quantum circuit."""

    def __init__(
        self,
        figure_of_merit: reward.figure_of_merit = "expected_fidelity",
        devices: list[Target] | None = None,
        logger_level: int = logging.INFO,
    ) -> None:
        """Initializes the Predictor class.

        Arguments:
            figure_of_merit: The figure of merit to be used for training.
            devices: The devices to be used for training. Defaults to None. If None, all available devices from MQT Bench are used.
            logger_level: The level of the logger. Defaults to logging.INFO.

        """
        logger.setLevel(logger_level)

        self.clf = None
        self.figure_of_merit = figure_of_merit
        if devices is None:
            self.devices = get_available_device_names()
        else:
            self.devices = devices
        self.devices.sort(
            key=lambda x: x.description
        )  # sorting is necessary to determine the ground truth label later on when generating the training data

    def set_classifier(self, clf: RandomForestClassifier) -> None:
        """Sets the classifier to the given classifier."""
        self.clf = clf

    def compile_all_circuits_devicewise(
        self,
        device: Target,
        timeout: int,
        source_path: Path | None = None,
        target_path: Path | None = None,
        logger_level: int = logging.INFO,
    ) -> None:
        """Compiles all circuits in the given directory with the given timeout and saves them in the given directory.

        Arguments:
            device: The device to be used for compilation.
            timeout: The timeout in seconds for the compilation of a single circuit.
            source_path: The path to the directory containing the circuits to be compiled. Defaults to None.
            target_path: The path to the directory where the compiled circuits should be saved. Defaults to None.
            logger_level: The level of the logger. Defaults to logging.INFO.
        """
        logger.setLevel(logger_level)

        logger.info("Processing: " + device.description + " for " + self.figure_of_merit)
        rl_pred = rl.Predictor(figure_of_merit=self.figure_of_merit, device=device)

        dev_max_qubits = device.num_qubits

        if source_path is None:
            source_path = ml.helper.get_path_training_circuits()

        if target_path is None:
            target_path = ml.helper.get_path_training_circuits_compiled()

        for filename in source_path.iterdir():
            if filename.suffix != ".qasm":
                continue
            qc = QuantumCircuit.from_qasm_file(filename)
            if qc.num_qubits > dev_max_qubits:
                continue

            target_filename = Path(filename).stem + "_" + self.figure_of_merit + "-" + device.description
            if (target_path / (target_filename + ".qasm")).exists():
                continue
            try:
                res = utils.timeout_watcher(
                    rl.qcompile, [qc, self.figure_of_merit, device.description, rl_pred], timeout
                )
                if isinstance(res, tuple):
                    compiled_qc = res[0]
                    with Path(target_path / (target_filename + ".qasm")).open("w", encoding="utf-8") as f:
                        dump(compiled_qc, f)

            except Exception as e:
                print(e, filename, device.description)
                raise RuntimeError("Error during compilation: " + str(e)) from e

    def generate_compiled_circuits(
        self,
        source_path: Path | None = None,
        target_path: Path | None = None,
        timeout: int = 600,
        num_workers: int = -1,
    ) -> None:
        """Compiles all circuits in the given directory with the given timeout and saves them in the given directory.

        Arguments:
            source_path: The path to the directory containing the circuits to be compiled. Defaults to None.
            target_path: The path to the directory where the compiled circuits should be saved. Defaults to None.
            timeout: The timeout in seconds for the compilation of a single circuit. Defaults to 600.
            num_workers: The number of workers to be used for parallelization. Defaults to -1.
        """
        if source_path is None:
            source_path = ml.helper.get_path_training_circuits()

        if target_path is None:
            target_path = ml.helper.get_path_training_circuits_compiled()

        path_zip = source_path / "training_data_device_selection.zip"
        if not any(file.suffix == ".qasm" for file in source_path.iterdir()) and path_zip.exists():
            with zipfile.ZipFile(str(path_zip), "r") as zip_ref:
                zip_ref.extractall(source_path)

        Parallel(n_jobs=num_workers, verbose=100)(
            delayed(self.compile_all_circuits_devicewise)(device, timeout, source_path, target_path, logger.level)
            for device in self.devices
        )

    def generate_trainingdata_from_qasm_files(
        self,
        path_uncompiled_circuits: Path | None = None,
        path_compiled_circuits: Path | None = None,
        num_workers: int = -1,
    ) -> tuple[list[NDArray[np.float64]], list[str], list[NDArray[np.float64]]]:
        """Handles to create training data from all generated training samples.

        Arguments:
            path_uncompiled_circuits: The path to the directory containing the uncompiled circuits. Defaults to None.
            path_compiled_circuits: The path to the directory containing the compiled circuits. Defaults to None.
            num_workers: The number of workers to be used for parallelization. Defaults to -1.

        Returns:
            The training data, consisting of training_data, name_list, scores_list

        """
        if not path_uncompiled_circuits:
            path_uncompiled_circuits = ml.helper.get_path_training_circuits()

        if not path_compiled_circuits:
            path_compiled_circuits = ml.helper.get_path_training_circuits_compiled()

        # init resulting list (feature vector, name, scores)
        training_data = []
        name_list = []
        scores_list = []

        results = Parallel(n_jobs=num_workers, verbose=100)(
            delayed(self.generate_training_sample)(
                filename.name,
                path_uncompiled_circuits,
                path_compiled_circuits,
                logger.level,
            )
            for filename in path_uncompiled_circuits.glob("*.qasm")
        )
        for sample in results:
            training_sample, circuit_name, scores = sample
            if all(score == -1 for score in scores):
                continue
            training_data.append(training_sample)
            name_list.append(circuit_name)
            scores_list.append(scores)

        return (training_data, name_list, scores_list)

    def generate_training_sample(
        self,
        file: Path,
        path_uncompiled_circuit: Path,
        path_compiled_circuits: Path,
        logger_level: int = logging.INFO,
    ) -> tuple[tuple[list[Any], Any], str, list[float]]:
        """Handles to create a training sample from a given file.

        Arguments:
            file: The name of the file to be used for training.
            path_uncompiled_circuit: The path to the directory containing the uncompiled circuits. Defaults to None.
            path_compiled_circuits: The path to the directory containing the compiled circuits. Defaults to None.
            logger_level: The level of the logger. Defaults to logging.INFO.

        Returns:
            Training_sample, circuit_name, scores
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
                score = reward.crit_depth(qc)
            elif self.figure_of_merit == "expected_fidelity":
                score = reward.expected_fidelity(qc, device)
            elif self.figure_of_merit == "estimated_success_probability":
                score = reward.estimated_success_probability(qc, device)
            elif self.figure_of_merit == "estimated_hellinger_distance":
                score = reward.estimated_hellinger_distance(qc, device)
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
        # target label is the dict key.name with the highest value, dont use max(scores, key=scores.get).name
        target_label = max(scores, key=lambda k: scores[k])

        qc = QuantumCircuit.from_qasm_file(path_uncompiled_circuit / file)
        feature_vec = ml.helper.create_feature_vector(qc)
        training_sample = (feature_vec, target_label)
        circuit_name = str(file).split(".")[0]
        return training_sample, circuit_name, scores_list

    def train_random_forest_classifier(
        self,
        save_classifier: bool = True,
    ) -> bool:
        """Trains a random forest classifier for the given figure of merit.

        Arguments:
            save_classifier: Whether to save the classifier. Defaults to True.

        Returns:
            True when the training was successful, False otherwise.
        """
        training_data = self.get_prepared_training_data()
        clf = self.train_random_forest_model(training_data, None, self.figure_of_merit, save_classifier)
        self.set_classifier(clf)
        return clf is not None

    @staticmethod
    def train_random_forest_model(
        training_data: ml.helper.TrainingData,
        device: Target | None,
        figure_of_merit: str | reward.figure_of_merit,
        save_model: bool = True,
    ) -> RandomForestRegressor | RandomForestClassifier:
        """Trains a random forest model for the given figure of merit.

        Arguments:
            training_data: The training data, the names list and the scores list to be saved.
            device: The device to be used for training.
            figure_of_merit: The figure of merit to be used for training.
            save_model: Whether to save the classifier. Defaults to True.

        Returns:
            Either a trained RandomForestRegressor to estimate the Hellinger distance for a single device,
            or a trained RandomForestClassifier to score multiple devices according to a specific figure of merit.
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
        if figure_of_merit == "hellinger_distance":
            if device is None:
                msg = "A device must be provided for Hellinger distance model training."
                raise ValueError(msg)

            mdl = RandomForestRegressor(random_state=0)
            save_mdl_path = str(get_hellinger_model_path(device))

        else:  # Default classification model to score all devices
            mdl = RandomForestClassifier(random_state=0)
            save_mdl_path = str(ml.helper.get_path_trained_model(figure_of_merit))

        num_cv = min(len(training_data.y_train), 5)
        mdl = GridSearchCV(mdl, tree_param, cv=num_cv, n_jobs=8).fit(training_data.X_train, training_data.y_train)

        if save_model:
            joblib_dump(mdl, save_mdl_path)
        logger.info("Random Forest model is trained and saved.")

        return mdl.best_estimator_

    def get_prepared_training_data(self) -> ml.helper.TrainingData:
        """Returns the training data for the given figure of merit."""
        training_data, names_list, raw_scores_list = self.load_training_data()
        unzipped_training_data_x, unzipped_training_data_y = zip(*training_data, strict=False)
        scores_list: list[list[float]] = [[] for _ in range(len(raw_scores_list))]
        x_raw = list(unzipped_training_data_x)
        x_list: list[list[float]] = [[] for _ in range(len(x_raw))]
        y_list = list(unzipped_training_data_y)
        for i in range(len(x_raw)):
            x_list[i] = list(x_raw[i])
            scores_list[i] = list(raw_scores_list[i])

        x, y, indices = (
            np.array(x_list, dtype=np.float64),
            np.array(y_list, dtype=str),
            np.array(range(len(y_list)), dtype=np.int64),
        )

        (
            x_train,
            x_test,
            y_train,
            y_test,
            indices_train,
            indices_test,
        ) = train_test_split(x, y, indices, test_size=0.3, random_state=5)

        return ml.helper.TrainingData(
            x_train,
            y_train,
            x_test,
            y_test,
            indices_train,
            indices_test,
            names_list,
            scores_list,
        )

    def save_training_data(
        self,
        training_data: list[NDArray[np.float64]],
        names_list: list[str],
        scores_list: list[NDArray[np.float64]],
    ) -> None:
        """Saves the given training data to the training data folder.

        Arguments:
            training_data: The training data, the names list and the scores list to be saved.
            names_list: The names list of the training data.
            scores_list: The scores list of the training data.
            figure_of_merit: The figure of merit to be used for compilation.
        """
        with resources.as_file(ml.helper.get_path_training_data() / "training_data_aggregated") as path:
            data = np.asarray(training_data, dtype=object)
            np.save(str(path / ("training_data_" + self.figure_of_merit + ".npy")), data)
            data = np.asarray(names_list, dtype=str)
            np.save(str(path / ("names_list_" + self.figure_of_merit + ".npy")), data)
            data = np.asarray(scores_list, dtype=object)
            np.save(str(path / ("scores_list_" + self.figure_of_merit + ".npy")), data)

    def load_training_data(self) -> tuple[list[NDArray[np.float64]], list[str], list[NDArray[np.float64]]]:
        """Loads and returns the training data from the training data folder."""
        with resources.as_file(ml.helper.get_path_training_data() / "training_data_aggregated") as path:
            if (
                path.joinpath("training_data_" + self.figure_of_merit + ".npy").is_file()
                and path.joinpath("names_list_" + self.figure_of_merit + ".npy").is_file()
                and path.joinpath("scores_list_" + self.figure_of_merit + ".npy").is_file()
            ):
                training_data = np.load(path / ("training_data_" + self.figure_of_merit + ".npy"), allow_pickle=True)
                names_list = list(np.load(path / ("names_list_" + self.figure_of_merit + ".npy"), allow_pickle=True))
                scores_list = list(np.load(path / ("scores_list_" + self.figure_of_merit + ".npy"), allow_pickle=True))
            else:
                error_msg = "Training data not found. Please run the training script first as described in the documentation that can be found at https://mqt.readthedocs.io/projects/predictor/en/latest/Usage.html."
                raise FileNotFoundError(error_msg)

            return training_data, names_list, scores_list


def predict_device_for_figure_of_merit(
    qc: Path | QuantumCircuit, figure_of_merit: reward.figure_of_merit = "expected_fidelity"
) -> Target:
    """Returns the probabilities for all supported quantum devices to be the most suitable one for the given quantum circuit.

    Arguments:
        qc: The QuantumCircuit or Path to the respective qasm file.
        figure_of_merit: The figure of merit to be used for compilation.

    Returns:
        The probabilities for all supported quantum devices to be the most suitable one for the given quantum circuit.
    """
    if isinstance(qc, Path) and qc.exists():
        qc = QuantumCircuit.from_qasm_file(qc)
    assert isinstance(qc, QuantumCircuit)

    path = ml.helper.get_path_trained_model(figure_of_merit)
    if not path.exists():
        error_msg = "The ML model is not trained yet. Please train the model before using it."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    clf = load(path)

    feature_vector = ml.helper.create_feature_vector(qc)

    probabilities = clf.predict_proba([feature_vector])[0]
    class_labels = clf.classes_
    # sort all devices with decreasing probabilities
    sorted_devices = np.array([
        label for _, label in sorted(zip(probabilities, class_labels, strict=False), reverse=True)
    ])

    for dev_name in sorted_devices:
        dev = get_device(dev_name)
        if dev.num_qubits >= qc.num_qubits:
            return dev
    msg = f"No suitable device found for the given quantum circuit with {qc.num_qubits} qubits."
    raise ValueError(msg)


def train_random_forest_regressor(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    device: Target | None,
    save_model: bool = True,
) -> RandomForestRegressor:
    """Trains a random forest regressor on a Hellinger distance dataset.

    Arguments:
        x_train: The training data (circuit feature vectors).
        y_train: The training labels (Hellinger distance values).
        device: The device to be used for training.
        save_model: Whether to save the trained model. Defaults to True.

    Returns:
        Either a trained RandomForestRegressor to estimate the Hellinger distance for a single device,
        or a trained RandomForestClassifier to score multiple devices according to a specific figure of merit.
    """
    # type cast the data to the expected format
    train_data = ml.helper.TrainingData(X_train=x_train, y_train=y_train)

    return Predictor.train_random_forest_model(
        train_data, device, figure_of_merit="hellinger_distance", save_model=save_model
    )
