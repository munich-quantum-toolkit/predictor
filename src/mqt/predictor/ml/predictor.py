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
from mqt.bench.targets import get_device
from qiskit import QuantumCircuit
from qiskit.qasm2 import dump
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

from mqt.predictor.hellinger import get_hellinger_model_path
from mqt.predictor.ml.helper import (
    TrainingData,
    create_feature_vector,
    get_path_trained_model,
    get_path_training_circuits,
    get_path_training_circuits_compiled,
    get_path_training_data,
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

plt.rcParams["font.family"] = "Times New Roman"

logger = logging.getLogger("mqt-predictor")


def setup_device_predictor(
    devices: list[Target],
    figure_of_merit: figure_of_merit = "expected_fidelity",
    path_uncompiled_circuits: Path | None = None,
    path_compiled_circuits: Path | None = None,
) -> bool:
    """Sets up the device predictor for the given figure of merit."""
    predictor = Predictor(
        figure_of_merit=figure_of_merit,
        devices=devices,
    )
    try:
        logger.info(f"Start the training for the figure of merit: {figure_of_merit}")
        # Step 1: Generate compiled circuits for all devices
        predictor.compile_training_circuits(
            path_uncompiled_circuits=path_uncompiled_circuits,
            path_compiled_circuits=path_compiled_circuits,
            timeout=600,
        )
        logger.info(f"Generated compiled circuit for {figure_of_merit}")
        # Step 2: Generate training data from the compiled circuits
        predictor.generate_training_data(
            path_uncompiled_circuits=path_uncompiled_circuits, path_compiled_circuits=path_compiled_circuits
        )
        logger.info(f"Generated training data for {figure_of_merit}")
        # Step 3: Train the random forest classifier
        predictor.train_random_forest_model()
        logger.info(f"Trained random forest classifier for {figure_of_merit}")

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
        logger_level: int = logging.INFO,
    ) -> None:
        """Initializes the Predictor class.

        Arguments:
            figure_of_merit: The figure of merit to be used for training.
            devices: The devices to be used for training.
            logger_level: The level of the logger. Defaults to logging.INFO.

        """
        logger.setLevel(logger_level)

        self.figure_of_merit = figure_of_merit
        self.devices = devices
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
        num_workers: int = -1,
    ) -> None:
        """Creates and saves training data from all generated training samples.

        Arguments:
            path_uncompiled_circuits: The path to the directory containing the uncompiled circuits. Defaults to None.
            path_compiled_circuits: The path to the directory containing the compiled circuits. Defaults to None.
            num_workers: The number of workers to be used for parallelization. Defaults to -1.

        Returns:
            The training data, consisting of training_data, name_list, scores_list

        """
        if not path_uncompiled_circuits:
            path_uncompiled_circuits = get_path_training_circuits()

        if not path_compiled_circuits:
            path_compiled_circuits = get_path_training_circuits_compiled()

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
        for sample in results:
            training_sample, circuit_name, scores = sample
            if all(score == -1 for score in scores):
                continue
            training_data.append(training_sample)
            names_list.append(circuit_name)
            scores_list.append(scores)

        with resources.as_file(get_path_training_data() / "training_data_aggregated") as path:
            data = np.asarray(training_data, dtype=object)
            np.save(str(path / ("training_data_" + self.figure_of_merit + ".npy")), data)
            data = np.asarray(names_list, dtype=str)
            np.save(str(path / ("names_list_" + self.figure_of_merit + ".npy")), data)
            data = np.asarray(scores_list, dtype=object)
            np.save(str(path / ("scores_list_" + self.figure_of_merit + ".npy")), data)

    def _generate_training_sample(
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
        feature_vec = create_feature_vector(qc)
        training_sample = (feature_vec, target_label)
        circuit_name = str(file).split(".")[0]
        return training_sample, circuit_name, scores_list

    def train_random_forest_model(
        self, training_data: TrainingData | None = None
    ) -> RandomForestRegressor | RandomForestClassifier:
        """Trains a random forest model for the given figure of merit.

        Arguments:
            training_data: The training data to be used for training the model. If None, the training data is loaded from the pre-prepared training data files.

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

    def _get_prepared_training_data(self) -> TrainingData:
        """Returns the training data for the given figure of merit."""
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
    qc: Path | QuantumCircuit, figure_of_merit: figure_of_merit = "expected_fidelity"
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

    path = get_path_trained_model(figure_of_merit)
    if not path.exists():
        error_msg = "The ML model is not trained yet. Please train the model before using it."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    clf = load(path)

    feature_vector = create_feature_vector(qc)

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
