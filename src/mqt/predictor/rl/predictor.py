# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""This module contains the Predictor class, which is used to predict the most suitable compilation pass sequence for a given quantum circuit."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.utils import set_random_seed

from mqt.predictor.rl.helper import get_path_trained_model, logger
from mqt.predictor.rl.predictorenv import PredictorEnv

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.transpiler import Target

    from mqt.predictor.reward import figure_of_merit


class Predictor:
    """The Predictor class is used to train a reinforcement learning model for a given figure of merit and device such that it acts as a compiler."""

    def __init__(
        self,
        figure_of_merit: figure_of_merit,
        device: Target,
        path_training_circuits: Path | None = None,
        logger_level: int = logging.INFO,
    ) -> None:
        """Initializes the Predictor object."""
        logger.setLevel(logger_level)

        self.env = PredictorEnv(
            reward_function=figure_of_merit, device=device, path_training_circuits=path_training_circuits
        )
        self.device_name = device.description
        self.figure_of_merit = figure_of_merit

    def compile_as_predicted(
        self,
        qc: QuantumCircuit,
    ) -> tuple[QuantumCircuit, list[str]]:
        """Compiles a given quantum circuit such that the given figure of merit is maximized by using the respectively trained optimized compiler.

        Arguments:
            qc: The quantum circuit to be compiled or the path to a qasm file containing the quantum circuit.

        Returns:
            A tuple containing the compiled quantum circuit and the compilation information. If compilation fails, False is returned.
        """
        trained_rl_model = load_model("model_" + self.figure_of_merit + "_" + self.device_name)

        obs, _ = self.env.reset(qc, seed=0)

        used_compilation_passes = []
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action_masks = get_action_masks(self.env)
            action, _ = trained_rl_model.predict(obs, action_masks=action_masks)
            action = int(action)
            action_item = self.env.action_set[action]
            used_compilation_passes.append(action_item.name)
            obs, _reward_val, terminated, truncated, _info = self.env.step(action)

        if not self.env.error_occurred:
            return self.env.state, used_compilation_passes

        msg = "Error occurred during compilation."
        raise RuntimeError(msg)

    def train_model(
        self,
        timesteps: int = 1000,
        model_name: str = "model",
        verbose: int = 2,
        test: bool = False,
    ) -> None:
        """Trains all models for the given reward functions and device.

        Arguments:
            timesteps: The number of timesteps to train the model. Defaults to 1000.
            model_name: The name of the model. Defaults to "model".
            verbose: The verbosity level. Defaults to 2.
            test: Whether to train the model for testing purposes. Defaults to False.
        """
        if test:
            set_random_seed(0)  # for reproducibility
            n_steps = 10
            n_epochs = 1
            batch_size = 10
            progress_bar = False
        else:
            # default PPO values
            n_steps = 2048
            n_epochs = 10
            batch_size = 64
            progress_bar = True

        logger.debug("Start training for: " + self.figure_of_merit + " on " + self.device_name)
        model = MaskablePPO(
            MaskableMultiInputActorCriticPolicy,
            self.env,
            verbose=verbose,
            tensorboard_log="./" + model_name + "_" + self.figure_of_merit + "_" + self.device_name,
            gamma=0.98,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
        )
        # Training Loop: In each iteration, the agent collects n_steps steps (rollout),
        # updates the policy for n_epochs, and then repeats the process until total_timesteps steps have been taken.
        model.learn(total_timesteps=timesteps, progress_bar=progress_bar)
        model.save(get_path_trained_model() / (model_name + "_" + self.figure_of_merit + "_" + self.device_name))


def load_model(model_name: str) -> MaskablePPO:
    """Loads a trained model from the trained model folder.

    Arguments:
        model_name: The name of the model to be loaded.

    Returns:
        The loaded model.
    """
    path = get_path_trained_model()
    if Path(path / (model_name + ".zip")).is_file():
        return MaskablePPO.load(path / (model_name + ".zip"))

    error_msg = f"The RL model '{model_name}' is not trained yet. Please train the model before using it."
    logger.error(error_msg)
    raise FileNotFoundError(error_msg)


def rl_compile(
    qc: QuantumCircuit | str,
    device: Target | None,
    figure_of_merit: figure_of_merit | None = "expected_fidelity",
    predictor_singleton: Predictor | None = None,
) -> tuple[QuantumCircuit, list[str]]:
    """Compiles a given quantum circuit to a device optimizing for the given figure of merit.

    Arguments:
        qc: The quantum circuit to be compiled. If a string is given, it is assumed to be a path to a qasm file.
        device: The device to compile to.
        figure_of_merit: The figure of merit to be used for compilation. Defaults to "expected_fidelity".
        predictor_singleton: A predictor object that is used for compilation to reduce compilation time when compiling multiple quantum circuits. If None, a new predictor object is created. Defaults to None.

    Returns:
        A tuple containing the compiled quantum circuit and the compilation information. If compilation fails, False is returned.
    """
    if predictor_singleton is None:
        if figure_of_merit is None:
            msg = "figure_of_merit must not be None if predictor_singleton is None."
            raise ValueError(msg)
        if device is None:
            msg = "device must not be None if predictor_singleton is None."
            raise ValueError(msg)
        predictor = Predictor(figure_of_merit=figure_of_merit, device=device)
    else:
        predictor = predictor_singleton

    return predictor.compile_as_predicted(qc)
