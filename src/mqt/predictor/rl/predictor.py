# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
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
        max_steps: int | None = 100,
        tracer_output_path: str | Path | None = None,
    ) -> None:
        """Initializes the Predictor object.

        Arguments:
            figure_of_merit: The figure of merit to optimize during compilation.
            device: The target device to compile to.
            path_training_circuits: The path to the training circuits folder. Defaults to None.
            logger_level: The logger level. Defaults to logging.INFO.
            max_steps: The maximum number of actions per episode. If None, no step limit is enforced. Defaults to 100.
            tracer_output_path: Optional path to export the compilation trace.
        """
        logger.setLevel(logger_level)

        self.env = PredictorEnv(
            reward_function=figure_of_merit,
            device=device,
            path_training_circuits=path_training_circuits,
            max_steps=max_steps,
            tracer_output_path=tracer_output_path,
        )
        self.device_name = device.description
        self.figure_of_merit = figure_of_merit

    def compile_as_predicted(
        self,
        qc: QuantumCircuit | str,
        tracer_output_path: str | Path | None = None,
    ) -> tuple[QuantumCircuit, list[str]]:
        """Compiles a given quantum circuit such that the given figure of merit is maximized by using the respectively trained optimized compiler.

        Arguments:
            qc: The quantum circuit to be compiled or the path to a qasm file containing the quantum circuit.
            tracer_output_path: Optional temporary path to export the compilation trace for this specific run.

        Returns:
            A tuple containing the compiled quantum circuit and the compilation information. If compilation fails, False is returned.

        Raises:
            RuntimeError: If an error occurs during compilation.
        """
        original_tracer_output_path = self.env.tracer_output_path

        # Temporarily override singleton if a new path is explicitly provided
        if tracer_output_path is not None:
            self.env.tracer_output_path = tracer_output_path

        try:
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

        finally:
            # Restore original singleton path
            self.env.tracer_output_path = original_tracer_output_path

    def train_model(
        self,
        timesteps: int = 1000,
        verbose: int = 2,
        test: bool = False,
        seed: int | None = None,
    ) -> None:
        """Trains all models for the given reward functions and device.

        Arguments:
            timesteps: The number of timesteps to train the model. Defaults to 1000.
            verbose: The verbosity level. Defaults to 2.
            test: Whether to train the model for testing purposes. Defaults to False.
            seed: The random seed to use for reproducible training. Set to None to use true randomness.
                Defaults to None.
        """
        if seed is not None:
            set_random_seed(seed)
        if test:
            # minimum training overhead
            n_steps = max(timesteps, 2)
            n_epochs = 1
            batch_size = n_steps
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
            tensorboard_log="./model_" + self.figure_of_merit + "_" + self.device_name,
            gamma=0.98,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            seed=seed,
        )
        # Training Loop: In each iteration, the agent collects n_steps steps (rollout),
        # updates the policy for n_epochs, and then repeats the process until total_timesteps steps have been taken.
        model.learn(total_timesteps=timesteps, progress_bar=progress_bar)
        model.save(get_path_trained_model() / ("model_" + self.figure_of_merit + "_" + self.device_name))


def load_model(model_name: str) -> MaskablePPO:
    """Loads a trained model from the trained model folder.

    Arguments:
        model_name: The name of the model to be loaded.

    Returns:
        The loaded model.

    Raises:
        FileNotFoundError: If the model file does not exist.
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
    tracer_output_path: str | Path | None = None,
) -> tuple[QuantumCircuit, list[str]]:
    """Compiles a given quantum circuit to a device optimizing for the given figure of merit.

    Arguments:
        qc: The quantum circuit to be compiled. If a string is given, it is assumed to be a path to a qasm file.
        device: The device to compile to.
        figure_of_merit: The figure of merit to be used for compilation. Defaults to "expected_fidelity".
        predictor_singleton: A predictor object that is used for compilation to reduce compilation time when compiling multiple quantum circuits. If None, a new predictor object is created. Defaults to None.
        tracer_output_path: If provided, enables compiler tracing and exports the JSON log to the specified path.

    Returns:
        A tuple containing the compiled quantum circuit and the compilation information. If compilation fails, False is returned.

    Raises:
        ValueError: If figure_of_merit or device is None and predictor_singleton is also None.
    """
    if predictor_singleton is None:
        if figure_of_merit is None:
            msg = "figure_of_merit must not be None if predictor_singleton is None."
            raise ValueError(msg)
        if device is None:
            msg = "device must not be None if predictor_singleton is None."
            raise ValueError(msg)
        predictor = Predictor(figure_of_merit=figure_of_merit, device=device, tracer_output_path=tracer_output_path)
        return predictor.compile_as_predicted(qc)

    return predictor_singleton.compile_as_predicted(qc, tracer_output_path=tracer_output_path)
