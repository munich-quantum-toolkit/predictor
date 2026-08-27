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
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.utils import set_random_seed

from mqt.predictor.rl.helper import get_path_trained_model, logger
from mqt.predictor.rl.predictorenv import MDPPolicy, PredictorEnv

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.transpiler import Target

    from mqt.predictor.reward import figure_of_merit
    from mqt.predictor.rl.gnn import GNNConfig, GNNObservationWrapper


class Predictor:
    """The Predictor class is used to train a reinforcement learning model for a given figure of merit and device such that it acts as a compiler."""

    def __init__(
        self,
        figure_of_merit: figure_of_merit,
        device: Target,
        path_training_circuits: Path | None = None,
        logger_level: int = logging.INFO,
        max_steps: int | None = None,
        tracer_output_path: str | Path | None = None,
        mdp: MDPPolicy = "v3",
        graph: bool = False,
        gnn_config: GNNConfig | None = None,
    ) -> None:
        """Initializes the Predictor object.

        Arguments:
            figure_of_merit: The figure of merit to optimize during compilation.
            device: The target device to compile to.
            path_training_circuits: The path to the training circuits folder. Defaults to None.
            logger_level: The logger level. Defaults to logging.INFO.
            max_steps: The maximum number of actions per episode. Defaults to None, which means no step limit is enforced.
            tracer_output_path: Path to export the compilation trace JSON. Defaults to None.
            mdp: The MDP transition policy. ``v2`` is the original strategy and
                ``v3`` is the default.
            graph: Whether to use the opt-in GNN policy. Defaults to False.
            gnn_config: Configuration for the GNN policy. Defaults to the prototype training configuration.
        """
        logger.setLevel(logger_level)

        self.env = PredictorEnv(
            reward_function=figure_of_merit,
            device=device,
            path_training_circuits=path_training_circuits,
            max_steps=max_steps,
            tracer_output_path=tracer_output_path,
            mdp=mdp,
        )
        self.device_name = device.description
        self.figure_of_merit = figure_of_merit
        self.graph = graph
        if graph:
            from mqt.predictor.rl.gnn import GNNConfig  # ruff: ignore[import-outside-top-level]

            self.gnn_config = gnn_config or GNNConfig()
        else:
            self.gnn_config = gnn_config
        model_prefix = "gnn" if graph else "model"
        self.model_name = f"{model_prefix}_{self.figure_of_merit}_{self.device_name}_{mdp}"

    def compile_as_predicted(
        self,
        qc: QuantumCircuit | str,
        tracer_output_path: str | Path | None = None,
        pass_timeout: float | None = None,
    ) -> tuple[QuantumCircuit, list[str]]:
        """Compiles a given quantum circuit such that the given figure of merit is maximized by using the respectively trained optimized compiler.

        Arguments:
            qc: The quantum circuit to be compiled or the path to a qasm file containing the quantum circuit.
            tracer_output_path: Optional temporary path to export the compilation trace for this specific run.
            pass_timeout: Maximum duration in seconds for one compilation pass.
                Defaults to None, which disables pass timeouts.

        Returns:
            A tuple containing the compiled quantum circuit and the compilation information. If compilation fails, False is returned.

        Raises:
            RuntimeError: If an error occurs during compilation.
            ValueError: If ``pass_timeout`` is not positive.
        """
        original_tracer_output_path = self.env.tracer_output_path
        original_pass_timeout = self.env.pass_timeout

        try:
            # Temporarily override singleton settings for this compilation.
            if tracer_output_path is not None:
                self.env.tracer_output_path = tracer_output_path
            self.env.pass_timeout = pass_timeout

            trained_rl_model = load_model(self.model_name, graph=self.graph)

            graph_env: GNNObservationWrapper | None = None
            policy_env = self.env
            if self.graph:
                from mqt.predictor.rl.gnn import GNNObservationWrapper  # ruff: ignore[import-outside-top-level]

                graph_env = GNNObservationWrapper(self.env)
                policy_env = graph_env

            obs, _ = policy_env.reset(qc, seed=0)

            used_compilation_passes = []
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action_masks = get_action_masks(policy_env)
                policy_observation = graph_env.graph_observation if graph_env is not None else obs
                action, _ = trained_rl_model.predict(
                    policy_observation,  # ty: ignore[invalid-argument-type]
                    action_masks=action_masks,
                )
                action = int(action)
                action_item = self.env.action_set[action]
                used_compilation_passes.append(action_item.name)
                obs, _reward_val, terminated, truncated, _info = policy_env.step(action)

            if not self.env.error_occurred:
                return self.env.state, used_compilation_passes

            msg = "Error occurred during compilation."
            raise RuntimeError(msg)

        finally:
            self.env.tracer_output_path = original_tracer_output_path
            self.env.pass_timeout = original_pass_timeout

    def train_model(
        self,
        timesteps: int = 1000,
        verbose: int = 2,
        test: bool = False,
        seed: int | None = None,
        pass_timeout: float | None = None,
        iterations: int | None = None,
    ) -> None:
        """Trains all models for the given reward functions and device.

        Arguments:
            timesteps: The number of timesteps for flat-policy training. Ignored by the GNN policy. Defaults to 1000.
            verbose: The verbosity level. Defaults to 2.
            test: Whether to train the model for testing purposes. Defaults to False.
            seed: The random seed to use for reproducible training. The GNN policy preserves the prototype's seed
                of 0 when this is None, while the flat policy uses true randomness. Defaults to None.
            pass_timeout: Maximum duration in seconds for one compilation pass.
                Defaults to None, which disables pass timeouts.
            iterations: The number of GNN rollout iterations. Defaults to 1000, or 10 in test mode.
                Ignored by the flat policy.

        Raises:
            ValueError: If ``pass_timeout`` is not positive.
        """
        training_seed = 0 if self.graph and seed is None else seed
        if training_seed is not None:
            set_random_seed(training_seed)
        if test:
            # minimum training overhead
            n_steps = 20 if self.graph else max(timesteps, 2)
            n_epochs = 1
            batch_size = n_steps
            progress_bar = False
        else:
            # default PPO values
            n_steps = 2048
            n_epochs = 10
            batch_size = 64
            progress_bar = True

        original_pass_timeout = self.env.pass_timeout
        self.env.pass_timeout = pass_timeout
        try:
            logger.debug("Start training for: " + self.figure_of_merit + " on " + self.device_name)
            if self.graph:
                from mqt.predictor.rl.gnn import (  # ruff: ignore[import-outside-top-level]
                    GNNObservationWrapper,
                    create_gnn_model,
                )

                assert self.gnn_config is not None
                effective_gnn_config = (
                    replace(self.gnn_config, n_steps=n_steps, n_epochs=n_epochs, batch_size=batch_size)
                    if test
                    else self.gnn_config
                )
                n_steps = effective_gnn_config.n_steps
                graph_env = GNNObservationWrapper(self.env)
                model = create_gnn_model(
                    graph_env,
                    effective_gnn_config,
                    verbose=verbose,
                    tensorboard_log=f"./{self.model_name}",
                    seed=training_seed,
                )
            else:
                model = MaskablePPO(
                    MaskableMultiInputActorCriticPolicy,
                    self.env,
                    verbose=verbose,
                    tensorboard_log=f"./{self.model_name}",
                    gamma=0.98,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    seed=training_seed,
                )
            # Training Loop: In each iteration, the agent collects n_steps steps (rollout),
            # updates the policy for n_epochs, and then repeats the process until total_timesteps steps have been taken.
            total_timesteps = (
                n_steps * (iterations if iterations is not None else (10 if test else 1000))
                if self.graph
                else timesteps
            )
            model.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)
            model.save(get_path_trained_model() / self.model_name)
        finally:
            self.env.pass_timeout = original_pass_timeout


def load_model(model_name: str, *, graph: bool = False) -> MaskablePPO:
    """Loads a trained model from the trained model folder.

    Arguments:
        model_name: The name of the model to be loaded.
        graph: Whether the model uses the GNN policy. Defaults to False.

    Returns:
        The loaded model.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    path = get_path_trained_model()
    if Path(path / (model_name + ".zip")).is_file():
        if graph:
            from mqt.predictor.rl.gnn import GNNMaskablePPO  # ruff: ignore[import-outside-top-level]

            return GNNMaskablePPO.load(path / (model_name + ".zip"))
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
    mdp: MDPPolicy = "v3",
    pass_timeout: float | None = None,
    graph: bool = False,
) -> tuple[QuantumCircuit, list[str]]:
    """Compiles a given quantum circuit to a device optimizing for the given figure of merit.

    Arguments:
        qc: The quantum circuit to be compiled. If a string is given, it is assumed to be a path to a qasm file.
        device: The device to compile to.
        figure_of_merit: The figure of merit to be used for compilation. Defaults to "expected_fidelity".
        predictor_singleton: A predictor object that is used for compilation to reduce compilation time when compiling multiple quantum circuits. If None, a new predictor object is created. Defaults to None.
        tracer_output_path: If provided, enables compiler tracing and exports the JSON log to the specified path.
        mdp: The MDP transition policy used when constructing a predictor. ``v2``
            is the original strategy and ``v3`` is the default. When
            ``predictor_singleton`` is provided, its configured policy is used instead.
        pass_timeout: Maximum duration in seconds for one compilation pass.
            Defaults to None, which disables pass timeouts.
        graph: Whether to use the opt-in GNN policy. Ignored when ``predictor_singleton`` is provided.
            Defaults to False.

    Returns:
        A tuple containing the compiled quantum circuit and the compilation information. If compilation fails, False is returned.

    Raises:
        ValueError: If figure_of_merit or device is None and predictor_singleton is also None,
            or if ``pass_timeout`` is not positive.
    """
    if predictor_singleton is None:
        if figure_of_merit is None:
            msg = "figure_of_merit must not be None if predictor_singleton is None."
            raise ValueError(msg)
        if device is None:
            msg = "device must not be None if predictor_singleton is None."
            raise ValueError(msg)
        predictor = Predictor(
            figure_of_merit=figure_of_merit,
            device=device,
            tracer_output_path=tracer_output_path,
            mdp=mdp,
            graph=graph,
        )
        return predictor.compile_as_predicted(qc, pass_timeout=pass_timeout)

    return predictor_singleton.compile_as_predicted(
        qc, tracer_output_path=tracer_output_path, pass_timeout=pass_timeout
    )
