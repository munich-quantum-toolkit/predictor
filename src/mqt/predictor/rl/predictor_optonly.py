# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Training wrapper for the optimization-only RL predictor."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.utils import set_random_seed

from mqt.predictor.rl.helper import get_path_trained_model, logger
from mqt.predictor.rl.predictorenv_optonly import KIT_QASM_DIR, OptOnlyPredictorEnv

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from qiskit import QuantumCircuit


class OptOnlyPredictor:
    """Train and use the RL predictor under KIT's optimization-only protocol."""

    MODEL_NAME = "model_optimization_ratio_alltoall_S"

    def __init__(
        self,
        path_training_circuits: Path | str | None = None,
        baseline_cx_lookup: Mapping[str, int | float] | None = None,
        excluded_circuit_ids: Collection[str] | None = None,
        test_circuits_csv: Path | str | None = None,
        logger_level: int = logging.INFO,
    ) -> None:
        """Initialize the opt-only predictor.

        Args:
            path_training_circuits: Directory containing QASM files for training.
                Defaults to the ICSE Qiskit-ML raw corpus checkout.
            baseline_cx_lookup: Optional precomputed baseline counts keyed by circuit stem.
            excluded_circuit_ids: Circuit stems excluded from training.
            test_circuits_csv: Optional held-out-circuit CSV. Columns ``circuit_id``
                and ``circuit_name`` are recognized.
            logger_level: Log level for the shared predictor logger.
        """
        logger.setLevel(logger_level)
        excluded_ids = {Path(circuit_id).stem for circuit_id in (excluded_circuit_ids or set())}
        if test_circuits_csv is not None:
            excluded_ids.update(load_circuit_ids_from_csv(Path(test_circuits_csv)))

        # TODO: Replace the default path/exclusion inputs with the authoritative
        # train/test split once the external ICSE benchmark split is checked in.
        self.env = OptOnlyPredictorEnv(
            path_training_circuits=Path(path_training_circuits) if path_training_circuits else KIT_QASM_DIR,
            baseline_cx_lookup=baseline_cx_lookup,
            excluded_circuit_ids=excluded_ids,
        )

    def train_model(self, timesteps: int = 200_000, verbose: int = 1, test: bool = False) -> None:
        """Train the optimization-only RL model.

        Args:
            timesteps: Number of PPO training timesteps.
            verbose: Stable-Baselines verbosity.
            test: Use a tiny rollout configuration suitable for tests.
        """
        set_random_seed(0)
        if test:
            n_steps = max(timesteps, 2)
            n_epochs = 1
            batch_size = n_steps
            progress_bar = False
        else:
            n_steps = 2048
            n_epochs = 10
            batch_size = 64
            progress_bar = True

        model = MaskablePPO(
            MaskableMultiInputActorCriticPolicy,
            self.env,
            verbose=verbose,
            tensorboard_log=f"./{self.MODEL_NAME}",
            gamma=0.98,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
        )
        model.learn(total_timesteps=timesteps, progress_bar=progress_bar)
        model_path = get_path_trained_model()
        model_path.mkdir(parents=True, exist_ok=True)
        model.save(model_path / self.MODEL_NAME)

    def compile_as_predicted(self, qc: QuantumCircuit | str | Path) -> tuple[QuantumCircuit, list[str], str]:
        """Compile a circuit with the trained opt-only policy.

        Args:
            qc: The quantum circuit or QASM path to optimize.

        Returns:
            The optimized circuit, selected pass names, and terminal status.
        """
        model = load_optonly_model()
        obs, _ = self.env.reset(qc, seed=0)
        used_compilation_passes = []
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action_masks = get_action_masks(self.env)
            action, _ = model.predict(obs, action_masks=action_masks)
            action = int(action)
            used_compilation_passes.append(str(self.env.action_set[action].name))
            obs, _reward_val, terminated, truncated, _info = self.env.step(action)

        return self.env.state, used_compilation_passes, self.env.status


def load_optonly_model(model_name: str = OptOnlyPredictor.MODEL_NAME) -> MaskablePPO:
    """Load the trained optimization-only RL model.

    Args:
        model_name: Name of the model without the ``.zip`` suffix.

    Returns:
        The loaded model.

    Raises:
        FileNotFoundError: If the model has not been trained.
    """
    model_path = get_path_trained_model() / f"{model_name}.zip"
    if model_path.is_file():
        return MaskablePPO.load(model_path)

    error_msg = f"The opt-only RL model '{model_name}' is not trained yet. Please train the model before using it."
    logger.error(error_msg)
    raise FileNotFoundError(error_msg)


def load_circuit_ids_from_csv(path: Path) -> set[str]:
    """Load circuit stems from a held-out-circuit CSV file.

    Args:
        path: CSV containing either a ``circuit_id`` or ``circuit_name`` column.

    Returns:
        Circuit stems from the CSV.

    Raises:
        ValueError: If no supported circuit-name column is present.
    """
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        column = next((name for name in ("circuit_id", "circuit_name") if name in fieldnames), None)
        if column is None:
            msg = f"{path} must contain a 'circuit_id' or 'circuit_name' column."
            raise ValueError(msg)
        return {Path(row[column]).stem for row in reader if row.get(column)}
