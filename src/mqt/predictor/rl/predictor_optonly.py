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
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.utils import set_random_seed

from mqt.predictor.rl.helper import get_path_trained_model, logger
from mqt.predictor.rl.predictorenv_optonly import KIT_QASM_DIR, OptOnlyPredictorEnv

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from qiskit import QuantumCircuit


_LATEST_CHECKPOINT_FILE = "latest.txt"
_CHECKPOINT_STATE_SUFFIX = ".state.pkl"


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

    def train_model(
        self,
        timesteps: int = 200_000,
        verbose: int = 1,
        test: bool = False,
        n_checkpoint: int | None = 1000,
        resume: bool = True,
        seed: int = 0,
        sampling_seed: int = 10,
        checkpoint_dir: Path | str | None = None,
        log_applied_passes: bool = True,
    ) -> None:
        """Train the optimization-only RL model.

        Args:
            timesteps: Number of additional PPO training timesteps.
            verbose: Stable-Baselines verbosity.
            test: Use a tiny rollout configuration suitable for tests.
            n_checkpoint: Save training state after each chunk of this many timesteps.
                Pass ``None`` to disable periodic checkpoints.
            resume: Continue from the latest checkpoint when one exists.
            seed: Random seed for PPO, Python, NumPy, and Torch when starting fresh.
            sampling_seed: Random seed for training-circuit sampling when starting fresh.
            checkpoint_dir: Directory for model and training-state checkpoints. Defaults
                to a model-specific directory next to the final trained model.
            log_applied_passes: Log every opt-only action before and after it runs.

        Raises:
            ValueError: If ``timesteps`` or ``n_checkpoint`` is invalid.
        """
        if timesteps <= 0:
            msg = "timesteps must be positive."
            raise ValueError(msg)
        if n_checkpoint is not None and n_checkpoint <= 0:
            msg = "n_checkpoint must be positive or None."
            raise ValueError(msg)

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

        model_path = get_path_trained_model()
        model_path.mkdir(parents=True, exist_ok=True)
        resolved_checkpoint_dir = _resolve_checkpoint_dir(model_path, self.MODEL_NAME, checkpoint_dir)
        resolved_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.env.log_applied_passes = log_applied_passes

        checkpoint = _latest_checkpoint(resolved_checkpoint_dir)
        if resume and checkpoint is not None:
            model = MaskablePPO.load(checkpoint.model_path, env=self.env, verbose=verbose)
            checkpoint_status = _restore_training_status(checkpoint.state_path, self.env)
            seed = int(checkpoint_status.get("seed", seed))
            sampling_seed = int(checkpoint_status.get("sampling_seed", sampling_seed))
            reset_num_timesteps = False
            logger.info("Resuming opt-only RL training from checkpoint %s.", checkpoint.model_path)
        else:
            set_random_seed(seed)
            random.seed(seed)
            # Stable-Baselines still uses NumPy's global RNG; keep it checkpointable.
            np.random.seed(seed)  # noqa: NPY002
            torch.manual_seed(seed)
            self.env.rng = np.random.default_rng(sampling_seed)
            model = MaskablePPO(
                MaskableMultiInputActorCriticPolicy,
                self.env,
                verbose=verbose,
                tensorboard_log=f"./{self.MODEL_NAME}",
                gamma=0.98,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                seed=seed,
            )
            reset_num_timesteps = True

        target_num_timesteps = int(model.num_timesteps) + timesteps
        while model.num_timesteps < target_num_timesteps:
            remaining_timesteps = target_num_timesteps - int(model.num_timesteps)
            chunk_timesteps = remaining_timesteps if n_checkpoint is None else min(n_checkpoint, remaining_timesteps)
            model.learn(
                total_timesteps=chunk_timesteps,
                progress_bar=progress_bar,
                reset_num_timesteps=reset_num_timesteps,
            )
            reset_num_timesteps = False

            if n_checkpoint is not None:
                _save_checkpoint(
                    model=model,
                    env=self.env,
                    checkpoint_dir=resolved_checkpoint_dir,
                    model_name=self.MODEL_NAME,
                    seed=seed,
                    sampling_seed=sampling_seed,
                )

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


@dataclass(frozen=True, slots=True)
class _Checkpoint:
    """Resolved checkpoint files."""

    model_path: Path
    state_path: Path


def _resolve_checkpoint_dir(model_path: Path, model_name: str, checkpoint_dir: Path | str | None) -> Path:
    """Resolve the checkpoint directory for opt-only training."""
    if checkpoint_dir is not None:
        return Path(checkpoint_dir)
    return model_path / f"{model_name}_checkpoints"


def _checkpoint_stem(model_name: str, num_timesteps: int) -> str:
    """Return the checkpoint stem for a model timestep."""
    return f"{model_name}_step_{num_timesteps}"


def _save_checkpoint(
    *,
    model: MaskablePPO,
    env: OptOnlyPredictorEnv,
    checkpoint_dir: Path,
    model_name: str,
    seed: int,
    sampling_seed: int,
) -> None:
    """Save the PPO model and matching training status."""
    checkpoint_stem = _checkpoint_stem(model_name, int(model.num_timesteps))
    checkpoint_model_path = checkpoint_dir / checkpoint_stem
    checkpoint_state_path = checkpoint_dir / f"{checkpoint_stem}{_CHECKPOINT_STATE_SUFFIX}"

    model.save(checkpoint_model_path)
    with checkpoint_state_path.open("wb") as file:
        pickle.dump(
            {
                "num_timesteps": int(model.num_timesteps),
                "seed": seed,
                "sampling_seed": sampling_seed,
                "env": env.snapshot_training_state(),
                "python_random_state": random.getstate(),
                # Stable-Baselines still uses NumPy's global RNG; keep it checkpointable.
                "numpy_random_state": np.random.get_state(),  # noqa: NPY002
                "torch_random_state": torch.random.get_rng_state(),
            },
            file,
        )

    latest_checkpoint_path = checkpoint_dir / _LATEST_CHECKPOINT_FILE
    latest_checkpoint_path.write_text(
        f"{checkpoint_model_path.with_suffix('.zip').name}\n{checkpoint_state_path.name}\n",
        encoding="utf-8",
    )
    logger.info("Saved opt-only RL checkpoint at %s.", checkpoint_model_path.with_suffix(".zip"))


def _latest_checkpoint(checkpoint_dir: Path) -> _Checkpoint | None:
    """Return the latest checkpoint recorded by the marker file."""
    latest_checkpoint_path = checkpoint_dir / _LATEST_CHECKPOINT_FILE
    if not latest_checkpoint_path.is_file():
        return None

    checkpoint_files = latest_checkpoint_path.read_text(encoding="utf-8").splitlines()
    if len(checkpoint_files) < 2:
        return None

    model_path = checkpoint_dir / checkpoint_files[0]
    state_path = checkpoint_dir / checkpoint_files[1]
    if not model_path.is_file() or not state_path.is_file():
        return None
    return _Checkpoint(model_path=model_path, state_path=state_path)


def _restore_training_status(state_path: Path, env: OptOnlyPredictorEnv) -> dict[str, Any]:
    """Restore random states and the optimization-only environment."""
    with state_path.open("rb") as file:
        status = pickle.load(file)

    env.restore_training_state(status["env"])
    random.setstate(status["python_random_state"])
    # Stable-Baselines still uses NumPy's global RNG; keep it checkpointable.
    np.random.set_state(status["numpy_random_state"])  # noqa: NPY002
    torch.random.set_rng_state(status["torch_random_state"])
    return status


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
