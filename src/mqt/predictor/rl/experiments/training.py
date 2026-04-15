# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Train-only entry point for RL predictor models."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mqt.bench.targets import get_device
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from mqt.predictor.rl.helper import get_path_trained_model, get_path_training_circuits_train
from mqt.predictor.rl.predictor import Predictor

if TYPE_CHECKING:
    from qiskit.transpiler import Target
    from stable_baselines3.common.callbacks import BaseCallback

    from mqt.predictor.reward import figure_of_merit


@dataclass(slots=True)
class RLTrainingResult:
    """Result of an RL training-only run."""

    model_path: Path
    training_directory: Path
    total_timesteps: int
    checkpoint_directory: Path | None = None
    resumed_from_checkpoint: Path | None = None


def run_rl_training(
    device: Target,
    figure_of_merit: figure_of_merit = "expected_fidelity",
    mdp: str = "paper",
    timesteps: int = 10000,
    verbose: int = 1,
    test: bool = False,
    path_training_circuits: str | Path | None = None,
    checkpoint_directory: str | Path | None = None,
    checkpoint_frequency: int | None = None,
    resume_from_checkpoint: str | Path | None = None,
    max_episode_steps: int | None = None,
    callback: BaseCallback | None = None,
) -> RLTrainingResult:
    """Train an RL predictor on an existing training split."""
    training_directory = (
        Path(path_training_circuits) if path_training_circuits is not None else get_path_training_circuits_train()
    )
    resolved_checkpoint_directory = Path(checkpoint_directory) if checkpoint_directory is not None else None
    resolved_resume_from_checkpoint = Path(resume_from_checkpoint) if resume_from_checkpoint is not None else None

    predictor = Predictor(
        figure_of_merit=figure_of_merit,
        device=device,
        mdp=mdp,
        path_training_circuits=training_directory,
        max_episode_steps=max_episode_steps,
    )
    callbacks: list[BaseCallback] = []
    if resolved_checkpoint_directory is not None and checkpoint_frequency is not None and checkpoint_frequency > 0:
        resolved_checkpoint_directory.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=checkpoint_frequency,
                save_path=str(resolved_checkpoint_directory),
                name_prefix="model_checkpoint",
            )
        )
    if callback is not None:
        callbacks.append(callback)

    trained_model = predictor.train_model(
        timesteps=timesteps,
        verbose=verbose,
        test=test,
        callback=CallbackList(callbacks) if callbacks else None,
        resume_from=resolved_resume_from_checkpoint,
    )
    assert trained_model is not None

    model_path = get_path_trained_model() / f"model_{figure_of_merit}_{device.description}.zip"
    return RLTrainingResult(
        model_path=model_path,
        training_directory=training_directory,
        total_timesteps=int(trained_model.num_timesteps),
        checkpoint_directory=resolved_checkpoint_directory,
        resumed_from_checkpoint=resolved_resume_from_checkpoint,
    )


def main() -> None:
    """Run RL training from the command line."""
    parser = argparse.ArgumentParser(description="Train an RL predictor on an existing training split.")
    parser.add_argument("--device", required=True, help="Target device name, e.g. ibm_falcon_127.")
    parser.add_argument(
        "--figure-of-merit",
        default="expected_fidelity",
        choices=[
            "expected_fidelity",
            "critical_depth",
            "estimated_success_probability",
            "estimated_hellinger_distance",
        ],
        help="Figure of merit used for RL training.",
    )
    parser.add_argument(
        "--mdp",
        default="paper",
        choices=["paper", "flexible", "thesis", "hybrid"],
        help="MDP transition policy used by the RL environment.",
    )
    parser.add_argument("--timesteps", type=int, default=10000, help="Training timesteps.")
    parser.add_argument("--verbose", type=int, default=1, help="Training verbosity passed to PPO.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use the lightweight test training configuration.",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=None,
        help="Optional path to the existing training circuits.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Optional directory for periodic PPO checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=None,
        help="Optional PPO checkpoint frequency in timesteps.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Optional PPO checkpoint to resume training from.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="Optional hard cap on environment steps per training episode.",
    )
    args = parser.parse_args()

    result = run_rl_training(
        device=get_device(args.device),
        figure_of_merit=args.figure_of_merit,
        mdp=args.mdp,
        timesteps=args.timesteps,
        verbose=args.verbose,
        test=args.test,
        path_training_circuits=args.train_dir,
        checkpoint_directory=args.checkpoint_dir,
        checkpoint_frequency=args.checkpoint_frequency,
        resume_from_checkpoint=args.resume_from_checkpoint,
        max_episode_steps=args.max_episode_steps,
    )

    print(f"Training directory: {result.training_directory}")
    print(f"Model: {result.model_path}")


if __name__ == "__main__":
    main()
