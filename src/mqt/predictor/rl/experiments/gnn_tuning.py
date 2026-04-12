# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""GNN hyperparameter tuning via Optuna for RL compilation predictor.

This module provides infrastructure for tuning GNN architecture hyperparameters
(hidden_dim, num_conv_wo_resnet, num_resnet_layers, dropout, bidirectional)
using Optuna and quick training trials (e.g., 2k steps each).

The goal is to identify optimal GNN configurations without full training,
outputting best parameters for downstream use.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import optuna
import torch
from optuna.samplers import TPESampler

from mqt.predictor.rl.gnn_ppo import create_gnn_policy, train_ppo_with_gnn
from mqt.predictor.rl.helper import GLOBAL_FEATURE_DIM
from mqt.predictor.rl.predictorenv import PredictorEnv

if TYPE_CHECKING:
    import logging
    from pathlib import Path

    from qiskit.transpiler import Target

    from mqt.predictor.reward import figure_of_merit


@dataclass(frozen=True, slots=True)
class GNNTrialConfig:
    """Configuration for one GNN hyperparameter tuning trial."""

    device_name: str
    figure_of_merit: figure_of_merit
    mdp: str = "paper"
    trial_steps: int = 2000
    steps_per_iteration: int = 2048
    lr: float = 3e-4
    gnn_lr: float = 1e-4
    num_epochs: int = 10
    minibatch_size: int = 64


def run_gnn_hyperparameter_tuning(
    *,
    device: Target,
    figure_of_merit: figure_of_merit,
    output_dir: Path,
    num_trials: int = 10,
    trial_steps: int = 2000,
    mdp: str = "paper",
    logger: logging.Logger,
) -> dict[str, Any]:
    """Run Optuna-based GNN hyperparameter tuning for one device.

    Args:
        device: Target quantum device for environment.
        figure_of_merit: Reward function for environment.
        output_dir: Directory to save results (per-device tuning directory).
        num_trials: Number of Optuna trials to run.
        trial_steps: Training steps per trial.
        mdp: MDP variant (fixed, typically "paper").
        logger: Logger for progress.

    Returns:
        Dict with best_params, best_score, trial_history, and metadata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    trial_config = GNNTrialConfig(
        device_name=device.description,
        figure_of_merit=figure_of_merit,
        mdp=mdp,
        trial_steps=trial_steps,
    )

    logger.info("Starting GNN hyperparameter tuning for %s.", device.description)
    logger.info("Number of trials: %d", num_trials)
    logger.info("Steps per trial: %d", trial_steps)

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function for GNN hyperparameter optimization."""
        return _objective(trial=trial, device=device, config=trial_config, logger=logger)

    sampler = TPESampler(n_startup_trials=10)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=num_trials, show_progress_bar=True)

    # Extract results
    best_trial = study.best_trial
    best_params = best_trial.params
    best_score = best_trial.value

    trial_history = [
        {
            "trial_id": trial.number,
            "params": trial.params,
            "score": trial.value,
            "state": trial.state.name,
        }
        for trial in study.trials
    ]

    results = {
        "device": device.description,
        "figure_of_merit": figure_of_merit,
        "mdp": mdp,
        "completed_at": datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat(),
        "best_params": best_params,
        "best_score": best_score,
        "best_trial": best_trial.number,
        "trials_completed": len(study.trials),
        "trial_history": trial_history,
    }

    result_path = output_dir / "gnn_tuning_results.json"
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("GNN tuning results saved to %s", result_path)
    logger.info("Best trial: %d, best score: %.4f", best_trial.number, best_score)

    return results


def _objective(
    trial: optuna.Trial,
    device: Target,
    config: GNNTrialConfig,
    logger: logging.Logger,
) -> float:
    """Optuna objective function for one GNN trial.

    Suggests hyperparameters, trains policy for trial_steps, and returns mean episode return.

    Args:
        trial: Optuna trial object.
        device: Target device for environment.
        config: GNNTrialConfig with fixed parameters.
        logger: Logger for progress.

    Returns:
        Mean episode return (higher is better).
    """
    # Suggest hyperparameters
    hidden_dim = trial.suggest_int("hidden_dim", 8, 64)
    num_conv_wo_resnet = trial.suggest_int("num_conv_wo_resnet", 1, 3)
    num_resnet_layers = trial.suggest_int("num_resnet_layers", 1, 9)
    dropout_p = trial.suggest_categorical("dropout_p", [0.0, 0.1, 0.2, 0.3])
    bidirectional = trial.suggest_categorical("bidirectional", [False, True])

    trial_id = trial.number
    logger.debug(
        "Trial %d: hidden_dim=%d, num_conv_wo_resnet=%d, num_resnet_layers=%d, dropout_p=%.1f, bidirectional=%s",
        trial_id,
        hidden_dim,
        num_conv_wo_resnet,
        num_resnet_layers,
        dropout_p,
        bidirectional,
    )

    try:
        # Create environment with fixed MDP
        env = PredictorEnv(
            reward_function=config.figure_of_merit,
            device=device,
            graph=True,  # Use graph observations for GNN
            mdp=config.mdp,
        )

        # Sample one observation to determine node feature dim
        sample_obs, _ = env.reset()
        node_feature_dim = sample_obs.x.shape[1]  # type: ignore[unresolved-attribute]

        # Create GNN policy with suggested hyperparameters
        policy = create_gnn_policy(
            node_feature_dim=node_feature_dim,
            num_actions=env.action_space.n,  # type: ignore[unresolved-attribute]
            hidden_dim=hidden_dim,
            num_conv_wo_resnet=num_conv_wo_resnet,
            num_resnet_layers=num_resnet_layers,
            dropout_p=dropout_p,
            bidirectional=bidirectional,
            global_feature_dim=GLOBAL_FEATURE_DIM,
        )

        # Compute number of iterations needed to reach trial_steps
        num_iterations = (config.trial_steps + config.steps_per_iteration - 1) // config.steps_per_iteration

        # Train GNN with PPO for this trial
        policy = train_ppo_with_gnn(
            env=env,
            policy=policy,
            num_iterations=num_iterations,
            steps_per_iteration=config.steps_per_iteration,
            num_epochs=config.num_epochs,
            minibatch_size=config.minibatch_size,
            lr=config.lr,
            gnn_lr=config.gnn_lr,
        )

        # Evaluate: run final episode to measure return
        obs, _ = env.reset()
        episode_return = 0.0
        terminated = False
        truncated = False
        step_count = 0

        while not (terminated or truncated) and step_count < 200:  # max 200 steps per episode
            # Get action from policy (no gradient)
            with torch.no_grad():
                action, _ = policy(obs)
            obs, reward, terminated, truncated, _info = env.step(action.item())
            episode_return += reward
            step_count += 1

        logger.debug("Trial %d completed: episode_return=%.4f", trial_id, episode_return)

    except Exception:
        logger.exception("Trial %d failed", trial_id)
        # Return a very low score to discourage this hyperparameter combination
        return -999.0
    else:
        return episode_return
