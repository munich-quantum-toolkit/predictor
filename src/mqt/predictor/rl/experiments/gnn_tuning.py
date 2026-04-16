# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""GNN hyperparameter tuning via Optuna for RL compilation predictor.

This module tunes GNN architecture hyperparameters
(``hidden_dim``, ``num_conv_wo_resnet``, ``num_resnet_layers``, ``dropout_p``,
``bidirectional``) using short PPO training runs and held-out evaluation.

The tuner trains on the configured RL training split and prefers a sibling
validation split when available. If no validation split exists, it falls back
to the regular held-out test split rather than evaluating on the training data.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from mqt.predictor.rl.experiments.evaluation import (
    average_figure_of_merit,
    evaluate_gnn_policy,
    resolve_validation_circuit_directory,
)
from mqt.predictor.rl.gnn_ppo import create_gnn_policy, train_ppo_with_gnn
from mqt.predictor.rl.helper import GLOBAL_FEATURE_DIM, get_path_training_circuits_train
from mqt.predictor.rl.predictorenv import PredictorEnv

if TYPE_CHECKING:
    import logging

    from optuna.trial import FrozenTrial
    from qiskit.transpiler import Target

    from mqt.predictor.reward import figure_of_merit


@dataclass(frozen=True, slots=True)
class GNNTrialConfig:
    """Configuration for one GNN hyperparameter tuning trial."""

    device_name: str
    figure_of_merit: figure_of_merit
    path_training_circuits: Path
    path_evaluation_circuits: Path
    mdp: str = "paper"
    trial_steps: int = 5000
    steps_per_iteration: int = 2048
    lr: float = 1e-3
    gnn_lr: float = 1e-3
    num_epochs: int = 10
    minibatch_size: int = 64
    max_episode_steps: int | None = None
    max_eval_circuits: int = 32
    max_eval_steps: int = 200
    evaluation_deterministic: bool = True


def _study_storage_uri(output_dir: Path) -> str:
    """Return the SQLite storage URI for one tuning directory."""
    return f"sqlite:///{(output_dir / 'gnn_tuning_optuna.db').resolve()}"


def _study_name(config: GNNTrialConfig) -> str:
    """Return a stable Optuna study name for one tuning configuration."""
    return f"gnn_tuning::{config.device_name}::{config.figure_of_merit}::{config.mdp}"


def _study_config_payload(config: GNNTrialConfig) -> dict[str, Any]:
    """Return the persisted study configuration used for resume validation."""
    return {
        "device_name": config.device_name,
        "figure_of_merit": config.figure_of_merit,
        "mdp": config.mdp,
        "path_training_circuits": str(config.path_training_circuits),
        "path_evaluation_circuits": str(config.path_evaluation_circuits),
        "trial_steps": config.trial_steps,
        "steps_per_iteration": config.steps_per_iteration,
        "lr": config.lr,
        "gnn_lr": config.gnn_lr,
        "num_epochs": config.num_epochs,
        "minibatch_size": config.minibatch_size,
        "max_episode_steps": config.max_episode_steps,
        "max_eval_circuits": config.max_eval_circuits,
        "max_eval_steps": config.max_eval_steps,
        "evaluation_deterministic": config.evaluation_deterministic,
    }


def _ensure_study_config_matches(study: optuna.Study, config: GNNTrialConfig) -> None:
    """Validate that a resumed study matches the requested tuning configuration."""
    persisted_config = study.user_attrs.get("gnn_trial_config")
    requested_config = _study_config_payload(config)
    if persisted_config is None:
        study.set_user_attr("gnn_trial_config", requested_config)
        return
    if persisted_config != requested_config:
        msg = (
            "Existing GNN tuning study configuration does not match the requested run. "
            f"Study name: {study.study_name}. Existing config: {persisted_config}. "
            f"Requested config: {requested_config}."
        )
        raise ValueError(msg)


def _fail_stale_trials(study: optuna.Study, logger: logging.Logger) -> int:
    """Mark unfinished trials from a previous interrupted run as failed."""
    stale_trials = study.get_trials(deepcopy=False, states=(TrialState.RUNNING, TrialState.WAITING))
    if not stale_trials:
        return 0

    failed_trials = 0
    for trial in stale_trials:
        if study._storage.set_trial_state_values(trial._trial_id, TrialState.FAIL):  # noqa: SLF001
            failed_trials += 1

    if failed_trials:
        logger.warning("Marked %d stale GNN tuning trials as failed before resuming.", failed_trials)
    return failed_trials


def _complete_trial_count(study: optuna.Study) -> int:
    """Return the number of completed trials with an objective value."""
    return len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))


def _build_trial_history(study: optuna.Study) -> list[dict[str, Any]]:
    """Convert study trials into a JSON-serializable history."""
    return [
        {
            "trial_id": trial.number,
            "params": trial.params,
            "score": trial.value,
            "state": trial.state.name,
        }
        for trial in study.trials
    ]


def _best_complete_trial(study: optuna.Study) -> FrozenTrial | None:
    """Return the best completed trial if one exists."""
    completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    if not completed_trials:
        return None
    return max(completed_trials, key=lambda trial: float(trial.value))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f"{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _build_results_payload(
    *,
    study: optuna.Study,
    config: GNNTrialConfig,
    figure_of_merit: figure_of_merit,
    startup_trials: int,
) -> dict[str, Any]:
    """Build the persisted tuning results document."""
    best_trial = _best_complete_trial(study)
    completed_trials = _complete_trial_count(study)
    failed_trials = len(study.get_trials(deepcopy=False, states=(TrialState.FAIL,)))
    pruned_trials = len(study.get_trials(deepcopy=False, states=(TrialState.PRUNED,)))

    return {
        "device": config.device_name,
        "figure_of_merit": figure_of_merit,
        "mdp": config.mdp,
        "completed_at": datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat(),
        "training_directory": str(config.path_training_circuits),
        "evaluation_directory": str(config.path_evaluation_circuits),
        "evaluation_deterministic": config.evaluation_deterministic,
        "trial_steps": config.trial_steps,
        "steps_per_iteration": config.steps_per_iteration,
        "max_episode_steps": config.max_episode_steps,
        "startup_trials": startup_trials,
        "study_name": study.study_name,
        "storage_uri": _study_storage_uri(Path(study.user_attrs["output_dir"])),
        "best_params": None if best_trial is None else best_trial.params,
        "best_score": None if best_trial is None else best_trial.value,
        "best_trial": None if best_trial is None else best_trial.number,
        "trials_completed": completed_trials,
        "trials_failed": failed_trials,
        "trials_pruned": pruned_trials,
        "trials_total": len(study.trials),
        "trial_history": _build_trial_history(study),
    }


def _persist_study_results(
    *,
    study: optuna.Study,
    config: GNNTrialConfig,
    figure_of_merit: figure_of_merit,
    output_dir: Path,
    startup_trials: int,
) -> dict[str, Any]:
    """Persist the current tuning snapshot to disk and return it."""
    results = _build_results_payload(
        study=study,
        config=config,
        figure_of_merit=figure_of_merit,
        startup_trials=startup_trials,
    )
    result_path = output_dir / "gnn_tuning_results.json"
    _write_json_atomic(result_path, results)
    return results


def run_gnn_hyperparameter_tuning(
    *,
    device: Target,
    figure_of_merit: figure_of_merit,
    output_dir: Path,
    num_trials: int = 10,
    trial_steps: int = 5000,
    mdp: str = "paper",
    path_training_circuits: str | Path | None = None,
    path_validation_circuits: str | Path | None = None,
    max_episode_steps: int | None = None,
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
        path_training_circuits: Optional training split directory for PPO.
        path_validation_circuits: Optional held-out validation directory.
        max_episode_steps: Optional hard cap on environment steps per training/evaluation episode.
        logger: Logger for progress.

    Returns:
        Dict with best_params, best_score, trial_history, and metadata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    training_dir = (
        Path(path_training_circuits) if path_training_circuits is not None else get_path_training_circuits_train()
    )
    evaluation_dir = resolve_validation_circuit_directory(training_dir, path_validation_circuits)

    trial_config = GNNTrialConfig(
        device_name=device.description,
        figure_of_merit=figure_of_merit,
        path_training_circuits=training_dir,
        path_evaluation_circuits=evaluation_dir,
        mdp=mdp,
        trial_steps=trial_steps,
        max_episode_steps=max_episode_steps,
    )

    logger.info("Starting GNN hyperparameter tuning for %s.", device.description)
    logger.info("Number of trials: %d", num_trials)
    logger.info("Steps per trial: %d", trial_steps)
    logger.info("Training circuits: %s", training_dir)
    logger.info("Held-out evaluation circuits: %s", evaluation_dir)

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function for GNN hyperparameter optimization."""
        return _objective(trial=trial, device=device, config=trial_config, logger=logger)

    startup_trials = min(10, max(1, num_trials // 3))
    logger.info("Optuna startup trials: %d", startup_trials)
    sampler = TPESampler(n_startup_trials=startup_trials)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        storage=_study_storage_uri(output_dir),
        study_name=_study_name(trial_config),
        load_if_exists=True,
    )
    study.set_user_attr("output_dir", str(output_dir))
    _ensure_study_config_matches(study, trial_config)
    _fail_stale_trials(study, logger)

    completed_before_resume = _complete_trial_count(study)
    remaining_trials = max(0, num_trials - completed_before_resume)
    logger.info(
        "Resuming GNN tuning study '%s': %d/%d completed trials, %d remaining.",
        study.study_name,
        completed_before_resume,
        num_trials,
        remaining_trials,
    )

    def persist_callback(study: optuna.Study, _trial: FrozenTrial) -> None:
        """Persist one tuning snapshot after every completed Optuna trial."""
        results = _persist_study_results(
            study=study,
            config=trial_config,
            figure_of_merit=figure_of_merit,
            output_dir=output_dir,
            startup_trials=startup_trials,
        )
        logger.info(
            "Persisted GNN tuning snapshot: %d completed, %d total trials.",
            results["trials_completed"],
            results["trials_total"],
        )

    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials, show_progress_bar=True, callbacks=[persist_callback])
    else:
        logger.info("Requested GNN tuning trial count already reached. Skipping new trials.")

    results = _persist_study_results(
        study=study,
        config=trial_config,
        figure_of_merit=figure_of_merit,
        output_dir=output_dir,
        startup_trials=startup_trials,
    )
    logger.info("GNN tuning results saved to %s", output_dir / "gnn_tuning_results.json")
    if results["best_trial"] is None:
        logger.warning("No completed GNN tuning trial produced a score.")
    else:
        logger.info("Best trial: %d, best score: %.4f", results["best_trial"], results["best_score"])

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
    hidden_dim = trial.suggest_int("hidden_dim", 112, 128)
    num_conv_wo_resnet = trial.suggest_categorical("num_conv_wo_resnet", [1, 2])
    num_resnet_layers = trial.suggest_int("num_resnet_layers", 4, 7)
    dropout_p = trial.suggest_categorical("dropout_p", [0.1])
    bidirectional = trial.suggest_categorical("bidirectional", [False])

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
            path_training_circuits=config.path_training_circuits,
            max_episode_steps=config.max_episode_steps,
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

        # Evaluate on a held-out split using the shared RL evaluation helpers
        _evaluation_dir, evaluation_results = evaluate_gnn_policy(
            policy=policy,
            device=device,
            figure_of_merit=config.figure_of_merit,
            mdp=config.mdp,
            path_training_circuits=config.path_training_circuits,
            path_evaluation_circuits=config.path_evaluation_circuits,
            max_steps=config.max_eval_steps,
            max_episode_steps=config.max_episode_steps,
            deterministic=config.evaluation_deterministic,
            seed=trial_id,
            max_circuits=config.max_eval_circuits,
        )
        avg_figure_of_merit = average_figure_of_merit(evaluation_results)
        logger.debug("Trial %d completed: avg_figure_of_merit=%.4f", trial_id, avg_figure_of_merit)

    except Exception:
        logger.exception("Trial %d failed", trial_id)
        # Return a very low score to discourage this hyperparameter combination
        return -999.0
    else:
        return avg_figure_of_merit
