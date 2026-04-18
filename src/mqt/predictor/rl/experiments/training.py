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


def _extract_checkpoint_steps(path: Path) -> int:
    """Extract the numeric step count from a checkpoint filename."""
    stem = path.stem
    if "_steps" not in stem:
        return -1
    prefix, _suffix = stem.rsplit("_steps", 1)
    try:
        return int(prefix.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def _find_latest_checkpoint(checkpoint_directory: Path, *, graph: bool) -> Path | None:
    """Return the newest checkpoint in a directory, ordered by encoded step count."""
    suffix = ".pt" if graph else ".zip"
    candidates = [
        path
        for path in checkpoint_directory.glob(f"model_checkpoint_*_steps{suffix}")
        if _extract_checkpoint_steps(path) >= 0
    ]
    if not candidates:
        return None
    return max(candidates, key=_extract_checkpoint_steps)


def _gnn_training_kwargs(
    *,
    iterations: int | None,
    steps: int | None,
    num_epochs: int | None,
    minibatch_size: int | None,
    hidden_dim: int | None,
    num_conv_wo_resnet: int | None,
    num_resnet_layers: int | None,
    dropout_p: float | None,
    bidirectional: bool | None,
    lr: float | None,
    gnn_lr: float | None,
) -> dict[str, object]:
    """Build the optional GNN training kwargs expected by ``Predictor.train_model``."""
    train_kwargs: dict[str, object] = {}
    if iterations is not None:
        train_kwargs["iterations"] = iterations
    if steps is not None:
        train_kwargs["steps"] = steps
    if num_epochs is not None:
        train_kwargs["num_epochs"] = num_epochs
    if minibatch_size is not None:
        train_kwargs["minibatch_size"] = minibatch_size
    if hidden_dim is not None:
        train_kwargs["hidden_dim"] = hidden_dim
    if num_conv_wo_resnet is not None:
        train_kwargs["num_conv_wo_resnet"] = num_conv_wo_resnet
    if num_resnet_layers is not None:
        train_kwargs["num_resnet_layers"] = num_resnet_layers
    if dropout_p is not None:
        train_kwargs["dropout_p"] = dropout_p
    if bidirectional is not None:
        train_kwargs["bidirectional"] = bidirectional
    if lr is not None:
        train_kwargs["lr"] = lr
    if gnn_lr is not None:
        train_kwargs["gnn_lr"] = gnn_lr
    return train_kwargs


def _resolve_total_timesteps_for_result(
    *,
    graph: bool,
    test: bool,
    timesteps: int,
    iterations: int | None,
    steps: int | None,
) -> int:
    """Return the total number of environment steps represented by the training run."""
    if not graph:
        return timesteps

    resolved_iterations = iterations if iterations is not None else (10 if test else 1000)
    resolved_steps = steps if steps is not None else (20 if test else 2048)
    return resolved_iterations * resolved_steps


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
    reward_scale: float = 1.0,
    no_effect_penalty: float = -0.001,
    max_episode_steps: int | None = None,
    graph: bool = False,
    iterations: int | None = None,
    steps: int | None = None,
    num_epochs: int | None = None,
    minibatch_size: int | None = None,
    hidden_dim: int | None = None,
    num_conv_wo_resnet: int | None = None,
    num_resnet_layers: int | None = None,
    dropout_p: float | None = None,
    bidirectional: bool | None = None,
    lr: float | None = None,
    gnn_lr: float | None = None,
    callback: BaseCallback | None = None,
) -> RLTrainingResult:
    """Train an RL predictor on an existing training split."""
    training_directory = (
        Path(path_training_circuits) if path_training_circuits is not None else get_path_training_circuits_train()
    )
    resolved_checkpoint_directory = Path(checkpoint_directory) if checkpoint_directory is not None else None
    resolved_resume_from_checkpoint = (
        Path(resume_from_checkpoint)
        if resume_from_checkpoint is not None
        else (
            _find_latest_checkpoint(resolved_checkpoint_directory, graph=graph)
            if resolved_checkpoint_directory is not None
            else None
        )
    )
    gnn_train_kwargs = _gnn_training_kwargs(
        iterations=iterations,
        steps=steps,
        num_epochs=num_epochs,
        minibatch_size=minibatch_size,
        hidden_dim=hidden_dim,
        num_conv_wo_resnet=num_conv_wo_resnet,
        num_resnet_layers=num_resnet_layers,
        dropout_p=dropout_p,
        bidirectional=bidirectional,
        lr=lr,
        gnn_lr=gnn_lr,
    )

    predictor = Predictor(
        figure_of_merit=figure_of_merit,
        device=device,
        mdp=mdp,
        path_training_circuits=training_directory,
        reward_scale=reward_scale,
        no_effect_penalty=no_effect_penalty,
        max_episode_steps=max_episode_steps,
        graph=graph,
    )
    callbacks: list[BaseCallback] = []
    if (
        not graph
        and resolved_checkpoint_directory is not None
        and checkpoint_frequency is not None
        and checkpoint_frequency > 0
    ):
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
        callback=CallbackList(callbacks) if callbacks and not graph else None,
        checkpoint_directory=resolved_checkpoint_directory,
        checkpoint_frequency=checkpoint_frequency,
        resume_from=resolved_resume_from_checkpoint,
        **gnn_train_kwargs,
    )

    model_name = (
        f"gnn_{figure_of_merit}_{device.description}.pt"
        if graph
        else f"model_{figure_of_merit}_{device.description}.zip"
    )
    model_path = get_path_trained_model() / model_name
    if graph:
        total_timesteps = _resolve_total_timesteps_for_result(
            graph=graph,
            test=test,
            timesteps=timesteps,
            iterations=iterations,
            steps=steps,
        )
    else:
        assert trained_model is not None
        total_timesteps = int(trained_model.num_timesteps)
    return RLTrainingResult(
        model_path=model_path,
        training_directory=training_directory,
        total_timesteps=total_timesteps,
        checkpoint_directory=resolved_checkpoint_directory if not graph else None,
        resumed_from_checkpoint=resolved_resume_from_checkpoint if not graph else None,
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
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Train the graph-based GNN policy instead of MaskablePPO.",
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
        "--reward-scale",
        type=float,
        default=1.0,
        help="Scaling factor for reward deltas inside the predictor environment.",
    )
    parser.add_argument(
        "--no-effect-penalty",
        type=float,
        default=-0.001,
        help="Penalty applied when an action does not change the circuit.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Optional directory for periodic checkpoints. If present, the newest checkpoint is resumed automatically.",
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=None,
        help="Optional checkpoint frequency in environment steps.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint to resume training from. Overrides auto-discovery in --checkpoint-dir.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="Optional hard cap on environment steps per training episode.",
    )
    parser.add_argument("--iterations", type=int, default=None, help="GNN PPO iterations.")
    parser.add_argument("--steps", type=int, default=None, help="GNN PPO steps per iteration.")
    parser.add_argument("--num-epochs", type=int, default=None, help="GNN PPO update epochs.")
    parser.add_argument("--minibatch-size", type=int, default=None, help="GNN PPO minibatch size.")
    parser.add_argument("--hidden-dim", type=int, default=None, help="GNN hidden dimension.")
    parser.add_argument(
        "--num-conv-wo-resnet",
        type=int,
        default=None,
        help="Number of non-residual graph convolution layers in the GNN encoder.",
    )
    parser.add_argument(
        "--num-resnet-layers",
        type=int,
        default=None,
        help="Number of residual graph convolution layers in the GNN encoder.",
    )
    parser.add_argument("--dropout-p", type=float, default=None, help="GNN dropout probability.")
    parser.add_argument(
        "--bidirectional",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable bidirectional message passing in the GNN encoder.",
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate for PPO actor/critic heads.")
    parser.add_argument("--gnn-lr", type=float, default=None, help="Learning rate for the GNN encoder.")
    args = parser.parse_args()

    result = run_rl_training(
        device=get_device(args.device),
        figure_of_merit=args.figure_of_merit,
        mdp=args.mdp,
        timesteps=args.timesteps,
        verbose=args.verbose,
        test=args.test,
        path_training_circuits=args.train_dir,
        reward_scale=args.reward_scale,
        no_effect_penalty=args.no_effect_penalty,
        checkpoint_directory=args.checkpoint_dir,
        checkpoint_frequency=args.checkpoint_frequency,
        resume_from_checkpoint=args.resume_from_checkpoint,
        max_episode_steps=args.max_episode_steps,
        graph=args.graph,
        iterations=args.iterations,
        steps=args.steps,
        num_epochs=args.num_epochs,
        minibatch_size=args.minibatch_size,
        hidden_dim=args.hidden_dim,
        num_conv_wo_resnet=args.num_conv_wo_resnet,
        num_resnet_layers=args.num_resnet_layers,
        dropout_p=args.dropout_p,
        bidirectional=args.bidirectional,
        lr=args.lr,
        gnn_lr=args.gnn_lr,
    )

    print(f"Training directory: {result.training_directory}")
    print(f"Model: {result.model_path}")


if __name__ == "__main__":
    main()
