# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Evaluation-only entry point for RL predictor models."""

from __future__ import annotations

import argparse
from pathlib import Path

from mqt.bench.targets import get_device

from mqt.predictor.rl.experiments.evaluation import evaluate_trained_predictor
from mqt.predictor.rl.helper import (
    get_path_trained_model,
    get_path_training_circuits_test,
    get_path_training_circuits_train,
)


def main() -> None:
    """Run RL evaluation from the command line."""
    parser = argparse.ArgumentParser(description="Evaluate a trained RL predictor on an existing test split.")
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
        help="Figure of merit used for RL evaluation.",
    )
    parser.add_argument(
        "--mdp",
        default="paper",
        choices=["paper", "flexible", "thesis", "hybrid"],
        help="MDP transition policy used by the RL environment.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional path to the trained model. Defaults to the standard RL model location.",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=None,
        help="Optional path to the training circuits directory.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="Optional path to the test circuits directory.",
    )
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum steps during evaluation rollout.")
    parser.add_argument("--seed", type=int, default=0, help="Evaluation seed.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy inference during evaluation.",
    )
    args = parser.parse_args()

    device = get_device(args.device)
    model_path = args.model_path or (
        get_path_trained_model() / f"model_{args.figure_of_merit}_{device.description}.zip"
    )
    train_dir = args.train_dir or get_path_training_circuits_train()
    test_dir = args.test_dir or get_path_training_circuits_test()

    result = evaluate_trained_predictor(
        model_path=model_path,
        device=device,
        figure_of_merit=args.figure_of_merit,
        mdp=args.mdp,
        path_training_circuits=train_dir,
        path_test_circuits=test_dir,
        max_steps=args.max_steps,
        deterministic=args.deterministic,
        seed=args.seed,
    )

    print(f"Model: {model_path}")
    print(f"Test directory: {result.test_directory}")
    print(f"Evaluated circuits: {len(result.circuits)}")
    print(f"Average metrics: {result.average_metrics}")
    print(
        "Grouped feature importance:",
        result.feature_importance.average_original_feature_importance,
        result.feature_importance.average_gate_count_feature_importance,
    )
    print(
        "Overall action effectiveness:",
        f"{result.action_effectiveness.total_effective_uses}/{result.action_effectiveness.total_uses}",
        f"({result.action_effectiveness.overall_effectiveness_ratio:.1%})",
    )
    print("Per-action effectiveness:")
    for stats in result.action_effectiveness.per_action:
        print(f"  {stats.action_name}: {stats.effective_uses}/{stats.total_uses} ({stats.effectiveness_ratio:.1%})")


if __name__ == "__main__":
    main()
