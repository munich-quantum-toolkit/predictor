# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""End-to-end workflow for RL data generation, training, and evaluation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mqt.bench.targets import get_device

from mqt.predictor.rl.data_generation import TrainTestGenerationResult, generate_rl_train_test_data
from mqt.predictor.rl.evaluation import evaluate_trained_predictor
from mqt.predictor.rl.helper import (
    ensure_training_circuit_directories,
    get_path_trained_model,
)
from mqt.predictor.rl.predictor import Predictor

if TYPE_CHECKING:
    from qiskit.transpiler import Target

    from mqt.predictor.reward import figure_of_merit
    from mqt.predictor.rl.evaluation import PredictorEvaluationResult


@dataclass(slots=True)
class RLWorkflowResult:
    """Result of the full RL workflow."""

    dataset: TrainTestGenerationResult
    model_path: Path
    evaluation: PredictorEvaluationResult


def load_existing_train_test_data(
    training_directory: Path,
    test_directory: Path,
) -> TrainTestGenerationResult:
    """Load an already generated RL train/test split from disk."""
    train_circuits = sorted(training_directory.glob("*.qasm"))
    test_circuits = sorted(test_directory.glob("*.qasm"))
    if not train_circuits:
        msg = f"No training circuits found in '{training_directory}'."
        raise FileNotFoundError(msg)
    if not test_circuits:
        msg = f"No test circuits found in '{test_directory}'."
        raise FileNotFoundError(msg)

    benchmark_names = sorted({
        path.stem.removesuffix("_indep").rsplit("_", 1)[0] for path in train_circuits + test_circuits
    })
    return TrainTestGenerationResult(
        training_directory=training_directory,
        test_directory=test_directory,
        train_circuits=train_circuits,
        test_circuits=test_circuits,
        staging_directory=training_directory.parent / "_generated_all",
        benchmark_names=benchmark_names,
    )


def run_rl_training_workflow(
    device: Target,
    figure_of_merit: figure_of_merit = "expected_fidelity",
    benchmark_names: list[str] | None = None,
    min_qubits: int = 2,
    max_qubits: int = 20,
    test_fraction: float = 0.1,
    seed: int = 0,
    timesteps: int = 1000,
    verbose: int = 2,
    test: bool = False,
    max_eval_steps: int = 200,
    deterministic: bool = False,
    path_training_circuits: str | Path | None = None,
    path_test_circuits: str | Path | None = None,
    reuse_existing_data: bool = False,
) -> RLWorkflowResult:
    """Generate data, train an RL predictor, and evaluate it on the held-out split."""
    default_train_directory, default_test_directory = ensure_training_circuit_directories()
    training_directory = Path(path_training_circuits) if path_training_circuits is not None else default_train_directory
    test_directory = Path(path_test_circuits) if path_test_circuits is not None else default_test_directory

    if reuse_existing_data:
        dataset = load_existing_train_test_data(training_directory, test_directory)
    else:
        dataset = generate_rl_train_test_data(
            path_training_circuits=training_directory,
            path_test_circuits=test_directory,
            benchmark_names=benchmark_names,
            min_qubits=min_qubits,
            max_qubits=max_qubits,
            test_fraction=test_fraction,
            seed=seed,
        )

    predictor = Predictor(
        figure_of_merit=figure_of_merit,
        device=device,
        path_training_circuits=dataset.training_directory,
    )
    predictor.train_model(
        timesteps=timesteps,
        verbose=verbose,
        test=test,
    )
    print("Training completed. Starting evaluation...")

    model_path = get_path_trained_model() / f"model_{figure_of_merit}_{device.description}.zip"
    evaluation = evaluate_trained_predictor(
        model_path=model_path,
        device=device,
        figure_of_merit=figure_of_merit,
        path_training_circuits=dataset.training_directory,
        path_test_circuits=dataset.test_directory,
        max_steps=max_eval_steps,
        deterministic=deterministic,
        seed=seed,
    )

    return RLWorkflowResult(
        dataset=dataset,
        model_path=model_path,
        evaluation=evaluation,
    )


def main() -> None:
    """Run the RL workflow from the command line."""
    parser = argparse.ArgumentParser(description="Generate RL data, train a model, and evaluate it.")
    parser.add_argument("--device", required=True, help="Target device name, e.g. ibm_washington.")
    parser.add_argument(
        "--figure-of-merit",
        default="expected_fidelity",
        choices=["expected_fidelity", "critical_depth", "estimated_success_probability"],
        help="Figure of merit used for RL training and evaluation.",
    )
    parser.add_argument(
        "--benchmarks",
        default=None,
        help="Comma-separated benchmark names. If omitted, all available working benchmarks are used.",
    )
    parser.add_argument("--min-qubits", type=int, default=2, help="Minimum benchmark size to generate.")
    parser.add_argument("--max-qubits", type=int, default=20, help="Maximum benchmark size to generate.")
    parser.add_argument("--test-fraction", type=float, default=0.1, help="Fraction of circuits used for testing.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for train/test split and evaluation.")
    parser.add_argument("--timesteps", type=int, default=1000, help="Training timesteps.")
    parser.add_argument("--verbose", type=int, default=2, help="Training verbosity passed to PPO.")
    parser.add_argument("--max-eval-steps", type=int, default=200, help="Maximum steps during evaluation rollout.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy inference during evaluation. Default matches original stochastic behavior.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use the lightweight test training configuration.",
    )
    parser.add_argument(
        "--reuse-existing-data",
        action="store_true",
        help="Skip data generation and reuse the existing train/test split from --train-dir and --test-dir.",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=None,
        help="Optional path for training circuits.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="Optional path for test circuits.",
    )
    args = parser.parse_args()

    benchmark_names = None
    if args.benchmarks:
        benchmark_names = [name.strip() for name in args.benchmarks.split(",") if name.strip()]

    result = run_rl_training_workflow(
        device=get_device(args.device),
        figure_of_merit=args.figure_of_merit,
        benchmark_names=benchmark_names,
        min_qubits=args.min_qubits,
        max_qubits=args.max_qubits,
        test_fraction=args.test_fraction,
        seed=args.seed,
        timesteps=args.timesteps,
        verbose=args.verbose,
        test=args.test,
        max_eval_steps=args.max_eval_steps,
        deterministic=args.deterministic,
        path_training_circuits=args.train_dir,
        path_test_circuits=args.test_dir,
        reuse_existing_data=args.reuse_existing_data,
    )

    print(f"Model: {result.model_path}")
    print(f"Train circuits: {len(result.dataset.train_circuits)}")
    print(f"Test circuits: {len(result.dataset.test_circuits)}")
    print(f"Average metrics: {result.evaluation.average_metrics}")
    print(
        "Grouped feature importance:",
        result.evaluation.feature_importance.average_original_feature_importance,
        result.evaluation.feature_importance.average_gate_count_feature_importance,
    )


if __name__ == "__main__":
    main()
