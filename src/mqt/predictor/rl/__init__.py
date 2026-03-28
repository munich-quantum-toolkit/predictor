# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Lazy exports for the RL functionality of MQT Predictor."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mqt.predictor.rl.data_generation import (
        GeneratedBenchmarkCircuit,
        TrainTestGenerationResult,
        TrainTestSplit,
        collect_working_benchmark_circuits,
        generate_rl_train_test_data,
    )
    from mqt.predictor.rl.evaluate import main as evaluate_main
    from mqt.predictor.rl.evaluation import (
        CircuitEvaluationResult,
        FeatureImportanceResult,
        FinalCircuitMetrics,
        PredictorEvaluationResult,
        compute_feature_importance,
        evaluate_trained_predictor,
    )
    from mqt.predictor.rl.predictor import Predictor, rl_compile
    from mqt.predictor.rl.predictorenv import PredictorEnv
    from mqt.predictor.rl.training import RLTrainingResult, run_rl_training
    from mqt.predictor.rl.workflow import RLWorkflowResult, run_rl_training_workflow

__all__ = [
    "CircuitEvaluationResult",
    "FeatureImportanceResult",
    "FinalCircuitMetrics",
    "GeneratedBenchmarkCircuit",
    "Predictor",
    "PredictorEnv",
    "PredictorEvaluationResult",
    "RLTrainingResult",
    "RLWorkflowResult",
    "TrainTestGenerationResult",
    "TrainTestSplit",
    "collect_working_benchmark_circuits",
    "compute_feature_importance",
    "evaluate_main",
    "evaluate_trained_predictor",
    "generate_rl_train_test_data",
    "rl_compile",
    "run_rl_training",
    "run_rl_training_workflow",
]

_NAME_TO_MODULE = {
    "CircuitEvaluationResult": "mqt.predictor.rl.evaluation",
    "FeatureImportanceResult": "mqt.predictor.rl.evaluation",
    "FinalCircuitMetrics": "mqt.predictor.rl.evaluation",
    "GeneratedBenchmarkCircuit": "mqt.predictor.rl.data_generation",
    "Predictor": "mqt.predictor.rl.predictor",
    "PredictorEvaluationResult": "mqt.predictor.rl.evaluation",
    "PredictorEnv": "mqt.predictor.rl.predictorenv",
    "evaluate_main": "mqt.predictor.rl.evaluate",
    "RLTrainingResult": "mqt.predictor.rl.training",
    "RLWorkflowResult": "mqt.predictor.rl.workflow",
    "TrainTestGenerationResult": "mqt.predictor.rl.data_generation",
    "TrainTestSplit": "mqt.predictor.rl.data_generation",
    "collect_working_benchmark_circuits": "mqt.predictor.rl.data_generation",
    "compute_feature_importance": "mqt.predictor.rl.evaluation",
    "evaluate_trained_predictor": "mqt.predictor.rl.evaluation",
    "generate_rl_train_test_data": "mqt.predictor.rl.data_generation",
    "run_rl_training_workflow": "mqt.predictor.rl.workflow",
    "run_rl_training": "mqt.predictor.rl.training",
    "rl_compile": "mqt.predictor.rl.predictor",
}


def __getattr__(name: str) -> object:
    """Lazily import RL symbols on first access."""
    if name not in _NAME_TO_MODULE:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)

    module = import_module(_NAME_TO_MODULE[name])
    return getattr(module, name)
