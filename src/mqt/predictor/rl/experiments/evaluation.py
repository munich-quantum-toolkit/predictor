# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Utilities for evaluating trained RL predictors on held-out test circuits."""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.transpiler.exceptions import TranspilerError
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sklearn.metrics import mutual_info_score
from torch.distributions import Categorical
from torch_geometric.data import Batch, Data

from mqt.predictor.reward import esp_data_available, estimated_success_probability, expected_fidelity
from mqt.predictor.rl.actions import PassType
from mqt.predictor.rl.gnn_ppo import create_gnn_policy
from mqt.predictor.rl.helper import (
    FLAT_RL_FEATURE_NAMES,
    get_path_training_circuits,
    get_path_training_circuits_test,
    get_path_training_circuits_train,
    predicted_action_to_index,
)
from mqt.predictor.rl.predictor import Predictor
from mqt.predictor.utils import get_openqasm_gates_for_rl

if TYPE_CHECKING:
    from collections.abc import Callable

    from qiskit.transpiler import Target

    from mqt.predictor.reward import figure_of_merit
    from mqt.predictor.rl.actions import Action
    from mqt.predictor.rl.gnn import SAGEActorCritic


logger = logging.getLogger("mqt-predictor")


NON_GATE_OBSERVATION_FEATURES = [
    "num_qubits",
    "depth",
    "program_communication",
    "critical_depth",
    "entanglement_ratio",
    "parallelism",
    "liveness",
    "measure",
]
FeatureValue = int | NDArray[np.float32]
Observation = dict[str, FeatureValue]
PolicyObservation = dict[str, NDArray[np.float32]]
CompilationStateFlags = tuple[bool, bool, bool]


@dataclass(slots=True)
class FinalCircuitMetrics:
    """Final metrics of a compiled circuit."""

    expected_fidelity: float | None
    estimated_success_probability: float | None
    depth: int
    size: int


@dataclass(slots=True)
class CircuitEvaluationResult:
    """Evaluation result for a single circuit."""

    circuit_name: str
    figure_of_merit_value: float
    figure_of_merit_kind: str
    metrics: FinalCircuitMetrics
    used_compilation_passes: list[str]
    effective_compilation_passes: list[str]
    effective_steps: int
    terminated: bool
    hit_step_limit: bool
    decision_trace: list[DecisionSample] = field(default_factory=list)


@dataclass(slots=True)
class DecisionSample:
    """One `(action, feature-vector)` sample collected during evaluation."""

    action_name: str
    features: dict[str, float]


@dataclass(slots=True)
class FeatureImportanceResult:
    """Mutual-information feature-importance summary."""

    baseline_mean_reward: float
    per_feature: dict[str, float]
    average_original_feature_importance: float
    average_gate_count_feature_importance: float


@dataclass(slots=True)
class ActionEffectivenessStats:
    """Aggregated usage/effectiveness statistics for one compilation action."""

    action_name: str
    total_uses: int
    effective_uses: int
    effectiveness_ratio: float


@dataclass(slots=True)
class ActionEffectivenessSummary:
    """Aggregated action-effectiveness statistics over the full test set."""

    total_uses: int
    total_effective_uses: int
    overall_effectiveness_ratio: float
    per_action: list[ActionEffectivenessStats]


@dataclass(slots=True)
class PredictorEvaluationResult:
    """Full evaluation result for a trained predictor."""

    test_directory: Path
    circuits: list[CircuitEvaluationResult]
    average_metrics: FinalCircuitMetrics
    feature_importance: FeatureImportanceResult
    action_effectiveness: ActionEffectivenessSummary


@dataclass(slots=True)
class LoadedPolicy:
    """Loaded evaluation policy together with its observation mode."""

    model: MaskablePPO | SAGEActorCritic
    graph: bool
    model_path: Path
    torch_device: str | None = None


def get_non_gate_observation_features() -> list[str]:
    """Return RL observation features that are not gate-count features."""
    return NON_GATE_OBSERVATION_FEATURES.copy()


def get_all_observation_features() -> list[str]:
    """Return all supported RL observation features."""
    return get_non_gate_observation_features() + get_openqasm_gates_for_rl()


def split_feature_groups(features: list[str] | None = None) -> tuple[list[str], list[str]]:
    """Split feature names into non-gate and gate-count groups."""
    feature_list = features or get_all_observation_features()
    gate_count_features = set(get_openqasm_gates_for_rl())
    original_features = [feature for feature in feature_list if feature not in gate_count_features]
    gate_features = [feature for feature in feature_list if feature in gate_count_features]
    return original_features, gate_features


def is_effective_action(
    action: Action,
    reward_value: float,
    previous_state_flags: CompilationStateFlags,
    current_state_flags: CompilationStateFlags,
) -> bool:
    """Return whether a step should count as effective for evaluation."""
    if action.pass_type == PassType.TERMINATE:
        return False
    previous_synthesized, previous_laid_out, previous_routed = previous_state_flags
    current_synthesized, current_laid_out, current_routed = current_state_flags

    if action.pass_type == PassType.SYNTHESIS:
        return not previous_synthesized and current_synthesized
    if action.pass_type == PassType.LAYOUT:
        return not previous_laid_out and current_laid_out
    if action.pass_type == PassType.ROUTING:
        return not previous_routed and current_routed
    if action.pass_type == PassType.MAPPING:
        return any(
            not before and after for before, after in zip(previous_state_flags, current_state_flags, strict=True)
        )
    return reward_value > 0.0


def evaluate_trained_predictor(
    model_path: str | Path,
    device: Target,
    figure_of_merit: figure_of_merit = "expected_fidelity",
    mdp: str = "paper",
    path_training_circuits: str | Path | None = None,
    path_test_circuits: str | Path | None = None,
    max_steps: int = 200,
    num_seeds: int = 1,
    graph: bool | None = None,
) -> PredictorEvaluationResult:
    """Evaluate a trained predictor on a held-out test set and compute feature importance.

    The default test-set lookup assumes the split layout:
    ``training_circuits/test/*.qasm``.
    Each circuit is rolled out ``num_seeds`` times with different random seeds and
    the results are averaged, which gives a more reliable estimate for stochastic policies.
    """
    training_dir = (
        Path(path_training_circuits) if path_training_circuits is not None else get_path_training_circuits_train()
    )
    test_dir = resolve_test_circuit_directory(training_dir, path_test_circuits)
    circuits = load_test_circuits(test_dir)
    resolved_model_path, resolved_graph = resolve_model_path_and_kind(model_path, graph=graph)

    predictor = Predictor(
        figure_of_merit=figure_of_merit,
        device=device,
        mdp=mdp,
        path_training_circuits=training_dir,
        graph=resolved_graph,
    )
    model = load_model_from_path(resolved_model_path, predictor=predictor, graph=resolved_graph)

    evaluation_results = [
        rollout_circuit(
            predictor=predictor,
            model=model,
            qc=qc,
            max_steps=max_steps,
            num_seeds=num_seeds,
        )
        for qc in circuits
    ]

    feature_importance = compute_feature_importance(
        predictor=predictor,
        model=model,
        circuits=circuits,
        baseline_results=evaluation_results,
        max_steps=max_steps,
        num_seeds=num_seeds,
    )

    return PredictorEvaluationResult(
        test_directory=test_dir,
        circuits=evaluation_results,
        average_metrics=compute_average_metrics(evaluation_results),
        feature_importance=feature_importance,
        action_effectiveness=compute_action_effectiveness_summary(evaluation_results),
    )


def compute_feature_importance(
    predictor: Predictor,
    model: LoadedPolicy,
    circuits: list[QuantumCircuit],
    baseline_results: list[CircuitEvaluationResult] | None = None,
    features: list[str] | None = None,
    max_steps: int = 200,
    num_seeds: int = 1,
) -> FeatureImportanceResult:
    """Compute feature importance as mutual information with the chosen action."""
    feature_names = features or get_all_observation_features()
    if baseline_results is None:
        baseline_results = [
            rollout_circuit(
                predictor=predictor,
                model=model,
                qc=qc,
                max_steps=max_steps,
                num_seeds=num_seeds,
            )
            for qc in circuits
        ]

    baseline_rewards = np.array([result.figure_of_merit_value for result in baseline_results], dtype=float)
    baseline_mean = nanmean_or_nan(baseline_rewards)
    per_feature: dict[str, float] = {}
    decision_samples = [sample for result in baseline_results for sample in result.decision_trace]
    action_labels = [sample.action_name for sample in decision_samples]

    for feature in feature_names:
        feature_values = [sample.features[feature] for sample in decision_samples if feature in sample.features]
        if len(feature_values) != len(action_labels):
            msg = f"Feature '{feature}' is missing from some evaluation samples."
            raise KeyError(msg)
        per_feature[feature] = estimate_mutual_information(action_labels, feature_values)

    original_features, gate_features = split_feature_groups(feature_names)

    return FeatureImportanceResult(
        baseline_mean_reward=baseline_mean,
        per_feature=per_feature,
        average_original_feature_importance=nanmean_or_nan([per_feature[feature] for feature in original_features]),
        average_gate_count_feature_importance=nanmean_or_nan([per_feature[feature] for feature in gate_features]),
    )


def _single_rollout(
    predictor: Predictor,
    model: LoadedPolicy,
    qc: QuantumCircuit,
    max_steps: int,
    seed: int,
) -> CircuitEvaluationResult:
    """Run one stochastic rollout of the policy on a circuit with the given seed."""
    predictor.env.episode_count = 0
    obs, _ = predictor.env.reset(qc, seed=seed)

    used_compilation_passes: list[str] = []
    effective_compilation_passes: list[str] = []
    effective_steps = 0
    terminated = False
    truncated = False
    hit_step_limit = False
    step_count = 0
    decision_trace: list[DecisionSample] = []

    while not (terminated or truncated) and step_count < max_steps:
        action_index, decision_features = predict_action_with_features(
            predictor=predictor,
            model=model,
            obs=obs,
        )
        action_item = predictor.env.action_set[action_index]
        used_compilation_passes.append(action_item.name)
        previous_state_flags = predictor.env.get_compilation_state_flags()

        obs, reward_value, terminated, truncated, _ = predictor.env.step(action_index)
        current_state_flags = predictor.env.get_compilation_state_flags()
        if is_effective_action(action_item, reward_value, previous_state_flags, current_state_flags):
            effective_compilation_passes.append(action_item.name)
            effective_steps += 1
            decision_trace.append(
                DecisionSample(
                    action_name=action_item.name,
                    features=decision_features,
                )
            )
        step_count += 1

        if not (terminated or truncated) and step_count >= max_steps:
            hit_step_limit = True
            if predictor.env.action_terminate_index in predictor.env.valid_actions:
                _obs, _reward_value, terminated, truncated, _ = predictor.env.step(
                    predictor.env.action_terminate_index
                )

    figure_of_merit_value, figure_of_merit_kind = predictor.env.calculate_reward(mode="auto")

    return CircuitEvaluationResult(
        circuit_name=predictor.env.current_circuit_name,
        figure_of_merit_value=figure_of_merit_value,
        figure_of_merit_kind=figure_of_merit_kind,
        metrics=collect_final_metrics(predictor.env.state, predictor.env.device),
        used_compilation_passes=used_compilation_passes,
        effective_compilation_passes=effective_compilation_passes,
        effective_steps=effective_steps,
        terminated=terminated,
        hit_step_limit=hit_step_limit,
        decision_trace=decision_trace,
    )


def _average_rollout_results(results: list[CircuitEvaluationResult]) -> CircuitEvaluationResult:
    """Merge multiple same-circuit rollout results by averaging numeric fields."""
    if len(results) == 1:
        return results[0]
    return CircuitEvaluationResult(
        circuit_name=results[0].circuit_name,
        figure_of_merit_value=float(np.nanmean([r.figure_of_merit_value for r in results])),
        figure_of_merit_kind=results[0].figure_of_merit_kind,
        metrics=FinalCircuitMetrics(
            expected_fidelity=optional_nanmean([r.metrics.expected_fidelity for r in results]),
            estimated_success_probability=optional_nanmean([
                r.metrics.estimated_success_probability for r in results
            ]),
            depth=round(nanmean_or_nan([r.metrics.depth for r in results])),
            size=round(nanmean_or_nan([r.metrics.size for r in results])),
        ),
        used_compilation_passes=[p for r in results for p in r.used_compilation_passes],
        effective_compilation_passes=[p for r in results for p in r.effective_compilation_passes],
        effective_steps=round(nanmean_or_nan([r.effective_steps for r in results])),
        terminated=any(r.terminated for r in results),
        hit_step_limit=any(r.hit_step_limit for r in results),
        decision_trace=[s for r in results for s in r.decision_trace],
    )


def rollout_circuit(
    predictor: Predictor,
    model: LoadedPolicy,
    qc: QuantumCircuit,
    max_steps: int = 200,
    num_seeds: int = 1,
) -> CircuitEvaluationResult:
    """Run the trained policy on a circuit and collect final metrics.

    When ``num_seeds > 1``, the circuit is rolled out that many times with seeds
    ``0 … num_seeds-1`` and numeric results are averaged, giving a lower-variance
    estimate for stochastic policies.
    """
    logger.info("Evaluating circuit=%s (num_seeds=%d)", qc.name or "<unnamed>", num_seeds)
    results = [_single_rollout(predictor, model, qc, max_steps, seed=seed) for seed in range(num_seeds)]
    return _average_rollout_results(results)


def predict_action_with_features(
    predictor: Predictor,
    model: LoadedPolicy,
    obs: object,
) -> tuple[int, dict[str, float]]:
    """Predict the next action stochastically and return the flat feature trace used for analysis."""
    if model.graph:
        if not isinstance(obs, Data):
            msg = f"Expected a graph observation for GNN evaluation, received {type(obs).__name__}."
            raise TypeError(msg)
        assert model.torch_device is not None
        batch_obs = Batch.from_data_list([obs]).to(model.torch_device)
        action_mask = torch.as_tensor(predictor.env.action_masks(), dtype=torch.bool, device=model.torch_device)
        with torch.no_grad():
            logits, _value = model.model(batch_obs)
            logits = logits.masked_fill(~action_mask.unsqueeze(0), float("-inf"))
            action_index = int(Categorical(logits=logits.squeeze(0)).sample().item())
        return action_index, flatten_graph_observation(obs)

    policy_obs = clone_observation(obs)
    action_masks = get_action_masks(predictor.env)
    action, _ = model.model.predict(policy_obs, action_masks=action_masks, deterministic=False)
    action_index = predicted_action_to_index(action)
    return action_index, flatten_observation(policy_obs)


def clone_observation(obs: object) -> PolicyObservation:
    """Return a detached copy of an observation dictionary for policy inference."""
    if not isinstance(obs, dict):
        msg = f"Expected a flat observation dictionary, received {type(obs).__name__}."
        raise TypeError(msg)

    cloned: PolicyObservation = {}
    for key, value in obs.items():
        if not isinstance(key, str):
            msg = f"Expected string observation keys, received {type(key).__name__}."
            raise TypeError(msg)
        cloned[key] = clone_feature_value(value)
    return cloned


def clone_feature_value(value: object) -> NDArray[np.float32]:
    """Clone a feature value into a float32 array for policy inference."""
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True, dtype=np.float32)
    if not isinstance(value, int):
        msg = f"Expected int or ndarray observation values, received {type(value).__name__}."
        raise TypeError(msg)
    return np.array([value], dtype=np.float32)


def flatten_observation(obs: PolicyObservation) -> dict[str, float]:
    """Flatten an observation into scalar feature values."""
    flattened: dict[str, float] = {}
    for key, value in obs.items():
        flattened[key] = float(value[0])
    return flattened


def flatten_graph_observation(obs: Data) -> dict[str, float]:
    """Flatten graph observations to the same feature names used by flat PPO evaluation."""
    global_features = getattr(obs, "global_features", None)
    if global_features is None:
        msg = "Graph observation is missing global_features."
        raise KeyError(msg)
    feature_values = torch.as_tensor(global_features, dtype=torch.float32).reshape(-1).cpu().numpy()
    if feature_values.size != len(FLAT_RL_FEATURE_NAMES):
        msg = (
            "Graph observation global_features have unexpected size: "
            f"{feature_values.size} != {len(FLAT_RL_FEATURE_NAMES)}."
        )
        raise ValueError(msg)
    return {name: float(value) for name, value in zip(FLAT_RL_FEATURE_NAMES, feature_values, strict=True)}


def estimate_mutual_information(action_labels: list[str], feature_values: list[float], num_bins: int = 10) -> float:
    """Estimate mutual information between a scalar feature and the chosen action."""
    if len(action_labels) != len(feature_values):
        msg = "Action and feature sample lengths must match."
        raise ValueError(msg)
    if not action_labels:
        return 0.0
    discretized_values = discretize_feature_values(feature_values, num_bins=num_bins)
    return float(mutual_info_score(action_labels, discretized_values))


def discretize_feature_values(feature_values: list[float], num_bins: int = 10) -> list[int]:
    """Discretize scalar feature values for mutual-information estimation."""
    values = np.asarray(feature_values, dtype=float)
    if values.size == 0:
        return []
    if np.isnan(values).all() or np.allclose(values, values[0], equal_nan=True):
        return [0] * int(values.size)

    integer_like = np.allclose(values, np.round(values), equal_nan=True)
    unique_values = np.unique(values[~np.isnan(values)])
    if integer_like and unique_values.size <= num_bins:
        mapping = {float(value): index for index, value in enumerate(sorted(float(v) for v in unique_values))}
        return [mapping.get(float(value), 0) for value in values]

    quantile_edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, num_bins + 1)))
    if quantile_edges.size <= 2:
        return [0] * int(values.size)
    bin_edges = quantile_edges[1:-1]
    return np.digitize(values, bin_edges, right=False).astype(int).tolist()


def collect_final_metrics(qc: QuantumCircuit, device: Target) -> FinalCircuitMetrics:
    """Collect exact final metrics for a compiled circuit when available."""
    return FinalCircuitMetrics(
        expected_fidelity=safe_metric(lambda: expected_fidelity(qc, device)),
        estimated_success_probability=(
            safe_metric(lambda: estimated_success_probability(qc, device)) if esp_data_available(device) else None
        ),
        depth=int(qc.depth()),
        size=int(qc.size()),
    )


def compute_average_metrics(results: list[CircuitEvaluationResult]) -> FinalCircuitMetrics:
    """Compute average final metrics over all evaluated circuits."""
    return FinalCircuitMetrics(
        expected_fidelity=optional_nanmean([result.metrics.expected_fidelity for result in results]),
        estimated_success_probability=optional_nanmean([
            result.metrics.estimated_success_probability for result in results
        ]),
        depth=round(nanmean_or_nan([result.metrics.depth for result in results])),
        size=round(nanmean_or_nan([result.metrics.size for result in results])),
    )


def compute_action_effectiveness_summary(results: list[CircuitEvaluationResult]) -> ActionEffectivenessSummary:
    """Aggregate action usage/effectiveness statistics over all evaluated circuits."""
    total_counter: Counter[str] = Counter()
    effective_counter: Counter[str] = Counter()

    for result in results:
        total_counter.update(p for p in result.used_compilation_passes if p != "terminate")
        effective_counter.update(result.effective_compilation_passes)

    action_names = sorted(total_counter, key=lambda name: (-total_counter[name], name))
    per_action = [
        ActionEffectivenessStats(
            action_name=action_name,
            total_uses=total_counter[action_name],
            effective_uses=effective_counter[action_name],
            effectiveness_ratio=(
                effective_counter[action_name] / total_counter[action_name] if total_counter[action_name] > 0 else 0.0
            ),
        )
        for action_name in action_names
    ]

    total_uses = sum(total_counter.values())
    total_effective_uses = sum(effective_counter.values())
    overall_effectiveness_ratio = total_effective_uses / total_uses if total_uses > 0 else 0.0

    return ActionEffectivenessSummary(
        total_uses=total_uses,
        total_effective_uses=total_effective_uses,
        overall_effectiveness_ratio=overall_effectiveness_ratio,
        per_action=per_action,
    )


def resolve_test_circuit_directory(
    path_training_circuits: Path,
    path_test_circuits: str | Path | None = None,
) -> Path:
    """Resolve the held-out test-circuit directory."""
    if path_test_circuits is not None:
        test_dir = Path(path_test_circuits)
    else:
        candidates = [
            get_path_training_circuits_test(),
            path_training_circuits.parent / "test"
            if path_training_circuits.name == "train"
            else path_training_circuits / "test",
            path_training_circuits / "new_indep_circuits" / "special_test",
            path_training_circuits / "special_test",
            path_training_circuits.parent / "special_test",
            get_path_training_circuits() / "new_indep_circuits" / "special_test",
        ]
        test_dir = next((candidate for candidate in candidates if candidate.exists()), candidates[0])

    if not test_dir.exists():
        msg = f"Test circuit directory '{test_dir}' does not exist."
        raise FileNotFoundError(msg)
    return test_dir


def load_test_circuits(test_dir: Path) -> list[QuantumCircuit]:
    """Load all QASM circuits from the test directory."""
    qasm_files = sorted(test_dir.glob("*.qasm"))
    if not qasm_files:
        msg = f"No QASM test circuits found in '{test_dir}'."
        raise FileNotFoundError(msg)
    return [load_qasm_circuit(path) for path in qasm_files]


def load_qasm_circuit(path: Path) -> QuantumCircuit:
    """Load a QASM circuit from disk."""
    qc = QuantumCircuit.from_qasm_file(str(path))
    qc.name = path.stem
    return qc


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


def _latest_checkpoint_in_directory(directory: Path, *, suffix: str) -> Path | None:
    """Return the newest checkpoint matching a suffix inside a directory."""
    candidates = [
        path for path in directory.glob(f"model_checkpoint_*_steps{suffix}") if _extract_checkpoint_steps(path) >= 0
    ]
    if not candidates:
        return None
    return max(candidates, key=_extract_checkpoint_steps)


def resolve_model_path_and_kind(model_path: str | Path, *, graph: bool | None = None) -> tuple[Path, bool]:
    """Resolve a checkpoint/model path and infer whether it belongs to graph evaluation."""
    resolved_model_path = Path(model_path)
    if resolved_model_path.is_dir():
        if graph is True:
            graph_checkpoint = _latest_checkpoint_in_directory(resolved_model_path, suffix=".pt")
            if graph_checkpoint is None:
                msg = f"No GNN checkpoints found in '{resolved_model_path}'."
                raise FileNotFoundError(msg)
            return graph_checkpoint, True
        if graph is False:
            ppo_checkpoint = _latest_checkpoint_in_directory(resolved_model_path, suffix=".zip")
            if ppo_checkpoint is None:
                msg = f"No PPO checkpoints found in '{resolved_model_path}'."
                raise FileNotFoundError(msg)
            return ppo_checkpoint, False

        graph_checkpoint = _latest_checkpoint_in_directory(resolved_model_path, suffix=".pt")
        ppo_checkpoint = _latest_checkpoint_in_directory(resolved_model_path, suffix=".zip")
        if graph_checkpoint is not None and ppo_checkpoint is None:
            return graph_checkpoint, True
        if ppo_checkpoint is not None and graph_checkpoint is None:
            return ppo_checkpoint, False
        if graph_checkpoint is None and ppo_checkpoint is None:
            msg = f"No supported RL checkpoints found in '{resolved_model_path}'."
            raise FileNotFoundError(msg)
        msg = (
            f"Checkpoint directory '{resolved_model_path}' contains both PPO and GNN checkpoints. "
            "Specify --graph or --no-graph."
        )
        raise ValueError(msg)

    if resolved_model_path.suffix == ".pt":
        if graph is False:
            msg = f"Model path '{resolved_model_path}' is a GNN checkpoint, but graph=False was requested."
            raise ValueError(msg)
        if not resolved_model_path.is_file():
            msg = f"Trained RL model '{resolved_model_path}' does not exist."
            raise FileNotFoundError(msg)
        return resolved_model_path, True

    if resolved_model_path.suffix == ".zip":
        if graph is True:
            msg = f"Model path '{resolved_model_path}' is a PPO checkpoint, but graph=True was requested."
            raise ValueError(msg)
        if not resolved_model_path.is_file():
            msg = f"Trained RL model '{resolved_model_path}' does not exist."
            raise FileNotFoundError(msg)
        return resolved_model_path, False

    zip_path = resolved_model_path.with_suffix(".zip")
    pt_path = resolved_model_path.with_suffix(".pt")
    if graph is True:
        if not pt_path.is_file():
            msg = f"Trained GNN RL model '{pt_path}' does not exist."
            raise FileNotFoundError(msg)
        return pt_path, True
    if graph is False:
        if not zip_path.is_file():
            msg = f"Trained PPO RL model '{zip_path}' does not exist."
            raise FileNotFoundError(msg)
        return zip_path, False
    if pt_path.is_file() and not zip_path.is_file():
        return pt_path, True
    if zip_path.is_file() and not pt_path.is_file():
        return zip_path, False
    if pt_path.is_file() and zip_path.is_file():
        msg = f"Both '{zip_path}' and '{pt_path}' exist. Specify --graph or --no-graph."
        raise ValueError(msg)
    msg = f"Trained RL model '{resolved_model_path}' does not exist."
    raise FileNotFoundError(msg)


def load_model_from_path(model_path: str | Path, *, predictor: Predictor, graph: bool) -> LoadedPolicy:
    """Load either a PPO or GNN policy from a resolved path."""
    resolved_model_path = Path(model_path)
    if graph:
        checkpoint = torch.load(resolved_model_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            msg = f"GNN checkpoint '{resolved_model_path}' does not contain a state_dict."
            raise ValueError(msg)
        config = checkpoint.get("config", {})
        if not isinstance(config, dict):
            msg = f"GNN checkpoint '{resolved_model_path}' does not contain a valid config."
            raise ValueError(msg)

        node_feature_dim = int(config.get("node_feature_dim", 0))
        num_actions = predictor.env.action_space.n  # ty: ignore[unresolved-attribute]
        checkpoint_num_actions = int(config.get("num_actions", num_actions))
        if checkpoint_num_actions != num_actions:
            msg = f"num_actions mismatch: checkpoint={checkpoint_num_actions} current={num_actions}"
            raise RuntimeError(msg)
        if node_feature_dim <= 0:
            msg = f"GNN checkpoint '{resolved_model_path}' has invalid node_feature_dim={node_feature_dim}."
            raise ValueError(msg)

        policy = create_gnn_policy(
            node_feature_dim=node_feature_dim,
            num_actions=num_actions,
            hidden_dim=int(config.get("hidden_dim", 128)),
            num_conv_wo_resnet=int(config.get("num_conv_wo_resnet", 2)),
            num_resnet_layers=int(config.get("num_resnet_layers", 5)),
            dropout_p=float(config.get("dropout_p", 0.2)),
            bidirectional=bool(config.get("bidirectional", True)),
            global_feature_dim=int(config.get("global_feature_dim", 0)),
        )
        policy.load_state_dict(checkpoint["state_dict"], strict=True)
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        policy = policy.to(torch_device)
        policy.eval()
        return LoadedPolicy(model=policy, graph=True, model_path=resolved_model_path, torch_device=torch_device)

    return LoadedPolicy(
        model=MaskablePPO.load(resolved_model_path),
        graph=False,
        model_path=resolved_model_path,
    )


def safe_metric(metric_fn: Callable[[], float]) -> float | None:
    """Safely evaluate a final metric."""
    try:
        return float(metric_fn())
    except (KeyError, TypeError, ValueError, RuntimeError, TranspilerError):
        return None


def optional_nanmean(values: list[float | None]) -> float | None:
    """Compute a mean over optional values."""
    filtered_values = [value for value in values if value is not None]
    if not filtered_values:
        return None
    return nanmean_or_nan(filtered_values)


def nanmean_or_nan(values: list[float] | NDArray[np.float64] | NDArray[np.float32]) -> float:
    """Return ``nanmean`` while handling empty inputs consistently."""
    array = np.asarray(values, dtype=float)
    if array.size == 0 or np.isnan(array).all():
        return float("nan")
    return float(np.nanmean(array))


def json_default(obj: object) -> object:
    """JSON serialization fallback: Path → str, NaN/Inf → null."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    msg = f"Object of type {type(obj).__name__} is not JSON serializable"
    raise TypeError(msg)
