# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for variable-size circuit graphs and masked GNN training."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import torch
from mqt.bench.targets import get_device
from qiskit import QuantumCircuit
from qiskit.qasm2 import dump
from qiskit.quantum_info import Operator

from mqt.predictor.rl import Predictor
from mqt.predictor.rl import predictor as predictor_module
from mqt.predictor.rl.helper import create_feature_dict

if TYPE_CHECKING:
    from pathlib import Path

    from mqt.predictor.rl.gnn import (
        GNNFeaturesExtractor,
        GNNMaskableDictRolloutBuffer,
        GNNMaskableMultiInputActorCriticPolicy,
        GraphBatch,
    )

pytest.importorskip("torch_geometric")
gnn = import_module("mqt.predictor.rl.gnn")

# PyG 2.8.0.post1 uses the deprecated typing._eval_type signature on Python 3.14.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Failing to pass a value to the 'type_params' parameter:DeprecationWarning:torch_geometric.inspector"
)


@pytest.fixture
def gnn_predictor(tmp_path: Path) -> Predictor:
    """Create a small predictor with a circuit that changes size when optimized."""
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.rz(0.5, 1)
    dump(circuit, tmp_path / "circuit_2.qasm")
    return Predictor(
        figure_of_merit="critical_depth",
        device=get_device("ibm_falcon_27"),
        path_training_circuits=tmp_path,
        max_steps=2,
        graph=True,
        gnn_config=gnn.GNNConfig(
            hidden_dim=8,
            num_conv_wo_resnet=1,
            num_resnet_layers=1,
            dropout_p=0,
            learning_rate=1e-3,
            gnn_learning_rate=2e-3,
            n_steps=4,
            batch_size=2,
            n_epochs=1,
        ),
    )


def test_graph_observation_preserves_dag_features() -> None:
    """Keep gate parameters, qubit dependencies, and raw global sizes in the graph."""
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.barrier()
    circuit.cx(0, 1)
    circuit.rz(np.pi / 2, 1)
    flat_observation = create_feature_dict(circuit, 27)

    graph = gnn.create_graph_observation(circuit, flat_observation)

    assert graph.num_nodes == 3
    assert graph["gate_indices"].tolist() == [gnn.NODE_OPERATION_NAMES.index(gate) for gate in ("h", "cx", "rz")]
    assert graph["edge_index"].tolist() == [[0, 1], [1, 2]]
    torch.testing.assert_close(graph["node_scalars"][2, :2], torch.tensor([1.0, 0.0]), atol=1e-7, rtol=0)
    assert graph["node_scalars"][:, 6:9].tolist() == [[1, 0, 0], [2, 1, 0], [1, 0, 1]]
    assert graph["node_scalars"][:, 9:].tolist() == [[1, 0, 1], [1, 1, 1], [1, 1, 0]]
    assert graph["global_features"][0, :2].tolist() == [2, 3]
    assert graph["global_features"][0, 2:].tolist() == pytest.approx([
        flat_observation[name][0] for name in gnn.GLOBAL_FEATURE_NAMES[2:]
    ])


def test_gnn_masked_training_and_saved_inference(
    gnn_predictor: Predictor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Train real graph minibatches, preserve masks, and reload the resulting compiler."""
    predictor = gnn_predictor
    env = predictor.env
    action_mask = [
        action.name in {"CommutativeCancellation", "InverseCancellation"} for action in env.action_set.values()
    ]
    monkeypatch.setattr(env, "action_masks", lambda: action_mask)
    graph_env = gnn.GNNObservationWrapper(env)
    assert predictor.gnn_config is not None
    model = gnn.create_gnn_model(graph_env, predictor.gnn_config, verbose=0, tensorboard_log=str(tmp_path), seed=7)
    policy = cast("GNNMaskableMultiInputActorCriticPolicy", model.policy)
    encoder = cast("GNNFeaturesExtractor", policy.features_extractor).encoder
    buffer = cast("GNNMaskableDictRolloutBuffer", model.rollout_buffer)
    before = [parameter.detach().clone() for parameter in encoder.parameters()]

    model.learn(total_timesteps=4)

    assert model.num_timesteps == 4
    assert not env.error_occurred
    assert all(action_mask[int(action)] for action in buffer.actions.flatten())
    assert {graph.num_nodes for graph in buffer.graph_observations} == {2, 4}
    assert any(not torch.equal(old, new) for old, new in zip(before, encoder.parameters(), strict=True))
    assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad) for parameter in encoder.parameters())
    assert all(torch.isfinite(parameter).all() for parameter in model.policy.parameters())
    assert [group["lr"] for group in model.policy.optimizer.param_groups] == pytest.approx([2e-4, 1e-4])

    graph_env.reset()
    circuit = env.state.copy()
    empty_circuit = QuantumCircuit(2)
    graphs = [
        gnn.create_graph_observation(empty_circuit, create_feature_dict(empty_circuit, env.device.num_qubits)),
        graph_env.graph_observation,
    ]
    batch, vectorized = policy.obs_to_tensor(graphs)
    assert vectorized
    policy.set_training_mode(False)
    with torch.no_grad():
        features = policy.features_extractor(batch)
        values = policy.predict_values(batch)
    assert features.shape == (2, 8)
    assert torch.isfinite(features).all()
    masks = np.tile(action_mask, (2, 1))
    expected_actions, _ = model.predict(graphs, deterministic=True, action_masks=masks)  # ty: ignore[invalid-argument-type]
    assert all(action_mask[int(action)] for action in expected_actions)
    empty_action, _ = model.predict(graphs[0], deterministic=True, action_masks=masks[0])  # ty: ignore[invalid-argument-type]
    assert action_mask[int(empty_action)]

    model.save(tmp_path / predictor.model_name)
    monkeypatch.setattr(predictor_module, "get_path_trained_model", lambda: tmp_path)
    loaded = predictor_module.load_model(predictor.model_name, graph=True)
    actual_actions, _ = loaded.predict(batch, deterministic=True, action_masks=masks)  # ty: ignore[invalid-argument-type]
    np.testing.assert_array_equal(actual_actions, expected_actions)
    loaded_policy = cast("GNNMaskableMultiInputActorCriticPolicy", loaded.policy)
    loaded_policy.set_training_mode(False)
    with torch.no_grad():
        torch.testing.assert_close(loaded_policy.predict_values(batch), values)
    compiled, passes = predictor.compile_as_predicted(circuit)
    assert len(passes) == 2
    assert set(passes) <= {"CommutativeCancellation", "InverseCancellation"}
    assert Operator(compiled).equiv(Operator(circuit))


@pytest.mark.parametrize("truncated", [False, True])
def test_gnn_bootstraps_only_the_truncated_terminal_graph(
    gnn_predictor: Predictor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, truncated: bool
) -> None:
    """Use the final graph on truncation, without bootstrapping a terminated episode."""
    env = gnn_predictor.env
    env.intermediate_reward = False
    action_mask = [index == env.action_terminate_index for index in env.action_set]
    monkeypatch.setattr(env, "action_masks", lambda: action_mask)

    def apply_action(_action: int) -> QuantumCircuit:
        env.state.x(0)
        if truncated:
            msg = "Compilation failed after changing the circuit."
            raise RuntimeError(msg)
        return env.state

    monkeypatch.setattr(env, "apply_action", apply_action)
    assert gnn_predictor.gnn_config is not None
    model = gnn.create_gnn_model(
        gnn.GNNObservationWrapper(env), gnn_predictor.gnn_config, verbose=0, tensorboard_log=str(tmp_path), seed=7
    )

    def graph_value(graphs: GraphBatch) -> torch.Tensor:
        return torch.bincount(graphs.batch, minlength=graphs.num_graphs).float().reshape(-1, 1)

    monkeypatch.setattr(model.policy, "predict_values", graph_value)

    model.learn(total_timesteps=4)

    expected_reward = env.no_effect_penalty + model.gamma * 5 if truncated else 0
    np.testing.assert_allclose(model.rollout_buffer.rewards, expected_reward)
    assert model.rollout_buffer.episode_starts.all()
    buffer = cast("GNNMaskableDictRolloutBuffer", model.rollout_buffer)
    assert {graph.num_nodes for graph in buffer.graph_observations} == {4}
