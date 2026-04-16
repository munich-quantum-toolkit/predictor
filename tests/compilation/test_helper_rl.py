# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the helper functions of the reinforcement learning predictor."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from mqt.bench import BenchmarkLevel, get_benchmark
from mqt.bench.targets import get_device
from qiskit import transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes.layout.vf2_post_layout import VF2PostLayoutStopReason
from torch_geometric.data import Batch, Data

from mqt.predictor.rl.actions import PassType, get_actions_by_pass_type
from mqt.predictor.rl.gnn import SAGEActorCritic
from mqt.predictor.rl.gnn_ppo import create_gnn_policy
from mqt.predictor.rl.helper import create_dag, create_feature_dict, get_path_trained_model, get_path_training_circuits
from mqt.predictor.rl.parsing import postprocess_vf2postlayout

if TYPE_CHECKING:
    from collections.abc import Callable

    from qiskit.passmanager.base_tasks import Task
    from qiskit.transpiler import Target


def test_create_feature_dict() -> None:
    """Test the creation of a feature dictionary."""
    qc = get_benchmark("dj", BenchmarkLevel.ALG, 5)
    features = create_feature_dict(qc)
    for feature in features.values():
        assert isinstance(feature, np.ndarray | int)


def test_get_path_trained_model() -> None:
    """Test the retrieval of the path to the trained model."""
    path = get_path_trained_model()
    assert path.exists()
    assert isinstance(path, Path)


def test_get_path_training_circuits() -> None:
    """Test the retrieval of the path to the training circuits."""
    path = get_path_training_circuits()
    assert path.exists()
    assert isinstance(path, Path)


def test_vf2_layout_and_postlayout() -> None:
    """Test the VF2Layout and VF2PostLayout passes."""
    qc = get_benchmark("ghz", BenchmarkLevel.ALG, 3)

    for dev in [get_device("ibm_falcon_27"), get_device("quantinuum_h2_56")]:
        passes: list[Task] | None = None
        for layout_action in get_actions_by_pass_type()[PassType.LAYOUT]:
            if layout_action.name == "VF2Layout":
                factory = cast("Callable[[Target], list[Task]]", layout_action.transpile_pass)
                passes = factory(dev)
                break
        assert passes is not None
        pm = PassManager(passes)
        layouted_qc = pm.run(qc)
        assert layouted_qc.layout is not None
        assert len(layouted_qc.layout.initial_layout) == dev.num_qubits

    dev_success = get_device("ibm_falcon_27")
    qc_transpiled = transpile(qc, target=dev_success, optimization_level=0)
    assert qc_transpiled.layout is not None

    initial_layout_before = qc_transpiled.layout.initial_layout

    post_layout_passes: list[Task] | None = None
    for layout_action in get_actions_by_pass_type()[PassType.FINAL_OPT]:
        if layout_action.name == "VF2PostLayout":
            factory = cast("Callable[[Target], list[Task]]", layout_action.transpile_pass)
            post_layout_passes = factory(dev_success)
            break
    assert post_layout_passes is not None

    pm = PassManager(post_layout_passes)
    altered_qc = pm.run(qc_transpiled)

    assert pm.property_set["VF2PostLayout_stop_reason"] == VF2PostLayoutStopReason.SOLUTION_FOUND

    _, pass_manager = postprocess_vf2postlayout(altered_qc, pm.property_set["post_layout"], qc_transpiled.layout)

    assert initial_layout_before != pass_manager.property_set["initial_layout"]


def test_create_dag_output_shapes() -> None:
    """Test that create_dag returns tensors with correct dtypes and shapes."""
    qc = get_benchmark("ghz", BenchmarkLevel.ALG, 3)
    node_vector, edge_index, number_of_gates = create_dag(qc)

    assert isinstance(node_vector, torch.Tensor)
    assert isinstance(edge_index, torch.Tensor)
    assert isinstance(number_of_gates, int)
    assert number_of_gates > 0
    assert node_vector.shape == (number_of_gates, node_vector.shape[1])
    assert node_vector.shape[1] > 0
    assert node_vector.dtype == torch.float32
    assert edge_index.shape[0] == 2
    assert edge_index.dtype == torch.long


def test_create_feature_dict_graph_mode() -> None:
    """Test that create_feature_dict with graph=True returns a PyG Data object."""
    qc = get_benchmark("dj", BenchmarkLevel.ALG, 5)
    data = create_feature_dict(qc, graph=True)

    assert isinstance(data, Data)
    assert isinstance(data.x, torch.Tensor)
    assert isinstance(data.edge_index, torch.Tensor)
    assert data.x.shape[0] > 0
    assert data.edge_index.shape[0] == 2
    # node feature dim must be consistent with create_dag
    _, _, num_nodes = create_dag(qc)
    assert data.x.shape[0] == num_nodes


def test_sage_actor_critic_forward() -> None:
    """Test SAGEActorCritic forward pass produces tensors with correct shapes."""
    qc = get_benchmark("ghz", BenchmarkLevel.ALG, 3)
    node_vector, edge_index, num_nodes = create_dag(qc)
    in_feats = node_vector.shape[1]
    num_actions = 42

    model = SAGEActorCritic(
        in_feats=in_feats,
        hidden_dim=32,
        num_conv_wo_resnet=1,
        num_resnet_layers=1,
        num_actions=num_actions,
    )
    model.eval()

    data = Data(x=node_vector, edge_index=edge_index, num_nodes=num_nodes)
    batch = Batch.from_data_list([data])
    with torch.no_grad():
        logits, value = model(batch)

    assert logits.shape == (1, num_actions)
    assert value.shape == (1, 1)


def test_sage_actor_critic_batch_forward() -> None:
    """Test SAGEActorCritic forward pass on a batch of graphs."""
    qc = get_benchmark("ghz", BenchmarkLevel.ALG, 3)
    node_vector, edge_index, num_nodes = create_dag(qc)
    in_feats = node_vector.shape[1]
    num_actions = 10
    batch_size = 4

    model = SAGEActorCritic(
        in_feats=in_feats,
        hidden_dim=32,
        num_conv_wo_resnet=1,
        num_resnet_layers=1,
        num_actions=num_actions,
    )
    model.eval()

    graphs = [Data(x=node_vector, edge_index=edge_index, num_nodes=num_nodes) for _ in range(batch_size)]
    batch = Batch.from_data_list(graphs)
    with torch.no_grad():
        logits, value = model(batch)

    assert logits.shape == (batch_size, num_actions)
    assert value.shape == (batch_size, 1)


def test_create_gnn_policy_factory() -> None:
    """Test that create_gnn_policy returns a correctly configured SAGEActorCritic."""
    qc = get_benchmark("ghz", BenchmarkLevel.ALG, 3)
    node_vector, _, _ = create_dag(qc)
    in_feats = node_vector.shape[1]
    num_actions = 15
    hidden_dim = 64

    policy = create_gnn_policy(
        node_feature_dim=in_feats,
        num_actions=num_actions,
        hidden_dim=hidden_dim,
        num_conv_wo_resnet=1,
        num_resnet_layers=1,
    )

    assert isinstance(policy, SAGEActorCritic)
    assert policy.encoder.out_dim == hidden_dim
