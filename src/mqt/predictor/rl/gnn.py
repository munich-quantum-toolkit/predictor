# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Graph neural network support for the reinforcement learning predictor."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from math import cos, pi
from typing import TYPE_CHECKING, Any, Self, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as functional
from gymnasium import spaces
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

try:
    _torch_geometric_nn = import_module("torch_geometric.nn")
except ModuleNotFoundError as error:
    msg = "GNN support requires the optional dependencies. Install MQT Predictor with the 'gnn' extra."
    raise ImportError(msg) from error

AttentionalAggregation = _torch_geometric_nn.AttentionalAggregation
GraphNorm = _torch_geometric_nn.GraphNorm
SAGEConv = _torch_geometric_nn.SAGEConv

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from qiskit import QuantumCircuit
    from stable_baselines3.common.type_aliases import Schedule
    from torch.optim import Optimizer

    from mqt.predictor.rl.predictorenv import PredictorEnv


NODE_OPERATION_NAMES = (
    "id",
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    "sx",
    "rx",
    "ry",
    "rz",
    "u",
    "u1",
    "u2",
    "u3",
    "cx",
    "cy",
    "cz",
    "ch",
    "crx",
    "cry",
    "crz",
    "cp",
    "cu",
    "cu1",
    "cu2",
    "cu3",
    "swap",
    "iswap",
    "ecr",
    "rzz",
    "rxx",
    "ryy",
    "zz",
    "ccx",
    "reset",
    "measure",
    "other",
)
GLOBAL_FEATURE_NAMES = (
    "num_qubits",
    "depth",
    "program_communication",
    "critical_depth",
    "entanglement_ratio",
    "parallelism",
    "liveness",
    "measure",
    "cx",
    "id",
    "p",
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    "rx",
    "ry",
    "rz",
    "sx",
    "sxdg",
    "cz",
    "cy",
    "swap",
    "ch",
    "crx",
    "cry",
    "crz",
    "cp",
    "cu3",
    "csx",
    "rxx",
    "rzz",
)
NODE_FEATURE_DIM = 52
GLOBAL_FEATURE_DIM = 36
NODE_SCALAR_DIM = NODE_FEATURE_DIM - len(NODE_OPERATION_NAMES)

_GATE_INDICES = "gate_indices"
_NODE_SCALARS = "node_scalars"
_EDGE_INDEX = "edge_index"
_NUM_NODES = "num_nodes"
_NUM_EDGES = "num_edges"
_GLOBAL_FEATURES = "global_features"


@dataclass(frozen=True, slots=True)
class GNNConfig:
    """Configuration for the opt-in GNN policy."""

    max_nodes: int = 512
    max_edges: int = 2048
    hidden_dim: int = 128
    num_conv_wo_resnet: int = 3
    num_resnet_layers: int = 5
    dropout_p: float = 0.2
    bidirectional: bool = True
    learning_rate: float = 3e-4
    gnn_learning_rate: float = 1e-4
    minimum_learning_rate_factor: float = 0.1
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.01

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.max_nodes < 1 or self.max_edges < 1:
            msg = "max_nodes and max_edges must be positive"
            raise ValueError(msg)
        if self.hidden_dim < 1 or self.num_conv_wo_resnet < 1 or self.num_resnet_layers < 0:
            msg = "hidden_dim and num_conv_wo_resnet must be positive, and num_resnet_layers must be non-negative"
            raise ValueError(msg)
        if not 0 <= self.dropout_p < 1:
            msg = "dropout_p must be in the interval [0, 1)"
            raise ValueError(msg)
        if self.learning_rate <= 0 or self.gnn_learning_rate <= 0:
            msg = "learning rates must be positive"
            raise ValueError(msg)
        if not 0 < self.minimum_learning_rate_factor <= 1:
            msg = "minimum_learning_rate_factor must be in the interval (0, 1]"
            raise ValueError(msg)

    @classmethod
    def paper(cls, *, max_nodes: int = 512, max_edges: int = 2048) -> Self:
        """Return the tuned configuration used for the paper model."""
        return cls(
            max_nodes=max_nodes,
            max_edges=max_edges,
            hidden_dim=119,
            num_conv_wo_resnet=1,
            num_resnet_layers=5,
            dropout_p=0.1,
            bidirectional=True,
            learning_rate=1e-3,
            gnn_learning_rate=1e-3,
        )


def _safe_float(value: object) -> float:
    try:
        return float(cast("Any", value))
    except (TypeError, ValueError):
        return 0.0


def _create_sparse_dag(
    qc: QuantumCircuit,
) -> tuple[NDArray[np.int32], NDArray[np.float32], NDArray[np.int32]]:
    dag = circuit_to_dag(PassManager(RemoveBarriers()).run(qc))
    nodes = list(dag.op_nodes())
    node_indices = {node: index for index, node in enumerate(nodes)}
    operation_indices = {name: index for index, name in enumerate(NODE_OPERATION_NAMES)}

    gate_indices = np.zeros(len(nodes), dtype=np.int32)
    node_scalars = np.zeros((len(nodes), NODE_SCALAR_DIM), dtype=np.float32)

    for index, node in enumerate(nodes):
        gate_indices[index] = operation_indices.get(node.op.name, operation_indices["other"])
        raw_parameters = list(getattr(node.op, "params", ()))
        parameters = [_safe_float(value) for value in raw_parameters[:3]]
        parameters.extend([0.0] * (3 - len(parameters)))
        node_scalars[index, :6] = [
            value for parameter in parameters for value in (np.sin(parameter), np.cos(parameter))
        ]
        node_scalars[index, 6] = len(node.qargs)
        node_scalars[index, 7] = getattr(node.op, "num_ctrl_qubits", 0)
        node_scalars[index, 8] = len(raw_parameters)
        node_scalars[index, 10] = sum(isinstance(predecessor, DAGOpNode) for predecessor in dag.predecessors(node))
        node_scalars[index, 11] = sum(isinstance(successor, DAGOpNode) for successor in dag.successors(node))

    edges = [
        (node_indices[source], node_indices[target])
        for source, target, _ in dag.edges()
        if source in node_indices and target in node_indices
    ]
    edge_index = np.asarray(edges, dtype=np.int32).T if edges else np.empty((2, 0), dtype=np.int32)

    topological_nodes = list(dag.topological_op_nodes())
    distance_from_start: dict[DAGOpNode, int] = dict.fromkeys(topological_nodes, 0)
    for node in topological_nodes:
        predecessors = [predecessor for predecessor in dag.predecessors(node) if isinstance(predecessor, DAGOpNode)]
        if predecessors:
            distance_from_start[node] = max(distance_from_start[predecessor] + 1 for predecessor in predecessors)

    distance_to_end: dict[DAGOpNode, int] = dict.fromkeys(topological_nodes, 0)
    for node in reversed(topological_nodes):
        successors = [successor for successor in dag.successors(node) if isinstance(successor, DAGOpNode)]
        if successors:
            distance_to_end[node] = max(distance_to_end[successor] + 1 for successor in successors)

    if topological_nodes:
        longest_path = max(distance_from_start[node] + distance_to_end[node] for node in topological_nodes)
        for index, node in enumerate(nodes):
            node_scalars[index, 9] = distance_from_start[node] + distance_to_end[node] == longest_path

    return gate_indices, node_scalars, edge_index


def _observation_scalar(observation: dict[str, Any], name: str) -> float:
    value = np.asarray(observation[name], dtype=np.float32).reshape(-1)
    if value.size != 1:
        msg = f"Expected one scalar for RL feature '{name}', got shape {value.shape}."
        raise ValueError(msg)
    return float(value[0])


def create_graph_observation(
    qc: QuantumCircuit,
    flat_observation: dict[str, Any],
    *,
    max_nodes: int,
    max_edges: int,
) -> dict[str, NDArray[np.generic]]:
    """Pack a circuit graph into the fixed-shape observation consumed by Stable-Baselines3."""
    gate_indices, node_scalars, edge_index = _create_sparse_dag(qc)
    num_nodes = gate_indices.shape[0]
    num_edges = edge_index.shape[1]
    if num_nodes > max_nodes or num_edges > max_edges:
        msg = (
            f"Circuit graph has {num_nodes} nodes and {num_edges} edges, but the configured GNN capacity is "
            f"{max_nodes} nodes and {max_edges} edges. Increase GNNConfig.max_nodes or GNNConfig.max_edges."
        )
        raise ValueError(msg)

    padded_gate_indices = np.zeros(max_nodes, dtype=np.int32)
    padded_gate_indices[:num_nodes] = gate_indices
    padded_node_scalars = np.zeros((max_nodes, NODE_SCALAR_DIM), dtype=np.float32)
    padded_node_scalars[:num_nodes] = node_scalars
    padded_edge_index = np.zeros((2, max_edges), dtype=np.int32)
    padded_edge_index[:, :num_edges] = edge_index

    global_features = np.asarray(
        [_observation_scalar(flat_observation, name) for name in GLOBAL_FEATURE_NAMES], dtype=np.float32
    )
    global_features[0] = qc.num_qubits
    global_features[1] = qc.depth()

    return {
        _GATE_INDICES: padded_gate_indices,
        _NODE_SCALARS: padded_node_scalars,
        _EDGE_INDEX: padded_edge_index,
        _NUM_NODES: np.asarray([num_nodes], dtype=np.int32),
        _NUM_EDGES: np.asarray([num_edges], dtype=np.int32),
        _GLOBAL_FEATURES: global_features,
    }


def _graph_observation_space(max_nodes: int, max_edges: int) -> spaces.Dict:
    return spaces.Dict({
        _GATE_INDICES: spaces.Box(0, len(NODE_OPERATION_NAMES) - 1, (max_nodes,), dtype=np.int32),
        _NODE_SCALARS: spaces.Box(-np.inf, np.inf, (max_nodes, NODE_SCALAR_DIM), dtype=np.float32),
        _EDGE_INDEX: spaces.Box(0, max_nodes - 1, (2, max_edges), dtype=np.int32),
        _NUM_NODES: spaces.Box(0, max_nodes, (1,), dtype=np.int32),
        _NUM_EDGES: spaces.Box(0, max_edges, (1,), dtype=np.int32),
        _GLOBAL_FEATURES: spaces.Box(-np.inf, np.inf, (GLOBAL_FEATURE_DIM,), dtype=np.float32),
    })


def _graph_capacities(observation_space: spaces.Space[Any]) -> tuple[int, int]:
    if not isinstance(observation_space, spaces.Dict):
        msg = "The GNN policy requires a dictionary observation space."
        raise TypeError(msg)
    try:
        node_space = observation_space[_NODE_SCALARS]
        edge_space = observation_space[_EDGE_INDEX]
        global_space = observation_space[_GLOBAL_FEATURES]
    except KeyError as error:
        msg = "The observation space does not contain the required GNN graph fields."
        raise ValueError(msg) from error
    if node_space.shape is None or edge_space.shape is None or global_space.shape != (GLOBAL_FEATURE_DIM,):
        msg = "The observation space contains incompatible GNN graph shapes."
        raise ValueError(msg)
    return node_space.shape[0], edge_space.shape[1]


class GNNObservationWrapper(gym.Wrapper):
    """Adapt the flat predictor observation to a sparse, fixed-capacity graph observation."""

    def __init__(
        self,
        env: PredictorEnv,
        config: GNNConfig | None = None,
        *,
        observation_space: spaces.Space[Any] | None = None,
    ) -> None:
        """Initialize the graph observation wrapper."""
        super().__init__(env)
        if observation_space is None:
            if config is None:
                msg = "A GNN configuration is required when no saved observation space is provided."
                raise ValueError(msg)
            observation_space = _graph_observation_space(config.max_nodes, config.max_edges)
        self.max_nodes, self.max_edges = _graph_capacities(observation_space)
        self.observation_space = cast("spaces.Dict", observation_space)

    def observation(self, observation: dict[str, Any]) -> dict[str, NDArray[np.generic]]:
        """Convert a flat observation and its circuit state to the graph representation."""
        predictor_env = cast("PredictorEnv", self.unwrapped)
        return create_graph_observation(
            predictor_env.state,
            observation,
            max_nodes=self.max_nodes,
            max_edges=self.max_edges,
        )

    def reset(
        self,
        qc: Path | str | QuantumCircuit | None = None,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, NDArray[np.generic]], dict[str, Any]]:
        """Reset the underlying predictor and return a graph observation."""
        predictor_env = cast("PredictorEnv", self.unwrapped)
        observation, info = predictor_env.reset(qc, seed=seed, options=options)
        return self.observation(observation), info

    def step(self, action: Any) -> tuple[dict[str, NDArray[np.generic]], float, bool, bool, dict[str, Any]]:
        """Step the underlying predictor and return a graph observation."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(observation), float(reward), terminated, truncated, info

    def action_masks(self) -> list[bool]:
        """Forward the action mask required by MaskablePPO."""
        predictor_env = cast("PredictorEnv", self.unwrapped)
        return predictor_env.action_masks()


class GraphSAGEEncoder(nn.Module):
    """Encode variable-size circuit graphs with the prototype GraphSAGE topology."""

    def __init__(
        self,
        hidden_dim: int,
        num_conv_wo_resnet: int,
        num_resnet_layers: int,
        dropout_p: float,
        *,
        bidirectional: bool,
    ) -> None:
        """Initialize the GraphSAGE encoder."""
        super().__init__()
        layer_count = num_conv_wo_resnet + num_resnet_layers
        self.convs = nn.ModuleList([
            SAGEConv(NODE_FEATURE_DIM if index == 0 else hidden_dim, hidden_dim) for index in range(layer_count)
        ])
        self.norms = nn.ModuleList(GraphNorm(hidden_dim) for _ in range(layer_count))
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.residual_start = num_conv_wo_resnet
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.pool = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """Return one embedding per circuit graph."""
        if node_features.shape[0] == 0:
            return node_features.new_zeros((num_graphs, self.hidden_dim))

        features = node_features
        for index, (conv, norm) in enumerate(zip(self.convs, self.norms, strict=True)):
            updated = conv(features, edge_index)
            if self.bidirectional:
                updated = 0.5 * (updated + conv(features, edge_index.flip(0)))
            updated = self.dropout(self.activation(norm(updated, batch=batch)))
            features = updated if index < self.residual_start else features + updated
        return self.pool(features, index=batch, dim_size=num_graphs)


class GNNFeaturesExtractor(BaseFeaturesExtractor):
    """Stable-Baselines3 feature extractor implementing the prototype GNN and shared trunk."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_dim: int = 128,
        num_conv_wo_resnet: int = 3,
        num_resnet_layers: int = 5,
        dropout_p: float = 0.2,
        *,
        bidirectional: bool = True,
    ) -> None:
        """Initialize the GNN feature extractor."""
        _graph_capacities(observation_space)
        super().__init__(observation_space, features_dim=hidden_dim)
        self.encoder = GraphSAGEEncoder(
            hidden_dim,
            num_conv_wo_resnet,
            num_resnet_layers,
            dropout_p,
            bidirectional=bidirectional,
        )
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim + GLOBAL_FEATURE_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
        )

    @staticmethod
    def _batch_graphs(observations: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = observations[_GLOBAL_FEATURES].shape[0]
        node_features: list[torch.Tensor] = []
        edge_indices: list[torch.Tensor] = []
        batches: list[torch.Tensor] = []
        node_offset = 0
        for graph_index in range(batch_size):
            num_nodes = int(observations[_NUM_NODES][graph_index].reshape(-1)[0].item())
            num_edges = int(observations[_NUM_EDGES][graph_index].reshape(-1)[0].item())
            gate_indices = observations[_GATE_INDICES][graph_index, :num_nodes].long()
            scalars = observations[_NODE_SCALARS][graph_index, :num_nodes]
            one_hot = functional.one_hot(gate_indices, num_classes=len(NODE_OPERATION_NAMES)).to(scalars.dtype)
            node_features.append(torch.cat((one_hot, scalars), dim=1))
            edge_indices.append(observations[_EDGE_INDEX][graph_index, :, :num_edges].long() + node_offset)
            batches.append(torch.full((num_nodes,), graph_index, dtype=torch.long, device=scalars.device))
            node_offset += num_nodes

        return torch.cat(node_features), torch.cat(edge_indices, dim=1), torch.cat(batches)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract a shared latent feature vector for the actor and critic."""
        node_features, edge_index, batch = self._batch_graphs(observations)
        global_features = observations[_GLOBAL_FEATURES].reshape(-1, GLOBAL_FEATURE_DIM)
        graph_embedding = self.encoder(node_features, edge_index, batch, global_features.shape[0])
        return self.trunk(torch.cat((graph_embedding, global_features), dim=1))


class GNNMaskableMultiInputActorCriticPolicy(MaskableMultiInputActorCriticPolicy):
    """Maskable multi-input policy with a separate learning rate for the GNN encoder."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space[Any],
        lr_schedule: Schedule,
        *,
        gnn_learning_rate: float,
        **kwargs: Any,
    ) -> None:
        """Initialize the maskable GNN policy."""
        self.gnn_learning_rate = gnn_learning_rate
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        extractor = cast("GNNFeaturesExtractor", self.features_extractor)
        encoder_parameters = list(extractor.encoder.parameters())
        encoder_parameter_ids = {id(parameter) for parameter in encoder_parameters}
        policy_parameters = [parameter for parameter in self.parameters() if id(parameter) not in encoder_parameter_ids]
        policy_learning_rate = float(lr_schedule(1.0))
        self.optimizer = self.optimizer_class(
            [
                {
                    "params": encoder_parameters,
                    "lr": self.gnn_learning_rate,
                    "learning_rate_scale": self.gnn_learning_rate / policy_learning_rate,
                },
                {"params": policy_parameters, "lr": policy_learning_rate, "learning_rate_scale": 1.0},
            ],
            **self.optimizer_kwargs,
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data["gnn_learning_rate"] = self.gnn_learning_rate
        return data


class GNNMaskablePPO(MaskablePPO):
    """MaskablePPO variant that retains the encoder-to-policy learning-rate ratio."""

    def _update_learning_rate(self, optimizers: list[Optimizer] | Optimizer) -> None:
        current_learning_rate = self.lr_schedule(self._current_progress_remaining)
        self.logger.record("train/learning_rate", current_learning_rate)
        optimizer_sequence = optimizers if isinstance(optimizers, list) else (optimizers,)
        for optimizer in optimizer_sequence:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = current_learning_rate * parameter_group.get("learning_rate_scale", 1.0)


def _cosine_schedule(initial_value: float, minimum_factor: float) -> Schedule:
    def schedule(progress_remaining: float) -> float:
        completed_fraction = 1.0 - progress_remaining
        factor = minimum_factor + (1.0 - minimum_factor) * 0.5 * (1.0 + cos(pi * completed_fraction))
        return initial_value * factor

    return schedule


def create_gnn_model(
    env: GNNObservationWrapper,
    config: GNNConfig,
    *,
    verbose: int,
    tensorboard_log: str,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
    seed: int | None,
) -> GNNMaskablePPO:
    """Create the GNN policy on top of the existing MaskablePPO implementation."""
    return GNNMaskablePPO(
        GNNMaskableMultiInputActorCriticPolicy,
        env,
        learning_rate=_cosine_schedule(config.learning_rate, config.minimum_learning_rate_factor),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        clip_range_vf=config.clip_range_vf,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        target_kl=config.target_kl,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
        seed=seed,
        policy_kwargs={
            "features_extractor_class": GNNFeaturesExtractor,
            "features_extractor_kwargs": {
                "hidden_dim": config.hidden_dim,
                "num_conv_wo_resnet": config.num_conv_wo_resnet,
                "num_resnet_layers": config.num_resnet_layers,
                "dropout_p": config.dropout_p,
                "bidirectional": config.bidirectional,
            },
            "net_arch": {"pi": [config.hidden_dim], "vf": [config.hidden_dim]},
            "activation_fn": nn.GELU,
            "optimizer_class": torch.optim.AdamW,
            "optimizer_kwargs": {"eps": 1e-5},
            "gnn_learning_rate": config.gnn_learning_rate,
        },
    )
