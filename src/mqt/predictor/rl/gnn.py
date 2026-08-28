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
from typing import TYPE_CHECKING, Any, NamedTuple, Self, cast

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
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

try:
    _torch_geometric_data = import_module("torch_geometric.data")
    _torch_geometric_nn = import_module("torch_geometric.nn")
except ModuleNotFoundError as error:
    msg = "GNN support requires the optional dependencies. Install MQT Predictor with the 'gnn' extra."
    raise ImportError(msg) from error

AttentionalAggregation = _torch_geometric_nn.AttentionalAggregation
_PyGBatch: Any = _torch_geometric_data.Batch
_PyGData: Any = _torch_geometric_data.Data
GraphNorm = _torch_geometric_nn.GraphNorm
SAGEConv = _torch_geometric_nn.SAGEConv

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence
    from pathlib import Path
    from typing import Protocol

    from numpy.typing import NDArray
    from qiskit import QuantumCircuit
    from sb3_contrib.common.maskable.distributions import MaskableDistribution
    from stable_baselines3.common.buffers import RolloutBuffer
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.type_aliases import Schedule
    from stable_baselines3.common.vec_env import VecEnv, VecNormalize
    from torch.optim import Optimizer

    from mqt.predictor.rl.predictorenv import PredictorEnv

    class GraphData(Protocol):
        """Typed subset of the optional PyG Data API used here."""

        num_nodes: int

        def __getitem__(self, key: str) -> torch.Tensor:
            """Return a graph tensor by key."""
            ...

        def to(self, device: torch.device | str) -> Self:
            """Move the graph tensors to a device."""
            ...

    class GraphBatch(GraphData, Protocol):
        """Typed subset of the optional PyG Batch API used here."""

        batch: torch.Tensor
        num_graphs: int


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
_GLOBAL_FEATURES = "global_features"
_TERMINAL_GRAPH_OBSERVATION = "_mqt_gnn_terminal_observation"


@dataclass(frozen=True, slots=True)
class GNNConfig:
    """Configuration for the opt-in GNN policy."""

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
    def paper(cls) -> Self:
        """Return the tuned configuration used for the paper model."""
        return cls(
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
) -> GraphData:
    """Create an exact, variable-size graph alongside the regular RL observation."""
    gate_indices, node_scalars, edge_index = _create_sparse_dag(qc)
    global_features = np.asarray(
        [_observation_scalar(flat_observation, name) for name in GLOBAL_FEATURE_NAMES], dtype=np.float32
    )
    global_features[0] = qc.num_qubits
    global_features[1] = qc.depth()

    return cast(
        "GraphData",
        _PyGData(
            gate_indices=torch.as_tensor(gate_indices, dtype=torch.long),
            node_scalars=torch.as_tensor(node_scalars, dtype=torch.float32),
            edge_index=torch.as_tensor(edge_index, dtype=torch.long),
            global_features=torch.as_tensor(global_features, dtype=torch.float32).reshape(1, -1),
            num_nodes=gate_indices.shape[0],
        ),
    )


class GNNObservationWrapper(gym.Wrapper):
    """Maintain a variable-size graph sidecar while preserving the regular RL interface."""

    def _update_graph_observation(self, observation: dict[str, Any]) -> None:
        predictor_env = cast("PredictorEnv", self.unwrapped)
        self.graph_observation = create_graph_observation(predictor_env.state, observation)

    def reset(
        self,
        qc: Path | str | QuantumCircuit | None = None,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, NDArray[np.generic]], dict[str, Any]]:
        """Reset the predictor and refresh the graph sidecar."""
        predictor_env = cast("PredictorEnv", self.unwrapped)
        observation, info = predictor_env.reset(qc, seed=seed, options=options)
        self._update_graph_observation(observation)
        return observation, info

    def step(self, action: int) -> tuple[dict[str, NDArray[np.generic]], float, bool, bool, dict[str, Any]]:
        """Step the predictor and refresh the graph sidecar."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._update_graph_observation(observation)
        if truncated and not terminated:
            info[_TERMINAL_GRAPH_OBSERVATION] = self.graph_observation
        return observation, float(reward), terminated, truncated, info

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

    def forward(self, observations: GraphBatch) -> torch.Tensor:
        """Extract a shared latent feature vector for the actor and critic."""
        gate_indices = observations[_GATE_INDICES]
        node_scalars = observations[_NODE_SCALARS]
        edge_index = observations[_EDGE_INDEX]
        batch = observations.batch
        global_features = observations[_GLOBAL_FEATURES].reshape(-1, GLOBAL_FEATURE_DIM)
        one_hot = functional.one_hot(gate_indices, num_classes=len(NODE_OPERATION_NAMES)).to(node_scalars.dtype)
        node_features = torch.cat((one_hot, node_scalars), dim=1)
        graph_embedding = self.encoder(node_features, edge_index, batch, observations.num_graphs)
        return self.trunk(torch.cat((graph_embedding, global_features), dim=1))


class GNNMaskableDictRolloutBufferSamples(NamedTuple):
    """A MaskablePPO minibatch with a variable-size graph batch."""

    observations: GraphBatch
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    action_masks: torch.Tensor


class GNNMaskableDictRolloutBuffer(MaskableDictRolloutBuffer):
    """Store graph sidecars while retaining MaskablePPO's numeric rollout data and GAE."""

    graph_observations: NDArray[np.object_]

    def reset(self) -> None:
        """Reset both the standard rollout data and graph sidecars."""
        self.graph_observations = np.empty((self.buffer_size, self.n_envs), dtype=object)
        super().reset()

    def add(  # ty: ignore[invalid-method-override]
        self,
        obs: dict[str, NDArray[np.generic]],
        action: NDArray[np.generic],
        reward: NDArray[np.generic],
        episode_start: NDArray[np.generic],
        value: torch.Tensor,
        log_prob: torch.Tensor,
        *,
        action_masks: NDArray[np.generic] | None = None,
        graph_observations: Sequence[GraphData] | None = None,
    ) -> None:
        """Add one vectorized transition and its graph sidecars."""
        if graph_observations is None:
            msg = "GNN rollouts require graph observations."
            raise ValueError(msg)
        if len(graph_observations) != self.n_envs:
            msg = f"Expected {self.n_envs} graph observations, got {len(graph_observations)}."
            raise ValueError(msg)
        self.graph_observations[self.pos] = graph_observations
        super().add(obs, action, reward, episode_start, value, log_prob, action_masks=action_masks)

    def get(  # ty: ignore[invalid-method-override]
        self, batch_size: int | None = None
    ) -> Generator[GNNMaskableDictRolloutBufferSamples, None, None]:
        """Yield shuffled minibatches with PyG's exact ragged batching."""
        assert self.full
        indices = np.random.permutation(self.buffer_size * self.n_envs)  # ruff: ignore[numpy-legacy-random]
        if not self.generator_ready:
            self.graph_observations = self.swap_and_flatten(self.graph_observations).reshape(-1)
            for tensor_name in ("actions", "values", "log_probs", "advantages", "returns", "action_masks"):
                self.__dict__[tensor_name] = self.swap_and_flatten(self.__dict__[tensor_name])
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_index = 0
        while start_index < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_index : start_index + batch_size])
            start_index += batch_size

    def _get_samples(  # ty: ignore[invalid-method-override]
        self,
        batch_inds: NDArray[np.int64],
        _env: VecNormalize | None = None,
    ) -> GNNMaskableDictRolloutBufferSamples:
        graphs = [cast("GraphData", self.graph_observations[index]) for index in batch_inds]
        observations = cast("GraphBatch", _PyGBatch.from_data_list(graphs).to(self.device))
        return GNNMaskableDictRolloutBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
            action_masks=self.to_torch(self.action_masks[batch_inds].reshape(-1, self.mask_dims)),
        )


class GNNMaskableMultiInputActorCriticPolicy(MaskableMultiInputActorCriticPolicy):
    """Maskable multi-input policy with a separate learning rate for the GNN encoder."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space[Any],
        lr_schedule: Schedule,
        *,
        gnn_learning_rate: float,
        **kwargs: Any,  # ruff: ignore[any-type]
    ) -> None:
        """Initialize the maskable GNN policy."""
        self.gnn_learning_rate = gnn_learning_rate
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def obs_to_tensor(  # ty: ignore[invalid-method-override]
        self,
        observation: GraphData | GraphBatch | Sequence[GraphData],
    ) -> tuple[GraphBatch, bool]:
        """Move one graph or an exact ragged graph batch to the policy device."""
        if isinstance(observation, _PyGBatch):
            batch = observation
            vectorized = True
        elif isinstance(observation, _PyGData):
            batch = _PyGBatch.from_data_list([observation])
            vectorized = False
        else:
            batch = _PyGBatch.from_data_list(list(observation))
            vectorized = True
        return cast("GraphBatch", batch.to(self.device)), vectorized

    def extract_features(  # ty: ignore[invalid-method-override]
        self,
        obs: GraphBatch,
        features_extractor: BaseFeaturesExtractor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply the GNN extractor directly, without SB3's fixed-shape preprocessing."""
        if features_extractor is not None:
            return features_extractor(obs)
        if self.share_features_extractor:
            return self.features_extractor(obs)
        return self.pi_features_extractor(obs), self.vf_features_extractor(obs)

    def get_distribution(  # ty: ignore[invalid-method-override]
        self,
        obs: GraphBatch,
        action_masks: NDArray[np.bool_] | None = None,
    ) -> MaskableDistribution:
        """Return the masked action distribution for a graph batch."""
        features = cast("torch.Tensor", self.extract_features(obs, self.pi_features_extractor))
        latent_pi = self.mlp_extractor.forward_actor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def predict_values(self, obs: GraphBatch) -> torch.Tensor:  # ty: ignore[invalid-method-override]
        """Estimate values for a graph batch."""
        features = cast("torch.Tensor", self.extract_features(obs, self.vf_features_extractor))
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

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

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> bool:
        """Collect rollouts with graph sidecars while retaining MaskablePPO's rollout logic."""
        assert isinstance(rollout_buffer, GNNMaskableDictRolloutBuffer)
        assert self._last_obs is not None
        policy = cast("GNNMaskableMultiInputActorCriticPolicy", self.policy)
        policy.set_training_mode(False)
        n_steps = 0
        action_masks = None
        rollout_buffer.reset()

        if use_masking and not is_masking_supported(env):
            msg = "Environment does not support action masking. Consider using ActionMasker wrapper."
            raise ValueError(msg)

        graph_observations = cast("list[GraphData]", env.get_attr("graph_observation"))
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with torch.no_grad():
                obs_tensor, _ = policy.obs_to_tensor(graph_observations)
                if use_masking:
                    action_masks = get_action_masks(env)
                actions, values, log_probs = policy(obs_tensor, action_masks=action_masks)

            actions = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions)
            terminal_graph_observations = [info.pop(_TERMINAL_GRAPH_OBSERVATION, None) for info in infos]
            new_graph_observations = cast("list[GraphData]", env.get_attr("graph_observation"))

            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            for index, done in enumerate(dones):
                if (
                    done
                    and infos[index].get("terminal_observation") is not None
                    and infos[index].get("TimeLimit.truncated", False)
                ):
                    terminal_graph_observation = terminal_graph_observations[index]
                    if terminal_graph_observation is None:
                        msg = "The GNN environment did not provide the graph for a truncated episode."
                        raise RuntimeError(msg)
                    terminal_obs, _ = policy.obs_to_tensor(cast("GraphData", terminal_graph_observation))
                    with torch.no_grad():
                        terminal_value = policy.predict_values(terminal_obs)[0]
                    rewards[index] += self.gamma * float(terminal_value.item())

            rollout_buffer.add(
                cast("dict[str, NDArray[np.generic]]", self._last_obs),
                actions,
                rewards,
                cast("NDArray[np.generic]", self._last_episode_starts),
                values,
                log_probs,
                action_masks=action_masks,
                graph_observations=graph_observations,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones
            graph_observations = new_graph_observations

        with torch.no_grad():
            final_obs, _ = policy.obs_to_tensor(graph_observations)
            values = policy.predict_values(final_obs)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.on_rollout_end()
        return True

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
    seed: int | None,
) -> GNNMaskablePPO:
    """Create the GNN policy on top of the existing MaskablePPO implementation."""
    return GNNMaskablePPO(
        GNNMaskableMultiInputActorCriticPolicy,
        env,
        learning_rate=_cosine_schedule(config.learning_rate, config.minimum_learning_rate_factor),
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
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
        rollout_buffer_class=GNNMaskableDictRolloutBuffer,
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
