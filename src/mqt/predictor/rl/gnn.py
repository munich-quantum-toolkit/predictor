# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""GNN actor-critic architecture for the RL compilation predictor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch_geometric.nn import GraphNorm, SAGEConv, global_mean_pool

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch_geometric.data import Data


class GraphConvolutionSageEncoder(nn.Module):
    """SAGEConv + GraphNorm encoder producing a graph embedding via global pooling."""

    def __init__(
        self,
        in_feats: int,
        hidden_dim: int,
        num_conv_wo_resnet: int,
        num_resnet_layers: int,
        *,
        conv_activation: Callable[..., torch.Tensor] = f.leaky_relu,
        conv_act_kwargs: dict[str, Any] | None = None,
        dropout_p: float = 0.2,
        bidirectional: bool = True,
    ) -> None:
        """Initialize the GraphConvolutionSageEncoder.

        Args:
            in_feats: Dimension of input node features.
            hidden_dim: Dimension of hidden layers and output graph embedding.
            num_conv_wo_resnet: Number of initial convolutional layers without residual connections.
            num_resnet_layers: Number of subsequent convolutional layers with residual connections.
            conv_activation: Activation function to apply after each convolutional layer.
            conv_act_kwargs: Optional keyword arguments for the activation function.
            dropout_p: Dropout probability to apply after each convolutional layer.
            bidirectional: If True, apply each SAGEConv in both directions and average the results.
        """
        super().__init__()

        if num_conv_wo_resnet < 1:
            msg = "num_conv_wo_resnet must be at least 1"
            raise ValueError(msg)

        self.conv_activation = conv_activation
        self.conv_act_kwargs = conv_act_kwargs or {}
        self.bidirectional = bidirectional

        self.convs: nn.ModuleList = nn.ModuleList()
        self.norms: nn.ModuleList = nn.ModuleList()

        # first layer
        self.convs.append(SAGEConv(in_feats, hidden_dim))
        self.norms.append(GraphNorm(hidden_dim))

        # remaining non-residual layers
        for _ in range(num_conv_wo_resnet - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(GraphNorm(hidden_dim))

        # residual layers (same width)
        for _ in range(num_resnet_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(GraphNorm(hidden_dim))

        self.drop = nn.Dropout(dropout_p)
        self._residual_start = num_conv_wo_resnet
        self.out_dim = hidden_dim
        self.graph_emb_dim = hidden_dim

    def _apply_conv_bidir(
        self,
        conv: SAGEConv,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the convolution in both directions if bidirectional is True, otherwise apply it once.

        Args:
            conv: The SAGEConv layer to apply.
            x: Node feature tensor of shape [num_nodes, in_feats].
            edge_index: Edge index tensor of shape [2, num_edges].

        Returns:
            The output node features after applying the convolution (and averaging if bidirectional).
        """
        x_f = conv(x, edge_index)
        if not self.bidirectional:
            return x_f
        x_b = conv(x, edge_index.flip(0))
        return 0.5 * (x_f + x_b)

    def forward(self, data: Data) -> torch.Tensor:
        """Encode the input graph data into a graph embedding.

        Args:
            data: A PyG Data object containing at least 'x' (node features) and 'edge_index' (graph connectivity).

        Returns:
            A tensor of shape [num_graphs, graph_emb_dim] representing the encoded graph embeddings.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i, conv in enumerate(self.convs):
            x_new = self._apply_conv_bidir(conv, x, edge_index)
            x_new = self.norms[i](x_new, batch=batch)
            x_new = self.conv_activation(x_new, **self.conv_act_kwargs)
            x_new = self.drop(x_new)

            # residual only after the initial stack
            x = x_new if i < self._residual_start else (x + x_new)

        # graph readout
        return global_mean_pool(x, batch)  # [num_graphs, hidden_dim]


class SAGEActorCritic(nn.Module):
    """Actor-Critic using the SAGE encoder.

    Model for RL predictor, composed of a SAGEConv encoder followed by separate actor and critic MLP heads.
    """

    def __init__(
        self,
        in_feats: int,
        hidden_dim: int,
        num_conv_wo_resnet: int,
        num_resnet_layers: int,
        num_actions: int,
        *,
        dropout_p: float = 0.2,
        bidirectional: bool = True,
        global_feature_dim: int = 0,
    ) -> None:
        """Initialize the SAGEActorCritic model.

        Args:
            in_feats: Dimension of input node features.
            hidden_dim: Dimension of hidden layers and graph embedding.
            num_conv_wo_resnet: Number of initial convolutional layers without residual connections in the encoder.
            num_resnet_layers: Number of subsequent convolutional layers with residual connections in the encoder.
            num_actions: Number of discrete actions for the actor head output.
            dropout_p: Dropout probability to apply in the encoder and trunk.
            bidirectional: If True, apply each SAGEConv in both directions and average the results.
            global_feature_dim: If > 0, the dimension of optional global features to concatenate to the graph embedding before the actor and critic heads.
        """
        super().__init__()

        self.global_feature_dim = global_feature_dim
        # Same encoder for actor and critic
        self.encoder = GraphConvolutionSageEncoder(
            in_feats=in_feats,
            hidden_dim=hidden_dim,
            num_conv_wo_resnet=num_conv_wo_resnet,
            num_resnet_layers=num_resnet_layers,
            dropout_p=dropout_p,
            bidirectional=bidirectional,
        )

        emb_dim = self.encoder.graph_emb_dim
        # Input to trunk combines graph embedding with optional global features.
        trunk_in_dim = emb_dim + global_feature_dim
        # 1 Layer of FFNN after the encoder before actor and critic heads, to allow them to diverge more.
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in_dim, emb_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
        )
        # 2-layer MLP actor and critic heads with shared embedding dimension, no activation on output layer.
        self.actor = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LeakyReLU(),
            nn.Linear(emb_dim, num_actions),
        )

        self.critic = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LeakyReLU(),
            nn.Linear(emb_dim, 1),
        )

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns action logits and values.

        Args:
            data: A PyG Data or Batch object. When global_feature_dim > 0
                  the object must contain a global_features attribute of shape [B, global_feature_dim]
                  (one row per graph in the batch).
        """
        output_encoder = self.encoder(data)  # [B, emb_dim]

        if self.global_feature_dim > 0:
            global_features = getattr(data, "global_features", None)
            if global_features is not None:
                # Reshape to [B, global_feature_dim] in case PyG batched it differently.
                global_features = global_features.view(output_encoder.shape[0], self.global_feature_dim).to(
                    output_encoder.device
                )
                output_encoder = torch.cat(
                    [output_encoder, global_features], dim=-1
                )  # [B, emb_dim + global_feature_dim]
            else:
                # No global features available — pad with zeros so the model still runs.
                pad = torch.zeros(output_encoder.shape[0], self.global_feature_dim, device=output_encoder.device)
                output_encoder = torch.cat([output_encoder, pad], dim=-1)

        output_common_model = self.trunk(output_encoder)  # [B, emb_dim]
        logits = self.actor(output_common_model)  # Dimension batch, num_actions
        value = self.critic(output_common_model)  # Dimension batch, 1
        return logits, value
