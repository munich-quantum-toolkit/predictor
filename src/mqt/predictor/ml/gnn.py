# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""This module contains the GNN module for graph neural networks."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch_geometric.nn import SAGEConv, global_mean_pool

if TYPE_CHECKING:
    from collections.abc import (
        Callable,  # on 3.10+ prefer collections.abc
    )

    from torch_geometric.data import Data


class GraphConvolutionSage(nn.Module):
    """Graph convolutional layer using SAGEConv."""

    def __init__(
        self,
        in_feats: int,
        hidden_dim: int,
        num_resnet_layers: int,
        *,
        conv_activation: Callable[..., torch.Tensor] = functional.leaky_relu,
        conv_act_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """A flexible SageConv graph classification model.

        Args:
          in_feats:        dimensionality of node features
          hidden_dim:      output size of SageConv
          num_resnet_layers:  how many SageConv layers (with residuals) to stack after the SageConvs
          mlp_units:       list of units for each layer of the final MLP
          conv_activation: activation fn after each graph layer
          conv_act_kwargs: extra kwargs for conv_activation
          final_activation: activation applied to the final scalar output
        """
        super().__init__()
        self.conv_activation = conv_activation
        self.conv_act_kwargs = conv_act_kwargs or {}

        # --- GRAPH ENCODER ---
        self.convs = nn.ModuleList()
        # 1) Convolution not in residual configuration
        # Possible to generalize the code
        self.convs.append(SAGEConv(in_feats, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        for _ in range(num_resnet_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

    def forward(self, data: Data) -> torch.Tensor:
        """Forward function that allows to elaborate the input graph."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # 1) Graph stack with residuals
        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index)
            x_new = self.conv_activation(x_new, **self.conv_act_kwargs)
            # the number 2 is set because two convolution without residual configuration are applied
            # and then all the others are in residual configuration
            x = x_new if i < 2 else x + x_new

        # 2) Global pooling
        return global_mean_pool(x, batch)


class GNN(nn.Module):
    """Architecture composed by a Graph Convolutional part with Sage Convolution module and followed by a MLP."""

    def __init__(
        self,
        in_feats: int,
        hidden_dim: int,
        num_resnet_layers: int,
        mlp_units: list[int],
        *,
        conv_activation: Callable[..., torch.Tensor] = functional.leaky_relu,
        conv_act_kwargs: dict[str, Any] | None = None,
        mlp_activation: Callable[..., torch.Tensor] = functional.leaky_relu,
        mlp_act_kwargs: dict[str, Any] | None = None,
        classes: list[str] | None = None,
        output_dim: int = 1,
    ) -> None:
        """Init class for the GNN.

        Arguments:
            in_feats: dimension of input features of the node
            hidden_dim: dimension of hidden output channels of the Convolutional part
            num_resnet_layers: number of residual layers
            mlp_units: list of units for each layer of the final MLP
            conv_activation: activation fn after each graph layer
            conv_act_kwargs: extra kwargs for conv_activation.
            mlp_activation: activation fn after each MLP layer
            mlp_act_kwargs: extra kwargs for mlp_activation.
            output_dim: dimension of the output, default is 1 for regression tasks
            classes: list of class names for classification tasks
        """
        super().__init__()
        # Convolutional part
        self.graph_conv = GraphConvolutionSage(
            in_feats, hidden_dim, num_resnet_layers, conv_activation=conv_activation, conv_act_kwargs=conv_act_kwargs
        )

        # MLP architecture
        self.mlp_activation = mlp_activation
        self.mlp_act_kwargs = mlp_act_kwargs or {}
        self.classes = classes
        self.fcs = nn.ModuleList()
        last_dim = hidden_dim
        for out_dim in mlp_units:
            self.fcs.append(nn.Linear(last_dim, out_dim))
            last_dim = out_dim
        self.out = nn.Linear(last_dim, output_dim)

    def forward(self, data: Data) -> torch.Tensor:
        """Forward function that allows to elaborate the input graph.

        Arguments:
            data: The input graph data.
        """
        # apply the convolution
        x = self.graph_conv(data)
        # Apply the MLP
        for fc in self.fcs:
            x = self.mlp_activation(fc(x), **self.mlp_act_kwargs)
        return self.out(x)
