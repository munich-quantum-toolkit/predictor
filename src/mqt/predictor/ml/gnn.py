# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Graph neural network models using SAGEConv layers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch_geometric.nn import (
    GraphNorm,
    SAGEConv,
    SAGPooling,
    global_mean_pool,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch_geometric.data import Data


class GraphConvolutionSage(nn.Module):
    """Graph convolutional encoder using SAGEConv layers."""

    def __init__(
        self,
        in_feats: int,
        hidden_dim: int,
        num_conv_wo_resnet: int,
        num_resnet_layers: int,
        *,
        conv_activation: Callable[..., torch.Tensor] = functional.leaky_relu,
        conv_act_kwargs: dict[str, Any] | None = None,
        dropout_p: float = 0.2,
        bidirectional: bool = True,
        use_sag_pool: bool = False,
        sag_ratio: float = 0.7,
        sag_nonlinearity: Callable[..., torch.Tensor] = torch.tanh,
    ) -> None:
        """Initialize the graph convolutional encoder.

        The encoder consists of a stack of SAGEConv layers followed by
        optional SAGPooling before the global readout.

        Args:
            in_feats: Dimensionality of the node features.
            hidden_dim: Output dimensionality of the first SAGEConv layer.
            num_conv_wo_resnet: Number of SAGEConv layers before residual
                connections are introduced.
            num_resnet_layers: Number of SAGEConv layers with residual
                connections.
            conv_activation: Activation function applied after each graph
                convolution. Defaults to torch.nn.functional.leaky_relu.
            conv_act_kwargs: Additional keyword arguments passed to
                conv_activation. Defaults to None.
            dropout_p: Dropout probability applied after each graph layer.
                Defaults to 0.2.
            bidirectional: If True, apply message passing in both
                directions (forward and reversed edges) and average the
                results. Defaults to True.
            use_sag_pool: If True, apply a single SAGPooling layer after
                the convolutions and before readout. Defaults to False.
            sag_ratio: Fraction of nodes to keep in SAGPooling. Must be in
                (0, 1]. Defaults to 0.7.
            sag_nonlinearity: Nonlinearity used inside SAGPooling for score
                computation. Defaults to torch.tanh.
        """
        super().__init__()

        if num_conv_wo_resnet < 1:
            msg = "num_conv_wo_resnet must be at least 1"
            raise ValueError(msg)

        self.conv_activation = conv_activation
        self.conv_act_kwargs = conv_act_kwargs or {}
        self.bidirectional = bidirectional
        self.use_sag_pool = use_sag_pool

        # --- GRAPH ENCODER ---
        self.convs: nn.ModuleList[SAGEConv] = nn.ModuleList()
        self.norms: nn.ModuleList[GraphNorm] = nn.ModuleList()

        # First layer: SAGE
        self.convs.append(SAGEConv(in_feats, hidden_dim))
        out_dim = hidden_dim
        self.graph_emb_dim = out_dim
        self.norms.append(GraphNorm(out_dim))

        # Subsequent layers: SAGE with fixed width == out_dim
        for _ in range(num_conv_wo_resnet - 1):
            self.convs.append(SAGEConv(out_dim, out_dim))
            self.norms.append(GraphNorm(out_dim))
        for _ in range(num_resnet_layers):
            self.convs.append(SAGEConv(out_dim, out_dim))
            self.norms.append(GraphNorm(out_dim))

        self.drop = nn.Dropout(dropout_p)
        # Start residuals after the initial non-residual stack
        self._residual_start = num_conv_wo_resnet
        # Expose the final node embedding width
        self.out_dim = out_dim

        # --- SAGPooling layer (applied once, after all convs) ---
        # Uses SAGEConv internally for attention scoring to match the stack.
        if self.use_sag_pool:
            if not (0.0 < sag_ratio <= 1.0):
                msg = "sag_ratio must be in (0, 1]"
                raise ValueError(msg)
            self.sag_pool: SAGPooling | None = SAGPooling(
                in_channels=self.out_dim,
                ratio=sag_ratio,
                GNN=SAGEConv,
                nonlinearity=sag_nonlinearity,
            )
        else:
            self.sag_pool = None

    def _apply_conv_bidir(
        self,
        conv: SAGEConv,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a SAGEConv layer in forward and backward directions and average.

        Args:
            conv: Convolution layer taken from self.convs.
            x: Node feature matrix of shape [num_nodes, in_channels].
            edge_index: Edge index tensor of shape [2, num_edges].

        Returns:
            Tensor with updated node features of shape
            [num_nodes, out_channels].
        """
        x_f = conv(x, edge_index)
        if not self.bidirectional:
            return x_f
        x_b = conv(x, edge_index.flip(0))
        return (x_f + x_b) / 2

    def forward(self, data: Data) -> torch.Tensor:
        """Encode a batch of graphs and return pooled graph embeddings.

        The input batch of graphs is processed by the SAGEConv stack,
        optionally followed by SAGPooling, and finally aggregated with
        global mean pooling.

        Args:
            data: Batched torch_geometric.data.Data object.
                Expected attributes:
                - x: Node features of shape [num_nodes, in_feats].
                - edge_index: Edge indices of shape [2, num_edges].
                - batch: Graph indices for each node of shape
                  [num_nodes].

        Returns:
            Tensor of shape [num_graphs, out_dim] containing one embedding
            per input graph.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i, conv in enumerate(self.convs):
            x_new = self._apply_conv_bidir(conv, x, edge_index)
            x_new = self.norms[i](x_new, batch=batch)
            x_new = self.conv_activation(x_new, **self.conv_act_kwargs)
            x_new = self.drop(x_new)

            x = x_new if i < self._residual_start else x + x_new

        # --- SAGPooling (hierarchical pooling before readout) ---
        if self.sag_pool is not None:
            # SAGPooling may also return edge_attr, perm, score; we ignore those here.
            x, edge_index, _, batch, _, _ = self.sag_pool(
                x,
                edge_index,
                batch=batch,
            )

        return global_mean_pool(x, batch)


class GNN(nn.Module):
    """Graph neural network with a SAGE-based encoder and MLP head.

    This model first encodes each input graph using GraphConvolutionSage
    and then applies a feed-forward neural network to the resulting graph
    embeddings to produce the final prediction.
    """

    def __init__(
        self,
        in_feats: int,
        hidden_dim: int,
        num_conv_wo_resnet: int,
        num_resnet_layers: int,
        mlp_units: list[int],
        *,
        conv_activation: Callable[..., torch.Tensor] = functional.leaky_relu,
        conv_act_kwargs: dict[str, Any] | None = None,
        mlp_activation: Callable[..., torch.Tensor] = functional.leaky_relu,
        mlp_act_kwargs: dict[str, Any] | None = None,
        dropout_p: float = 0.2,
        bidirectional: bool = True,
        output_dim: int = 1,
        use_sag_pool: bool = False,
        sag_ratio: float = 0.7,
        sag_nonlinearity: Callable[..., torch.Tensor] = torch.tanh,
    ) -> None:
        """Initialize the GNN model.

        Args:
            in_feats: Dimensionality of the input node features.
            hidden_dim: Hidden dimensionality of the SAGEConv layers.
            num_conv_wo_resnet: Number of SAGEConv layers before residual
                connections are introduced in the encoder.
            num_resnet_layers: Number of SAGEConv layers with residual
                connections in the encoder.
            mlp_units: List specifying the number of units in each hidden
                layer of the MLP head.
            conv_activation: Activation function applied after each graph
                convolution. Defaults to torch.nn.functional.leaky_relu.
            conv_act_kwargs: Additional keyword arguments passed to
                conv_activation. Defaults to None.
            mlp_activation: Activation function applied after each MLP layer.
                Defaults to torch.nn.functional.leaky_relu.
            mlp_act_kwargs: Additional keyword arguments passed to
                mlp_activation. Defaults to None.
            dropout_p: Dropout probability applied in the model (graph encoder and the MLP).
                Defaults to 0.2.
            bidirectional: If True, apply bidirectional message passing in
                the encoder. Defaults to True.
            output_dim: Dimensionality of the model output (e.g. number of
                targets per graph). Defaults to 1.
            use_sag_pool: If True, enable SAGPooling in the encoder.
                Defaults to False.
            sag_ratio: Fraction of nodes to keep in SAGPooling. Must be in
                (0, 1]. Defaults to 0.7.
            sag_nonlinearity: Nonlinearity used inside SAGPooling for score
                computation. Defaults to torch.tanh.
        """
        super().__init__()

        # Graph encoder
        self.graph_conv = GraphConvolutionSage(
            in_feats=in_feats,
            hidden_dim=hidden_dim,
            num_conv_wo_resnet=num_conv_wo_resnet,
            num_resnet_layers=num_resnet_layers,
            conv_activation=conv_activation,
            conv_act_kwargs=conv_act_kwargs,
            dropout_p=dropout_p,
            bidirectional=bidirectional,
            use_sag_pool=use_sag_pool,
            sag_ratio=sag_ratio,
            sag_nonlinearity=sag_nonlinearity,
        )

        self.mlp_activation = mlp_activation
        self.mlp_act_kwargs = mlp_act_kwargs or {}
        last_dim = self.graph_conv.graph_emb_dim
        self.mlp_drop = nn.Dropout(dropout_p)
        self.fcs: nn.ModuleList[nn.Linear] = nn.ModuleList()
        for out_dim_ in mlp_units:
            self.fcs.append(nn.Linear(last_dim, out_dim_))
            last_dim = out_dim_
        self.out = nn.Linear(last_dim, output_dim)

    def forward(self, data: Data) -> torch.Tensor:
        """Compute predictions for a batch of graphs.

        The input graphs are encoded into graph embeddings by the
        GraphConvolutionSage encoder, then passed through the MLP head
        to obtain final predictions.

        Args:
            data: Batched torch_geometric.data.Data object
                containing the graphs to be evaluated.

        Returns:
            Tensor of shape [num_graphs, output_dim] with the model
            predictions for each graph in the batch.
        """
        x = self.graph_conv(data)
        for fc in self.fcs:
            x = self.mlp_drop(self.mlp_activation(fc(x), **self.mlp_act_kwargs))
        return self.out(x)
