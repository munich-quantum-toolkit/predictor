import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from typing import Callable, List, Optional
from torch_geometric.data import Data
from typing import Any, Optional, Sequence
from collections.abc import Callable  # on 3.10+ prefer collections.abc

class GraphConvolution_Sage(nn.Module): 
    def __init__(
        self,
        in_feats: int,
        hidden_dim: int,
        num_resnet_layers: int,
        *,
        conv_activation: Callable[..., torch.Tensor] = F.leaky_relu,
        conv_act_kwargs: Optional[dict[str, Any]] = None,
        
    ) -> None:
        """
        A flexible SageConv graph classification model.

        Args:
          in_feats:        dimensionality of node features
          hidden_dim:      output size of SageConv
          num_resnet_layers:  how many SageConv layers (with residuals) to stack after the SageConvs
          mlp_units:       list of hidden‐layer sizes for the final MLP
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
        x, edge_index, batch = data.X, data.edge_index, data.batch
        # 1) Graph stack with residuals
        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index)
            x_new = self.conv_activation(x_new, **self.conv_act_kwargs)
            # the number 2 is set because two convolution without residual configuration are applied
            # and then all the others are in residual configuration
            x = x_new if i < 2 else x + x_new

        # 2) Global pooling
        x = global_mean_pool(x, batch)

        # 3) MLP head
        return x
    
class GraphClassifier(nn.Module):
    def __init__(
        self,
        in_feats: int,
        hidden_dim: int,
        num_resnet_layers: int,
        mlp_units: List[int],
        *,
        conv_activation: Callable[..., torch.Tensor] = F.leaky_relu,
        conv_act_kwargs: Optional[dict[str, Any]] = None,
        mlp_activation: Callable[..., torch.Tensor] = F.leaky_relu,
        mlp_act_kwargs: Optional[dict[str, Any]] = None,
        final_activation: Callable[..., torch.Tensor] = torch.sigmoid,
        output_dim: int = 1
    ) -> None:
        super().__init__()
        # Convolutional part
        self.graph_conv = GraphConvolution_Sage(
            in_feats, hidden_dim, num_resnet_layers, 
            conv_activation=conv_activation, 
            conv_act_kwargs=conv_act_kwargs
        )

        # MLP architecture
        self.mlp_activation = mlp_activation
        self.mlp_act_kwargs = mlp_act_kwargs or {}
        self.final_activation = final_activation
        self.fcs = nn.ModuleList()
        last_dim = hidden_dim
        for out_dim in mlp_units:
            self.fcs.append(nn.Linear(last_dim, out_dim))
            last_dim = out_dim
        self.out = nn.Linear(last_dim, output_dim)

    def forward(self, data: Data)->torch.Tensor:

        # apply the convolution
        x = self.graph_conv(data)
        # Apply the MLP
        for fc in self.fcs:
            x = self.mlp_activation(fc(x), **self.mlp_act_kwargs)
        x = self.out(x)
        
        return x.squeeze(1)
        
        
