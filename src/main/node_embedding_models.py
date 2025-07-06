import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GraphNorm
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import GATConv


class GraphSAGE(torch.nn.Module):
    """
    Implementation of the GraphSAGE model for node representation learning in graphs.

    Parameters:
        channels (list of int): List of layer dimensions, including input and output.
        
    """

    def __init__(self, channels):
        super(GraphSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()

        # Create SAGEConv layers using the dimensions from the channels list - Old - No layer norm
        for i in range(len(channels) - 1):
            self.convs.append(SAGEConv(channels[i], channels[i + 1]))

    def forward(self, x, edge_index):
        # Propagation through intermediate layers with ReLU activation
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        # Last layer without activation
        x = self.convs[-1](x, edge_index)
        return x
    

class GraphSAGE_V2(torch.nn.Module):
    """
    Implementation of the GraphSAGE model for node representation learning in graphs.
    Adds support for layer norm compared to V1.

    Parameters:
        channels (list of int): List of layer dimensions, including input and output.

    """

    def __init__(self, channels):
        super(GraphSAGE_V2, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        # Create SAGEConv + GraphNorm for each pair of dimensions
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            self.convs.append(SAGEConv(in_ch, out_ch))
            self.norms.append(GraphNorm(out_ch))


    def forward(self, x, edge_index):
        # Propagation through intermediate layers with ReLU activation
        for conv, norm in zip(self.convs[:-1], self.norms[:-1]):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
        # Last layer without activation
        x = self.convs[-1](x, edge_index) 
        return x


class GraphSAGE_ResidualMixing(torch.nn.Module):
    """
    GraphSAGE with residual (skip) connections and layer normalization.

    Args:
        channels (List[int]): list of feature sizes, e.g. [in_dim, hidden_dim, out_dim].
    """
    def __init__(self, channels):
        super().__init__()
        self.convs  = torch.nn.ModuleList()
        self.norms  = torch.nn.ModuleList()
        self.resids = torch.nn.ModuleList()

        # Build one SAGEConv + GraphNorm + optional residual projector per layer
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            self.convs.append(SAGEConv(in_ch, out_ch))
            self.norms.append(GraphNorm(out_ch))

            # If dimensions differ, project input to match out_ch, else identity
            if in_ch != out_ch:
                self.resids.append(torch.nn.Linear(in_ch, out_ch))
            else:
                self.resids.append(torch.nn.Identity())

    def forward(self, x, edge_index):
        # Apply all but last layer with residual + ReLU
        for conv, norm, resid in zip(self.convs[:-1],
                                     self.norms[:-1],
                                     self.resids[:-1]):
            h_in = x                          # save the layer’s input
            h    = conv(h_in, edge_index)    # message-passing step
            h    = norm(h)                   # layer normalization
            h    = h + resid(h_in)           # residual (skip) connection
            x    = F.relu(h)                 # non-linearity

        # Final layer without activation or residual
        x = self.convs[-1](x, edge_index)
        return x


class GraphSAGE_V3_mean(torch.nn.Module):
    """
    GraphSAGE with:
      - Residual (skip) connections with learnable mix weights
      - Layer normalization
      - Feature dropout
      - Edge dropout (DropEdge)
    """
    def __init__(self, channels, drop_edge_rate=0.2, dropout_rate=0.2):
        super().__init__()
        self.drop_edge_rate = drop_edge_rate
        self.dropout_rate = dropout_rate

        self.convs  = torch.nn.ModuleList()
        self.norms  = torch.nn.ModuleList()
        self.resids = torch.nn.ModuleList()
        self.alphas = torch.nn.ParameterList()

        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            self.convs.append(SAGEConv(in_ch, out_ch, aggr='mean'))
            self.norms.append(GraphNorm(out_ch))

            # RESIDUAL PROJECTOR (NEW)
            if in_ch != out_ch:
                self.resids.append(torch.nn.Linear(in_ch, out_ch))
            else:
                self.resids.append(torch.nn.Identity())

            # LEARNABLE SKIP WEIGHT α (NEW)
            alpha = torch.nn.Parameter(torch.tensor(0.5))
            self.alphas.append(alpha)

    def forward(self, x, edge_index):
        for conv, norm, resid, alpha in zip(
            self.convs[:-1], self.norms[:-1], self.resids[:-1], self.alphas[:-1]
        ):
            h_in = x

            # EDGE DROPOUT (NEW)
            if self.training and self.drop_edge_rate > 0:
                edge_index, _ = dropout_edge(
                    edge_index, p=self.drop_edge_rate, force_undirected=True
                )

            # GRAPH SAGE + NORM
            h = conv(h_in, edge_index)
            h = norm(h)

            # RESIDUAL MIXING WITH LEARNABLE WEIGHT α (NEW)
            h = alpha * h + (1.0 - alpha) * resid(h_in)

            # ACTIVATION
            x = F.relu(h)

            # FEATURE DROPOUT (NEW)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # FINAL LAYER WITHOUT RESIDUAL OR ACTIVATION
        if self.training and self.drop_edge_rate > 0:
            edge_index, _ = dropout_edge(
                edge_index, p=self.drop_edge_rate, force_undirected=True
            )
        x = self.convs[-1](x, edge_index)
        return x
    

class GraphSAGE_V3_max(torch.nn.Module):
    """
    GraphSAGE with:
      - Residual (skip) connections with learnable mix weights
      - Layer normalization
      - Feature dropout
      - Edge dropout (DropEdge)
    """
    def __init__(self, channels, drop_edge_rate=0.2, dropout_rate=0.2):
        super().__init__()
        self.drop_edge_rate = drop_edge_rate
        self.dropout_rate = dropout_rate

        self.convs  = torch.nn.ModuleList()
        self.norms  = torch.nn.ModuleList()
        self.resids = torch.nn.ModuleList()
        self.alphas = torch.nn.ParameterList()

        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            self.convs.append(SAGEConv(in_ch, out_ch, aggr='max'))
            self.norms.append(GraphNorm(out_ch))

            # RESIDUAL PROJECTOR (NEW)
            if in_ch != out_ch:
                self.resids.append(torch.nn.Linear(in_ch, out_ch))
            else:
                self.resids.append(torch.nn.Identity())

            # LEARNABLE SKIP WEIGHT α (NEW)
            alpha = torch.nn.Parameter(torch.tensor(0.5))
            self.alphas.append(alpha)

    def forward(self, x, edge_index):
        for conv, norm, resid, alpha in zip(
            self.convs[:-1], self.norms[:-1], self.resids[:-1], self.alphas[:-1]
        ):
            h_in = x

            # EDGE DROPOUT (NEW)
            if self.training and self.drop_edge_rate > 0:
                edge_index, _ = dropout_edge(
                    edge_index, p=self.drop_edge_rate, force_undirected=True
                )

            # GRAPH SAGE + NORM
            h = conv(h_in, edge_index)
            h = norm(h)

            # RESIDUAL MIXING WITH LEARNABLE WEIGHT α (NEW)
            h = alpha * h + (1.0 - alpha) * resid(h_in)

            # ACTIVATION
            x = F.relu(h)

            # FEATURE DROPOUT (NEW)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # FINAL LAYER WITHOUT RESIDUAL OR ACTIVATION
        if self.training and self.drop_edge_rate > 0:
            edge_index, _ = dropout_edge(
                edge_index, p=self.drop_edge_rate, force_undirected=True
            )
        x = self.convs[-1](x, edge_index)
        return x
    

class GraphSAGE_V3_lstm(torch.nn.Module):
    """
    GraphSAGE with:
      - Residual (skip) connections with learnable mix weights
      - Layer normalization
      - Feature dropout
      - Edge dropout (DropEdge)
    """
    def __init__(self, channels, drop_edge_rate=0.2, dropout_rate=0.2):
        super().__init__()
        self.drop_edge_rate = drop_edge_rate
        self.dropout_rate = dropout_rate

        self.convs  = torch.nn.ModuleList()
        self.norms  = torch.nn.ModuleList()
        self.resids = torch.nn.ModuleList()
        self.alphas = torch.nn.ParameterList()

        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            self.convs.append(SAGEConv(in_ch, out_ch, aggr='lstm'))
            self.norms.append(GraphNorm(out_ch))

            # RESIDUAL PROJECTOR (NEW)
            if in_ch != out_ch:
                self.resids.append(torch.nn.Linear(in_ch, out_ch))
            else:
                self.resids.append(torch.nn.Identity())

            # LEARNABLE SKIP WEIGHT α (NEW)
            alpha = torch.nn.Parameter(torch.tensor(0.5))
            self.alphas.append(alpha)

    def forward(self, x, edge_index):
        for conv, norm, resid, alpha in zip(
            self.convs[:-1], self.norms[:-1], self.resids[:-1], self.alphas[:-1]
        ):
            h_in = x

            # EDGE DROPOUT (NEW)
            if self.training and self.drop_edge_rate > 0:
                edge_index, _ = dropout_edge(
                    edge_index, p=self.drop_edge_rate, force_undirected=True
                )

            # GRAPH SAGE + NORM
            h = conv(h_in, edge_index)
            h = norm(h)

            # RESIDUAL MIXING WITH LEARNABLE WEIGHT α (NEW)
            h = alpha * h + (1.0 - alpha) * resid(h_in)

            # ACTIVATION
            x = F.relu(h)

            # FEATURE DROPOUT (NEW)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # FINAL LAYER WITHOUT RESIDUAL OR ACTIVATION
        if self.training and self.drop_edge_rate > 0:
            edge_index, _ = dropout_edge(
                edge_index, p=self.drop_edge_rate, force_undirected=True
            )
        x = self.convs[-1](x, edge_index)
        return x
    

class GAT_V1(torch.nn.Module):
    """
    Vanilla Graph Attention Network for node embeddings.

    Args:
        channels (List[int]): list of feature sizes, e.g. [in_dim, hidden_dim, out_dim].
        heads (int): number of attention heads in each intermediate layer.
        dropout_rate (float): node‐feature dropout probability.
    """
    def __init__(self, channels, heads=4, dropout_rate=0.2):
        super().__init__()
        self.dropout_rate = dropout_rate

        # Build attention layers
        self.convs = torch.nn.ModuleList()
        # First L-1 layers: multi‐head, with concat
        for in_ch, out_ch in zip(channels[:-2], channels[1:-1]):
            self.convs.append(
                GATConv(in_ch, out_ch // heads, heads=heads, dropout=dropout_rate)
            )
        # Final layer: single head, output dim exactly channels[-1]
        self.convs.append(
            GATConv(channels[-2], channels[-1], heads=1, concat=False, dropout=dropout_rate)
        )

    def forward(self, x, edge_index):
        # Apply all but last layer
        for conv in self.convs[:-1]:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = conv(x, edge_index)
            x = F.elu(x)

        # Last layer (no activation)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
    

class GraphSAGE_V3_max_norm(torch.nn.Module):
    """
    GraphSAGE with:
      - Residual (skip) connections with learnable mix weights
      - Layer normalization
      - Feature dropout
      - Edge dropout (DropEdge)
    """
    def __init__(self, channels, drop_edge_rate=0.2, dropout_rate=0.2):
        super().__init__()
        self.drop_edge_rate = drop_edge_rate
        self.dropout_rate = dropout_rate

        self.convs  = torch.nn.ModuleList()
        self.norms  = torch.nn.ModuleList()
        self.resids = torch.nn.ModuleList()
        self.alphas = torch.nn.ParameterList()

        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            self.convs.append(SAGEConv(in_ch, out_ch, aggr='max'))
            self.norms.append(GraphNorm(out_ch))

            # RESIDUAL PROJECTOR (NEW)
            if in_ch != out_ch:
                self.resids.append(torch.nn.Linear(in_ch, out_ch))
            else:
                self.resids.append(torch.nn.Identity())

            # LEARNABLE SKIP WEIGHT α (NEW)
            alpha = torch.nn.Parameter(torch.tensor(0.5))
            self.alphas.append(alpha)

    def forward(self, x, edge_index):
        for conv, norm, resid, alpha in zip(
            self.convs[:-1], self.norms[:-1], self.resids[:-1], self.alphas[:-1]
        ):
            h_in = x

            # EDGE DROPOUT (NEW)
            if self.training and self.drop_edge_rate > 0:
                edge_index, _ = dropout_edge(
                    edge_index, p=self.drop_edge_rate, force_undirected=True
                )

            # GRAPH SAGE + NORM
            h = conv(h_in, edge_index)
            h = norm(h)

            # RESIDUAL MIXING WITH LEARNABLE WEIGHT α (NEW)
            h = alpha * h + (1.0 - alpha) * resid(h_in)

            # ACTIVATION
            x = F.relu(h)

            # FEATURE DROPOUT (NEW)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # FINAL LAYER WITHOUT RESIDUAL OR ACTIVATION
        if self.training and self.drop_edge_rate > 0:
            edge_index, _ = dropout_edge(
                edge_index, p=self.drop_edge_rate, force_undirected=True
            )
        x = self.convs[-1](x, edge_index)
        x = self.norms[-1](x)                     # FINAL LAYER NORM
        x = F.normalize(x, p=2, dim=1)           # UNIT-NORM EMBEDDINGS (NEW)
        return x