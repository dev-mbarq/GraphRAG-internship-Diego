import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


# Define the GraphSAGE model class
class GraphSAGE(torch.nn.Module):
    """
    Implementation of the GraphSAGE model for node representation learning in graphs.

    Parameters:
    -----------
    in_channels : int
        The dimensionality of input node features (e.g., embedding size).
    hidden_channels : int
        The dimensionality of hidden layers in the GraphSAGE model.
    out_channels : int
        The dimensionality of the output node representations.
    num_layers : int, optional
        The number of GraphSAGE layers (default is 2).

    Methods:
    --------
    forward(x, edge_index):
        Performs forward propagation through the GraphSAGE layers.

    Returns:
    --------
    torch.Tensor
        The learned node embeddings of shape (num_nodes, out_channels).

    Notes:
    ------
    - The first layer transforms input embeddings into a hidden representation.
    - Intermediate layers apply non-linear transformations (`ReLU` activation).
    - The final layer outputs node embeddings without activation.
    - Uses `SAGEConv` layers from PyTorch Geometric (`torch_geometric.nn`).
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super(GraphSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        
        # First GraphSAGE layer: input (embeddings) → hidden layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        
        # Intermediate layers (if num_layers > 2)
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        # Last GraphSAGE layer: hidden layer → final embedding
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:  # Intermediate layers
            x = conv(x, edge_index)
            x = F.relu(x)  # ReLU activation
        x = self.convs[-1](x, edge_index)  # Last layer (no activation)
        return x