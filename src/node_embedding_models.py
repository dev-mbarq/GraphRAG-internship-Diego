import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(torch.nn.Module):
    """
    Implementation of the GraphSAGE model for node representation learning in graphs.

    Parameters:
        channels (list of int): List of layer dimensions, including input and output.
    """

    def __init__(self, channels):
        super(GraphSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()

        # Create SAGEConv layers using the dimensions from the channels list
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
