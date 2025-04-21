import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GraphNorm


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
