import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(torch.nn.Module):
    """
    Implementación del modelo GraphSAGE para aprendizaje de representaciones de nodos en grafos.

    Parameters:
        in_channels (int): Dimensionalidad de las características de entrada.
        hidden_channels (int): Dimensionalidad de la capa oculta.
        out_channels (int): Dimensionalidad de las representaciones de salida.
        num_layers (int): Número de capas GraphSAGE (default es 2).
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super(GraphSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()

        # Primera capa: de in_channels a hidden_channels
        self.convs.append(SAGEConv(in_channels, hidden_channels))

        # Capas intermedias (si num_layers > 2)
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        # Última capa: de hidden_channels a out_channels
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        # Propagación en capas intermedias con activación ReLU
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        # Última capa sin activación
        x = self.convs[-1](x, edge_index)
        return x
