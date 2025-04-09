import torch
import random


def unsupervised_loss(z, edge_index, num_neg_samples=5):
    """
    Calcula la pérdida no supervisada utilizando muestreo negativo basado en contraste.

    Parameters:
        z (torch.Tensor): Embeddings de nodos de forma (num_nodes, embedding_dim).
        edge_index (torch.Tensor): Matriz de conectividad (2, num_edges).
        num_neg_samples (int): Número de muestras negativas por arista (default 5).

    Returns:
        torch.Tensor: Valor de la pérdida calculada.
    """
    pos_loss = torch.tensor(0.0, device=z.device)
    neg_loss = torch.tensor(0.0, device=z.device)
    num_nodes = z.shape[0]

    for edge in edge_index.T:
        u, v = edge  # Nodos conectados
        pos_loss += torch.log(torch.sigmoid(torch.dot(z[u], z[v])))

        # Muestreo negativo
        for _ in range(num_neg_samples):
            v_neg = random.randint(0, num_nodes - 1)
            # Asegurar que no es vecino (puede ajustarse la condición según la estructura real)
            while v_neg in edge_index[1]:
                v_neg = random.randint(0, num_nodes - 1)
            neg_loss += torch.log(1 - torch.sigmoid(torch.dot(z[u], z[v_neg])))

    loss = -(pos_loss + neg_loss) / edge_index.shape[1]
    return loss
