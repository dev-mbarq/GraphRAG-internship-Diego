import torch
import torch.nn.functional as F
import random


def unsupervised_loss_V0(z, edge_index, num_neg_samples=5):
    """
    Computes the unsupervised loss using contrastive negative sampling.

    The loss function is designed to encourage high dot-product similarity
    between embeddings of connected nodes (positive pairs) and low similarity
    between embeddings of non-connected nodes (negative pairs). For every
    positive edge (u, v), the loss accumulates:
      - The log-probability of the positive pair: log(sigmoid(dot(z[u], z[v])))
      - For a specified number of negative samples, the log-probability of the
        negative pair: log(1 - sigmoid(dot(z[u], z[v_neg])))
    Parameters:
        z (torch.Tensor): Node embeddings with shape (num_nodes, embedding_dim).
        edge_index (torch.Tensor): Connectivity matrix of shape (2, num_edges),
                                   where each column represents an edge (u, v).
        num_neg_samples (int): Number of negative samples per edge (default is 5).

    Returns:
        torch.Tensor: The averaged loss value as a tensor.
    """
    # Initialize accumulators for the positive and negative loss components
    pos_loss = torch.tensor(0.0, device=z.device)
    neg_loss = torch.tensor(0.0, device=z.device)

    # Get the total number of nodes from the shape of the embeddings tensor
    num_nodes = z.shape[0]

    # Iterate over each edge in the graph.
    # edge_index.T iterates column-wise over edges, where each edge is a tensor of shape (2,)
    for edge in edge_index.T:
        u, v = edge  # u and v are connected nodes (positive pair)

        # Positive term:
        # Compute the dot product between the embeddings of connected nodes u and v.
        # Then calculate log(sigmoid(dot_product)).
        pos_loss += torch.log(torch.sigmoid(torch.dot(z[u], z[v])))

        # Negative sampling: For each positive edge, sample a number of negative nodes.
        for _ in range(num_neg_samples):
            # Randomly sample a negative node index from 0 to (num_nodes - 1)
            v_neg = random.randint(0, num_nodes - 1)

            # Ensure that the negative sample is not a neighbor.
            # Note: This condition checks if the sampled node is in the global set
            # of destination nodes. In practice, you should adjust this condition to
            # check the neighbors of u specifically to avoid infinite loops.
            while v_neg in edge_index[1]:
                v_neg = random.randint(0, num_nodes - 1)

            # Negative term:
            # Compute the dot product between the embedding of u and that of the negative sample.
            # Calculate log(1 - sigmoid(dot_product)), which is equivalent to log(sigmoid(-dot_product)).
            neg_loss += torch.log(1 - torch.sigmoid(torch.dot(z[u], z[v_neg])))

    # Compute the final loss:
    # Sum the positive and negative losses for all edges, negate the sum (since we want to minimize loss),
    # and average it by dividing by the total number of positive edges (columns in edge_index).
    loss = -(pos_loss + neg_loss) / edge_index.shape[1]

    return loss


def unsupervised_loss_V1(z, edge_index, num_neg_samples=5):
    """
    Computes the unsupervised loss for GraphSAGE using contrastive negative sampling.

    The loss encourages high dot-product similarity between connected node embeddings
    (using log(sigmoid(dot_product))) and low similarity between embeddings of non-connected nodes
    (using log(sigmoid(-dot_product))). Negative samples are drawn from nodes that are not neighbors
    of the source node.

    Improvements over the naive implementation:
      - Uses F.logsigmoid for improved numerical stability.
      - Precomputes a neighbor set per node to efficiently check for negative samples.
      - Ensures that a node does not sample itself as a negative example.

    Parameters:
        z (torch.Tensor): Node embeddings of shape (num_nodes, embedding_dim).
        edge_index (torch.Tensor): Connectivity matrix of shape (2, num_edges), where each column represents an edge (u, v).
        num_neg_samples (int): Number of negative samples per edge (default: 5).

    Returns:
        torch.Tensor: The averaged unsupervised loss, which requires gradients.
    """
    device = z.device
    num_nodes = z.shape[0]

    # Precompute a dictionary mapping each node to its set of neighbors.
    # Note: Assuming the graph is undirected, we add the edge in both directions.
    neighbors = {u: set() for u in range(num_nodes)}
    for edge in edge_index.T:
        u, v = edge.tolist()
        neighbors[u].add(v)
        neighbors[v].add(u)

    # Initialize accumulators for the positive and negative loss components as PyTorch tensors.
    pos_loss = torch.tensor(0.0, device=device)
    neg_loss = torch.tensor(0.0, device=device)
    total_edges = edge_index.shape[1]

    # Loop over each positive edge
    for edge in edge_index.T:
        u, v = edge.tolist()  # Extract node indices
        # Positive term: compute dot product for connected nodes and apply logsigmoid for stability.
        dot_pos = torch.dot(z[u], z[v])
        pos_loss += F.logsigmoid(dot_pos)

        # Generate negative samples for the current edge.
        for _ in range(num_neg_samples):
            # Sample a negative node v_neg which is NOT a neighbor of u and is not u.
            v_neg = random.randint(0, num_nodes - 1)
            while v_neg in neighbors[u] or v_neg == u:
                v_neg = random.randint(0, num_nodes - 1)
            dot_neg = torch.dot(z[u], z[v_neg])
            neg_loss += F.logsigmoid(-dot_neg)

    # Compute the final loss:
    # Sum the positive and negative losses, negate the sum (for minimization),
    # and average it by dividing by the total number of positive edges.
    loss = -(pos_loss + neg_loss) / total_edges

    # Return the loss tensor (already connected to the computation graph)
    return loss
