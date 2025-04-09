import networkx as nx
import numpy as np
import torch


def prepare_graph_for_gnn(G, embedding_dim=1024):
    """
    Preprocess a NetworkX graph for GNN training by ensuring all nodes have only the 'embedding' attribute,
    converting embeddings to torch.Tensor, and making the graph undirected.

    Parameters:
    -----------
    G : networkx.Graph
        The input graph.
    embedding_dim : int
        Expected dimensionality of each node's embedding.

    Returns:
    --------
    networkx.Graph
        A cleaned, undirected graph where each node has an embedding as a `torch.Tensor`.

    list
        A list of node IDs that do not have the expected embedding dimension.

    Notes:
    ------
    - All node attributes except 'embedding' are removed.
    - Embeddings are converted to `torch.Tensor` with dtype `torch.float32`.
    - The graph is converted to an undirected format.
    - Nodes with incorrect embedding dimensions are identified and returned.
    """

    # Convert the graph to an undirected version
    G_undirected = nx.Graph(nx.to_undirected(G))

    # List to store nodes with incorrect embedding dimensions
    incorrect_dim_nodes = []

    # Iterate over nodes to clean and check embeddings
    for node in list(G_undirected.nodes()):
        node_attrs = G_undirected.nodes[node]
        embedding = node_attrs.get("embedding", None)

        # Remove all attributes except 'embedding'
        keys_to_remove = [k for k in node_attrs.keys() if k != "embedding"]
        for k in keys_to_remove:
            del node_attrs[k]

        # Check if the embedding exists
        if embedding is not None:
            # Convert to numpy array if it is a list
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
            elif isinstance(embedding, np.ndarray):
                embedding = embedding.astype(np.float32)

            # Convert to torch.Tensor if necessary
            if not isinstance(embedding, torch.Tensor):
                embedding = torch.tensor(embedding, dtype=torch.float32)

            # Ensure the embedding has the expected shape
            if embedding.dim() != 1 or embedding.shape[0] != embedding_dim:
                incorrect_dim_nodes.append(node)

            # Assign cleaned embedding back to the node
            node_attrs["embedding"] = embedding
        else:
            # If no embedding, add to incorrect dimension list
            incorrect_dim_nodes.append(node)

    # Return the undirected graph and list of nodes with incorrect dimensions
    return G_undirected, incorrect_dim_nodes
