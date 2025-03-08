import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected

def format_graph_for_graphsage(G, embedding_dim=1024, remove_incomplete=False):
    """
    Preprocess a NetworkX graph for GraphSAGE training by ensuring all nodes have embeddings,
    converting it to an undirected format, and removing unnecessary attributes.

    Parameters:
    -----------
    G : networkx.Graph
        The input graph.
    embedding_dim : int, optional
        Expected dimensionality of each node's embedding (default is 1024).
    remove_incomplete : bool, optional
        If True, nodes without embeddings are removed.
        If False, they are assigned a default zero embedding.

    Returns:
    --------
    networkx.Graph
        A cleaned, undirected graph where each node has an embedding as a `torch.Tensor`
        and all edge attributes have been removed.

    Notes:
    ------
    - If a node has an embedding stored as a dictionary with the key `"embedding"`, the value is extracted.
    - If an embedding is a list or `numpy.ndarray`, it is converted to a `torch.Tensor`.
    - Any node without an embedding is either removed (if `remove_incomplete=True`) or assigned a zero tensor.
    - All edge attributes are deleted to ensure compatibility with GraphSAGE training.
    """

    # Convert the graph to an undirected version
    G_undirected = nx.Graph(nx.to_undirected(G))

    # Iterate over nodes to clean embeddings
    for node in list(G_undirected.nodes()):  # Use list to avoid modifying during iteration
        node_attrs = G_undirected.nodes[node]
        embedding = node_attrs.get("embedding", None)

        if embedding is None:
            if remove_incomplete:
                print(f"[INFO] Removing node {node} due to missing 'embedding'.")
                G_undirected.remove_node(node)
                continue  # Skip further processing for this node
            else:
                print(f"[WARNING] Node {node} missing 'embedding'. Assigning default zero tensor.")
                embedding = np.zeros(embedding_dim, dtype=np.float32)

        # If embedding is stored as a dictionary, extract the value
        if isinstance(embedding, dict) and "embedding" in embedding:
            embedding = embedding["embedding"]

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
            print(f"[WARNING] Node {node} embedding has shape {embedding.shape}, expected ({embedding_dim},).")

        # Assign cleaned embedding back to the node
        node_attrs["embedding"] = embedding

        # Remove all attributes except 'embedding'
        keys_to_remove = [k for k in node_attrs.keys() if k != "embedding"]
        for k in keys_to_remove:
            del node_attrs[k]

    # Remove all edge attributes
    for u, v in G_undirected.edges():
        G_undirected[u][v].clear()

    return G_undirected
