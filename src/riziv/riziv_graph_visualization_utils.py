import networkx as nx
import matplotlib.pyplot as plt


def visualize_node_1hop(G, node_id, figsize=(12, 8)):
    """
    Visualize a node and its immediate neighbors in the graph

    Args:
        G: NetworkX graph
        node_id: ID of the central node to visualize
        figsize: Size of the figure (width, height)
    """
    # Create a subgraph with the node and its neighbors
    neighbors = list(G.predecessors(node_id)) + list(G.successors(node_id))
    subgraph_nodes = [node_id] + neighbors
    subgraph = G.subgraph(subgraph_nodes)

    # Create figure
    plt.figure(figsize=figsize)

    # Create layout for nodes
    pos = nx.spring_layout(subgraph, k=1, iterations=50)

    # Draw nodes
    # Central node in red
    nx.draw_networkx_nodes(
        subgraph, pos, nodelist=[node_id], node_color="red", node_size=2000
    )

    # Act nodes in gold
    act_nodes = [n for n in subgraph.nodes() if subgraph.nodes[n]["type_node"] == "act"]
    act_nodes = [n for n in act_nodes if n != node_id]
    nx.draw_networkx_nodes(
        subgraph, pos, nodelist=act_nodes, node_color="gold", node_size=2000
    )

    # Article nodes in purple
    article_nodes = [
        n for n in subgraph.nodes() if subgraph.nodes[n]["type_node"] == "article"
    ]
    article_nodes = [n for n in article_nodes if n != node_id]
    nx.draw_networkx_nodes(
        subgraph, pos, nodelist=article_nodes, node_color="purple", node_size=1500
    )

    # Sequence nodes in blue
    sequence_nodes = [
        n for n in subgraph.nodes() if subgraph.nodes[n]["type_node"] == "sequence"
    ]
    sequence_nodes = [n for n in sequence_nodes if n != node_id]
    nx.draw_networkx_nodes(
        subgraph, pos, nodelist=sequence_nodes, node_color="lightblue", node_size=1500
    )

    # Chunk nodes in green
    chunk_nodes = [
        n for n in subgraph.nodes() if subgraph.nodes[n]["type_node"] == "text_chunk"
    ]
    chunk_nodes = [n for n in chunk_nodes if n != node_id]
    nx.draw_networkx_nodes(
        subgraph, pos, nodelist=chunk_nodes, node_color="lightgreen", node_size=1500
    )

    # Draw edges with different colors and labels
    edge_colors = {
        "contains_article": "darkgoldenrod",
        "belongs_to_act": "gold",
        "has_version": "purple",
        "version_of": "lavender",
        "contains_chunk": "blue",
        "contained_in_sequence": "lightblue",
        "followed_by": "green",
        "preceded_by": "lightgreen",
    }

    # Create dictionaries to store edge labels for each direction
    edge_labels_forward = {}
    edge_labels_backward = {}

    # Draw edges for each relationship type
    for rel_type, color in edge_colors.items():
        edges = [
            (u, v)
            for (u, v, d) in subgraph.edges(data=True)
            if d["relationship_type"] == rel_type
        ]

        # Draw edges
        nx.draw_networkx_edges(
            subgraph, pos, edgelist=edges, edge_color=color, arrows=True, arrowsize=20
        )

        # Add edge labels to appropriate dictionary based on relationship type
        for u, v in edges:
            # Use shorter labels for better visibility
            short_labels = {
                "contains_article": "contains",
                "belongs_to_act": "belongs to",
                "has_version": "has ver.",
                "version_of": "ver. of",
                "contains_chunk": "contains",
                "contained_in_sequence": "part of",
                "followed_by": "next",
                "preceded_by": "prev",
            }
            label = short_labels.get(rel_type, rel_type)

            # Determine if this is a forward or backward relationship
            if rel_type in [
                "belongs_to_act",
                "version_of",
                "contained_in_sequence",
                "preceded_by",
            ]:
                edge_labels_backward[(u, v)] = label
            else:
                edge_labels_forward[(u, v)] = label

    # Draw edge labels with offset for better visibility
    nx.draw_networkx_edge_labels(
        subgraph,
        pos,
        edge_labels=edge_labels_forward,
        font_size=8,  # Increased font size
        font_color="black",
        font_family="sans-serif",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        label_pos=0.2,
    )  # Position closer to target node

    nx.draw_networkx_edge_labels(
        subgraph,
        pos,
        edge_labels=edge_labels_backward,
        font_size=8,  # Increased font size
        font_color="black",
        font_family="sans-serif",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        label_pos=0.2,
    )  # Position closer to source node

    # Add labels to nodes
    labels = {node: node for node in subgraph.nodes()}

    nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color="darkgoldenrod", label="Contains Article"),
        plt.Line2D([0], [0], color="gold", label="Belongs to Act"),
        plt.Line2D([0], [0], color="purple", label="Has Version"),
        plt.Line2D([0], [0], color="lavender", label="Version Of"),
        plt.Line2D([0], [0], color="blue", label="Contains Chunk"),
        plt.Line2D([0], [0], color="lightblue", label="Contained in Sequence"),
        plt.Line2D([0], [0], color="green", label="Followed by"),
        plt.Line2D([0], [0], color="lightgreen", label="Preceded by"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Central Node",
            markerfacecolor="red",
            markersize=10,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Act Node",
            markerfacecolor="gold",
            markersize=10,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Article Node",
            markerfacecolor="purple",
            markersize=10,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Sequence Node",
            markerfacecolor="lightblue",
            markersize=10,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Chunk Node",
            markerfacecolor="lightgreen",
            markersize=10,
        ),
    ]
    plt.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1))

    # Add title
    plt.title(f"Neighborhood of node {node_id}")

    # Adjust layout to prevent legend overlap
    plt.tight_layout()

    # Show plot
    plt.show()


def visualize_node_2hop(
    G, node_id, figsize=(15, 10)
):  # Aumentado el tamaño por defecto para acomodar más nodos
    """
    Visualize a node, its immediate neighbors, and neighbors of neighbors (2-hop neighborhood)

    Args:
        G: NetworkX graph
        node_id: ID of the central node to visualize
        figsize: Size of the figure (width, height)
    """
    # Create a subgraph with the node, its neighbors, and neighbors of neighbors
    first_neighbors = set(list(G.predecessors(node_id)) + list(G.successors(node_id)))
    second_neighbors = set()
    for neighbor in first_neighbors:
        second_neighbors.update(G.predecessors(neighbor))
        second_neighbors.update(G.successors(neighbor))

    # Remove first neighbors and central node from second_neighbors to avoid duplicates
    second_neighbors = second_neighbors - first_neighbors - {node_id}

    # Create subgraph with all nodes
    subgraph_nodes = [node_id] + list(first_neighbors) + list(second_neighbors)
    subgraph = G.subgraph(subgraph_nodes)

    # Create figure
    plt.figure(figsize=figsize)

    # Create layout for nodes with more space between them
    pos = nx.spring_layout(
        subgraph, k=2, iterations=50
    )  # Aumentado k para más separación

    # Draw nodes
    # Central node in red
    nx.draw_networkx_nodes(
        subgraph, pos, nodelist=[node_id], node_color="red", node_size=3000
    )  # Aumentado el tamaño

    # Act nodes in gold
    act_nodes = [n for n in subgraph.nodes() if subgraph.nodes[n]["type_node"] == "act"]
    act_nodes = [n for n in act_nodes if n != node_id]
    nx.draw_networkx_nodes(
        subgraph, pos, nodelist=act_nodes, node_color="gold", node_size=2500
    )

    # Article nodes in purple
    article_nodes = [
        n for n in subgraph.nodes() if subgraph.nodes[n]["type_node"] == "article"
    ]
    article_nodes = [n for n in article_nodes if n != node_id]
    nx.draw_networkx_nodes(
        subgraph, pos, nodelist=article_nodes, node_color="purple", node_size=2000
    )

    # Sequence nodes in blue
    sequence_nodes = [
        n for n in subgraph.nodes() if subgraph.nodes[n]["type_node"] == "sequence"
    ]
    sequence_nodes = [n for n in sequence_nodes if n != node_id]
    nx.draw_networkx_nodes(
        subgraph, pos, nodelist=sequence_nodes, node_color="lightblue", node_size=2000
    )

    # Chunk nodes in green
    chunk_nodes = [
        n for n in subgraph.nodes() if subgraph.nodes[n]["type_node"] == "text_chunk"
    ]
    chunk_nodes = [n for n in chunk_nodes if n != node_id]
    nx.draw_networkx_nodes(
        subgraph, pos, nodelist=chunk_nodes, node_color="lightgreen", node_size=2000
    )

    # Draw edges with different colors and labels
    edge_colors = {
        "contains_article": "darkgoldenrod",
        "belongs_to_act": "gold",
        "has_version": "purple",
        "version_of": "lavender",
        "contains_chunk": "blue",
        "contained_in_sequence": "lightblue",
        "followed_by": "green",
        "preceded_by": "lightgreen",
    }

    # Create dictionaries to store edge labels for each direction
    edge_labels_forward = {}
    edge_labels_backward = {}

    # Draw edges for each relationship type
    for rel_type, color in edge_colors.items():
        edges = [
            (u, v)
            for (u, v, d) in subgraph.edges(data=True)
            if d["relationship_type"] == rel_type
        ]

        # Draw edges
        nx.draw_networkx_edges(
            subgraph, pos, edgelist=edges, edge_color=color, arrows=True, arrowsize=20
        )

        # Add edge labels to appropriate dictionary based on relationship type
        for u, v in edges:
            # Use shorter labels for better visibility
            short_labels = {
                "contains_article": "contains",
                "belongs_to_act": "belongs to",
                "has_version": "has ver.",
                "version_of": "ver. of",
                "contains_chunk": "contains",
                "contained_in_sequence": "part of",
                "followed_by": "next",
                "preceded_by": "prev",
            }
            label = short_labels.get(rel_type, rel_type)

            # Determine if this is a forward or backward relationship
            if rel_type in [
                "belongs_to_act",
                "version_of",
                "contained_in_sequence",
                "preceded_by",
            ]:
                edge_labels_backward[(u, v)] = label
            else:
                edge_labels_forward[(u, v)] = label

    # Draw edge labels with offset for better visibility
    nx.draw_networkx_edge_labels(
        subgraph,
        pos,
        edge_labels=edge_labels_forward,
        font_size=8,
        font_color="black",
        font_family="sans-serif",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        label_pos=0.2,
    )

    nx.draw_networkx_edge_labels(
        subgraph,
        pos,
        edge_labels=edge_labels_backward,
        font_size=8,
        font_color="black",
        font_family="sans-serif",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        label_pos=0.2,
    )

    # Add labels to nodes
    labels = {node: node for node in subgraph.nodes()}

    nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color="darkgoldenrod", label="Contains Article"),
        plt.Line2D([0], [0], color="gold", label="Belongs to Act"),
        plt.Line2D([0], [0], color="purple", label="Has Version"),
        plt.Line2D([0], [0], color="lavender", label="Version Of"),
        plt.Line2D([0], [0], color="blue", label="Contains Chunk"),
        plt.Line2D([0], [0], color="lightblue", label="Contained in Sequence"),
        plt.Line2D([0], [0], color="green", label="Followed by"),
        plt.Line2D([0], [0], color="lightgreen", label="Preceded by"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Central Node",
            markerfacecolor="red",
            markersize=10,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Act Node",
            markerfacecolor="gold",
            markersize=10,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Article Node",
            markerfacecolor="purple",
            markersize=10,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Sequence Node",
            markerfacecolor="lightblue",
            markersize=10,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Chunk Node",
            markerfacecolor="lightgreen",
            markersize=10,
        ),
    ]
    plt.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1))

    # Add title
    plt.title(f"2-hop neighborhood of node {node_id}")

    # Adjust layout to prevent legend overlap
    plt.tight_layout()

    # Show plot
    plt.show()
