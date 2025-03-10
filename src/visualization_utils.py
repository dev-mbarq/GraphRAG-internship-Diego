import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE
from collections import defaultdict

def multihop_interactive_graph(graph, embedding_attribute, random_seed, perplexity_val, title, save=None):
    """
    Generates an interactive visualization of a graph, where the node's positions is determined 
    by 2D coordinates derived from an specified embedding attribute processed by t-SNE.

    Parameters:
    -----------
    graph : networkx.Graph
        The graph object from which to extract node embeddings and associated attributes.
    embedding_attribute : str
        The key for the node attribute that contains the embedding vector.
    random_seed : int
        Seed for the t-SNE algorithm's random_state parameter, ensuring reproducibility.
    perplexity_val : int
        The perplexity parameter for the t-SNE algorithm, balancing local and global data structure.
    title : str
        The title to be displayed on the plot.
    save : str or None, optional
        The file path where the interactive HTML visualization should be saved.
        If None, the figure is displayed interactively without saving (default is None).

    Returns:
    -----------
    None
        Either displays the interactive Plotly figure or saves it as an HTML file if a save path is provided.

    Notes:
    ------
    - The function extracts node embeddings and supplementary attributes (such as type, category, author, and source)
      from the provided graph.
    - It applies t-SNE to reduce the high-dimensional embeddings to 2D coordinates suitable for visualization.
    - Edges are grouped by their "relation" attribute and drawn as line segments connecting corresponding nodes.
    - Node coloring is determined by node type: "article" nodes are assigned varying shades of blue based on their category,
      while other node types are assigned fixed colors.
    - The interactive visualization is created using Plotly, allowing for toggling of node types and edge relations.
    """

    # -------------------------------------------
    # 1) Extract embeddings and node attributes
    # -------------------------------------------
    embedding_list = []
    node_ids = []
    node_types = []
    node_categories = []
    node_authors = []
    node_sources = []

    # Loop over all nodes in the graph and extract data if the embedding attribute exists.
    for node, data in graph.nodes(data=True):
        if embedding_attribute in data:
            embedding_list.append(data[embedding_attribute])
            node_ids.append(node)
            node_types.append(data.get("type", "unknown"))
            node_categories.append(data.get("category", "unknown"))
            node_authors.append(data.get("author", "unknown"))
            node_sources.append(data.get("source", "unknown"))

    print(f"Extracted embeddings from {len(embedding_list)} nodes.")

    # -------------------------------------------
    # 2) Convert embeddings to a numpy array
    # -------------------------------------------
    embeddings_array = np.array(embedding_list)
    print(f"Embeddings array shape: {embeddings_array.shape}")

    # -------------------------------------------
    # 3) Apply t-SNE dimensionality reduction
    # -------------------------------------------
    tsne = TSNE(n_components=2, random_state=random_seed, perplexity=perplexity_val)
    embeddings_2d = tsne.fit_transform(embeddings_array)
    print("t-SNE transformation complete.")

    # -------------------------------------------
    # 4) Create a DataFrame with coordinates and node attributes
    # -------------------------------------------
    df_tsne = pd.DataFrame(embeddings_2d, columns=['x', 'y'])
    df_tsne['node_id'] = node_ids
    df_tsne['node_type'] = node_types
    df_tsne['category'] = node_categories
    df_tsne['author'] = node_authors
    df_tsne['source'] = node_sources

    # Filter out nodes with type "chunk" if they are not intended for visualization.
    df_tsne = df_tsne[df_tsne["node_type"] != "chunk"].copy()

    # -------------------------------------------
    # 5) Generate edge traces grouped by "relation"
    # -------------------------------------------
    # Explanation:
    # We loop over each edge in the graph, and if both endpoints exist in the filtered DataFrame,
    # we retrieve their 2D coordinates. We then group these coordinates based on the "relation"
    # attribute of the edge. The None value is used to separate individual edge segments in the trace.
    edges_by_relation = defaultdict(lambda: {"x": [], "y": []})
    valid_nodes = set(df_tsne["node_id"])  # Nodes that will be visualized

    for u, v, data in graph.edges(data=True):
        if u in valid_nodes and v in valid_nodes:
            relation = data.get("relation", "unknown")
            x0, y0 = df_tsne.loc[df_tsne['node_id'] == u, ['x', 'y']].values[0]
            x1, y1 = df_tsne.loc[df_tsne['node_id'] == v, ['x', 'y']].values[0]
            edges_by_relation[relation]["x"].extend([x0, x1, None])
            edges_by_relation[relation]["y"].extend([y0, y1, None])

    # -------------------------------------------
    # 6) Prepare node colors:
    #     - For "article" nodes, use varying shades of blue based on the "category".
    #     - For other types, use a fixed color defined in the mapping.
    # -------------------------------------------
    color_map_other_types = {
        "author": "red",
        "source": "green",
        "category": "orange",
        "unknown": "gray"
        # "article" nodes are handled separately
    }

    # Create a discrete mapping from article category to a specific shade of blue.
    article_categories = df_tsne.loc[df_tsne["node_type"] == "article", "category"].unique()
    article_categories = sorted(article_categories)
    cat_color_scale = px.colors.sequential.Blues  # List of hex colors from the "Blues" scale.
    n_scale = len(cat_color_scale) - 1

    category_color_map = {}
    for i, cat in enumerate(article_categories):
        # Calculate an index to evenly distribute categories over the color scale.
        idx = int((i / max(1, len(article_categories) - 1)) * n_scale)
        category_color_map[cat] = cat_color_scale[idx]

    # -------------------------------------------
    # 7) Create the figure and add edge traces for each "relation"
    # -------------------------------------------
    # Explanation:
    # We initialize a Plotly figure and add a separate trace for each type of edge relation.
    # This allows users to differentiate and toggle the visibility of various edge types via the legend.
    fig = go.Figure()

    for rel, coords in edges_by_relation.items():
        fig.add_trace(go.Scatter(
            x=coords["x"],
            y=coords["y"],
            mode='lines',
            name=f"Edge: {rel}",  # This will appear in the legend.
            line=dict(color='gray', width=1),
            hoverinfo='none'
        ))

    # -------------------------------------------
    # 8) Create node traces for each "node_type" to allow toggling by type
    # -------------------------------------------
    # Explanation:
    # We iterate over each unique node type and filter the DataFrame accordingly.
    # For each type, we generate hover text with all relevant node attributes.
    # "Article" nodes receive colors based on their category, while others use fixed colors.
    # This segmentation facilitates interactive control over the display of different node groups.
    unique_types = df_tsne['node_type'].unique()

    for ntype in unique_types:
        sub_df = df_tsne[df_tsne['node_type'] == ntype].copy()

        hover_text = sub_df.apply(
            lambda row: (
                f"node_id: {row['node_id']}<br>"
                f"type: {row['node_type']}<br>"
                f"category: {row['category']}<br>"
                f"author: {row['author']}<br>"
                f"source: {row['source']}"
            ),
            axis=1
        )

        if ntype == "article":
            # For each "article" node, determine the color based on its category.
            node_colors = [category_color_map.get(row['category'], "#CCCCCC") for _, row in sub_df.iterrows()]
        else:
            # For other node types, use a pre-defined fixed color.
            node_colors = [color_map_other_types.get(ntype, 'gray')] * len(sub_df)

        node_trace = go.Scatter(
            x=sub_df["x"],
            y=sub_df["y"],
            mode='markers',
            name=ntype,  # Legend entry for node type.
            marker=dict(size=8, opacity=0.8, color=node_colors),
            text=hover_text,
            hoverinfo='text'
        )
        fig.add_trace(node_trace)

    # -------------------------------------------
    # 9) Update layout and display or save the figure
    # -------------------------------------------
    # Explanation:
    # The layout is adjusted to set the title, legend visibility, hover behavior, margins, and figure dimensions.
    # Finally, if a save path is provided, the figure is saved as an HTML file.
    # Otherwise, the figure is displayed interactively using fig.show().
    fig.update_layout(
        title=f"{title}",
        showlegend=True,
        hovermode="closest",
        margin=dict(b=0, l=0, r=0, t=40),
        width=1000,
        height=800,
        plot_bgcolor='#aed1b7'
    )

    if save is not None:
        fig.write_html(save)
        print(f"Figure saved to {save}")
    else:
        fig.show()