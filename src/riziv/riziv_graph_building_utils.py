import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime


def create_base_document_graph(sequence_df, chunks_df, work_article_df, work_act_df):
    """
    Create a directed graph with four levels of hierarchy:
    1. Acts: Represent collections of articles
    2. Articles: Represent the persistent concept of an article across time
    3. Sequences: Represent specific versions of articles at different points in time
    4. Chunks: Represent text fragments of sequences
    """
    G = nx.DiGraph()

    # First, create act nodes
    print("Adding act nodes...")
    for _, row in work_act_df.iterrows():
        act_node_id = f"act_{row['Id']}"
        act_attrs = {
            "type_node": "act",
            "Id": row["Id"],
            "TypeDocument": row["TypeDocument"],
            "DateDocument": row["DateDocument"],
            "FirstEntryInForce": row["FirstEntryInForce"],
            "DateNoLongerInForce": row["DateNoLongerInForce"],
            "DatePublication": row["DatePublication"],
            "Title": row["Title"],
            "TitleShort": row["TitleShort"],
        }
        G.add_node(act_node_id, **act_attrs)

    # Create article nodes and their relationships with acts
    print("Adding article nodes and act relationships...")
    unique_articles = sequence_df[
        ["article", "article_number_std", "sort_tuple"]
    ].drop_duplicates("article")

    for _, row in unique_articles.iterrows():
        article_node_id = f"article_{row['article']}"
        # Get act information from workArticlePlusLanguageFR
        act_info = int(
            work_article_df.loc[
                work_article_df["Id"] == row["article"], "IsPartOf"
            ].iloc[0]
        )

        article_attrs = {
            "type_node": "article",
            "article": row["article"],
            "article_number_std": row["article_number_std"],
            "sort_tuple": row["sort_tuple"],
            "act": act_info,
        }
        G.add_node(article_node_id, **article_attrs)

        # Create bidirectional edges between act and article
        act_node_id = f"act_{act_info}"
        G.add_edge(act_node_id, article_node_id, relationship_type="contains_article")
        G.add_edge(article_node_id, act_node_id, relationship_type="belongs_to_act")

    # Add sequence nodes and their relationships with articles
    print("Adding sequence nodes and article relationships...")
    for _, row in sequence_df.iterrows():
        # Convert row to dictionary of attributes
        node_attrs = row.to_dict()
        # Add node type attribute
        node_attrs["type_node"] = "sequence"
        # Add node to graph with its attributes using sequence_id
        sequence_node_id = (
            f"sequence_{row['sequence_id']}"  # Changed to use sequence_id
        )
        G.add_node(sequence_node_id, **node_attrs)

        # Create bidirectional edges between article and sequence
        article_node_id = f"article_{row['article']}"
        G.add_edge(article_node_id, sequence_node_id, relationship_type="has_version")
        G.add_edge(sequence_node_id, article_node_id, relationship_type="version_of")

    # Add chunk nodes and their edges
    print("Adding chunk nodes and their relationships...")
    chunk_groups = chunks_df.groupby("sequence_id")

    for sequence_id, group in chunk_groups:
        # Sort chunks by their index
        sorted_chunks = group.sort_values("chunk_index")

        # Process each chunk in the group
        previous_chunk_id = None
        for _, row in sorted_chunks.iterrows():
            # Convert row to dictionary of attributes
            node_attrs = row.to_dict()
            node_attrs["type_node"] = "text_chunk"
            # Create unique identifier for chunk node using sequence_id and chunk_index
            chunk_node_id = f"chunk_{sequence_id}_{row['chunk_index']}"
            G.add_node(chunk_node_id, **node_attrs)

            # Add bidirectional edges between chunk and sequence
            sequence_node_id = f"sequence_{sequence_id}"
            G.add_edge(
                sequence_node_id, chunk_node_id, relationship_type="contains_chunk"
            )
            G.add_edge(
                chunk_node_id,
                sequence_node_id,
                relationship_type="contained_in_sequence",
            )

            # Add bidirectional edges between consecutive chunks
            if previous_chunk_id is not None:
                G.add_edge(
                    previous_chunk_id, chunk_node_id, relationship_type="followed_by"
                )
                G.add_edge(
                    chunk_node_id, previous_chunk_id, relationship_type="preceded_by"
                )

            # Update previous chunk for next iteration
            previous_chunk_id = chunk_node_id

    # Print statistics
    print("\nGraph Statistics:")
    print(f"Total number of nodes: {G.number_of_nodes()}")
    print(f"Total number of edges: {G.number_of_edges()}")
    print(
        f"Number of article nodes: {len([n for n in G.nodes if G.nodes[n]['type_node'] == 'article'])}"
    )
    print(
        f"Number of sequence nodes: {len([n for n in G.nodes if G.nodes[n]['type_node'] == 'sequence'])}"
    )
    print(
        f"Number of chunk nodes: {len([n for n in G.nodes if G.nodes[n]['type_node'] == 'text_chunk'])}"
    )

    # Count edges by type
    edge_types = [
        "contains_article",
        "belongs_to_act",
        "has_version",
        "version_of",
        "contains_chunk",
        "contained_in_sequence",
        "followed_by",
        "preceded_by",
    ]

    print("\nEdge Statistics:")
    for edge_type in edge_types:
        count = len(
            [e for e in G.edges(data=True) if e[2]["relationship_type"] == edge_type]
        )
        print(f"{edge_type} edges: {count}")

    return G
