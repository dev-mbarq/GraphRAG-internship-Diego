import networkx as nx
from chromadb import Client

def add_semantic_edges(G: nx.DiGraph, n: int) -> nx.DiGraph:
    """
    For each node of type 'article' in graph G, find the top-n most similar
    article nodes based on their 'embedding' attribute, and add edges
    connecting them with a 'semantic_similarity' relation.
    """
    # If G isn't directed, convert it to a DiGraph
    if not isinstance(G, nx.DiGraph):
        G = nx.DiGraph(G)

    # Initialize the ChromaDB client (in-memory by default)
    client = Client()
    # Create or get a collection named 'articles' to store embeddings
    collection = client.create_collection(name="Articles")

    # Gather all article-type nodes
    article_nodes = [
        node
        for node, data in G.nodes(data=True)
        if data.get("node_type") == "Article"
    ]

    print(f"Number of article nodes: {len(article_nodes)}")  # Debug: print number of article nodes


    # Map string IDs (for Chroma) back to original node keys
    id_map = {str(node): node for node in article_nodes}
    # Extract embeddings in the same order as article_nodes
    embeddings = [G.nodes[node]["embedding"] for node in article_nodes]

    print(embeddings[0])  # Debug: print first embedding
    print((len(embeddings)))  # Debug: print number of embeddings

    # Add embeddings to the ChromaDB collection
    collection.add(
        embeddings=embeddings,
        ids=list(id_map.keys()),
        #metadatas=[{} for _ in article_nodes],  # optional metadata placeholder
    )

    # For each article, query for the top-n similar embeddings
    for str_id in id_map.keys():
        # Retrieve the original embedding from the graph
        embedding = G.nodes[id_map[str_id]]["embedding"]
        # Query ChromaDB for n+1 results (including itself)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n + 1,
        )
        # Extract the list of similar IDs
        similar_ids = results["ids"][0]

        # Create edges for each similar article (excluding self)
        for sim_str in similar_ids:
            if sim_str == str_id:
                continue  # skip self-match
            # Map back to original node keys
            src = id_map[str_id]
            dst = id_map[sim_str]
            # Add an undirected edge with a semantic similarity label
            G.add_edge(src, dst, relation="semantic_similarity")

    return G