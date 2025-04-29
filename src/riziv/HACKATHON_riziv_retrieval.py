# Standard library imports
import sys
from typing import Dict
import pickle
import os

# Third-party imports
import pandas as pd

# Local imports
from src.riziv.HACKATHON_riziv_retrieval_utils import (
    build_sources_citation,
    create_chroma_collection_and_retrieve_top_k,
    extract_date_from_query,
    filter_relevant_evidence,
    generate_final_answer,
    get_document_context,
    get_query_embedding,
    get_valid_graph_at_date,
)


def process_legal_query(
    query: str,
    graph_path: str = "../data/document_graph_with_embeddings.pkl",
    articles_path: str = "../data/df_workArticlePlusLanguageFR.csv",
    verbose: bool = False,
) -> Dict:
    """
    Process a legal query and return the answer with relevant citations.

    Args:
        query (str): The user's query in French
        graph_path (str): Path to the pickled graph with embeddings
        articles_path (str): Path to the articles DataFrame
        verbose (bool): If True, prints progress information

    Returns:
        Dict: {
            'answer': str,           # The generated answer to the query
            'citations': str,        # Formatted citations of relevant sources
            'relevant_date': str,    # Date extracted from the query
            'evidence_count': int    # Number of pieces of evidence found
        }

    Raises:
        Exception: If any of the processing steps fail
    """

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Construct absolute paths
    graph_path = os.path.join(project_root, "data", graph_path)
    articles_path = os.path.join(project_root, "data", articles_path)

    try:
        # Load resources
        with open(graph_path, "rb") as f:
            G_emb = pickle.load(f)

        workArticlePlusLanguageFR = pd.read_csv(articles_path)

        if verbose:
            print("-------------------------")
            print("Extract_date_from_query")

        # Extract date
        relevant_date = extract_date_from_query(query)
        if verbose:
            print(f"Extracted date: {relevant_date}")
            print("-------------------------")

        if verbose:
            print("-------------------------")
            print("Get_valid_graph_at_date")

        # Get graph at date
        G_at_date = get_valid_graph_at_date(G_emb, relevant_date, verbose=verbose)

        if verbose:
            print("-------------------------")

        # Process query embedding
        query_embedding = get_query_embedding(query)

        if verbose:
            print("-------------------------")
            print("Create_chroma_collection_and_retrieve_top_k")

        # Query vector database
        chroma_results = create_chroma_collection_and_retrieve_top_k(
            G=G_at_date, query_embedding=query_embedding, k=10, verbose=verbose
        )

        chroma_results_texts = chroma_results["documents"][0]

        if verbose:
            print("-------------------------")
            print("Filter_relevant_evidence")

        # Filter evidence
        relevant_evidence = filter_relevant_evidence(
            query=query, evidence_texts=chroma_results_texts
        )

        if verbose:
            for i in relevant_evidence:
                print(i[1], "\n")
            print("-------------------------")

        if verbose:
            print("-------------------------")
            print("Get_document_context")

        # Get context
        relevant_chunk_ids = [
            chroma_results["ids"][0][idx] for idx, _ in relevant_evidence
        ]
        context = get_document_context(G=G_at_date, chunk_ids=relevant_chunk_ids)

        if verbose:
            for chunk_info in context:
                print(f"\nChunk: {chunk_info['chunk_id']}")
                if chunk_info["article_info"]:
                    print(f"Article: {chunk_info['article_info']['sort_tuple']}")
                if chunk_info["act_info"]:
                    print(f"Act: {chunk_info['act_info']['title_short']}")
            print("-------------------------")

        if verbose:
            print("-------------------------")
            print("Build_sources_citation")

        # Generate citations
        citations = build_sources_citation(context, workArticlePlusLanguageFR)

        if verbose:
            print(citations)
            print("-------------------------")

        if verbose:
            print("-------------------------")
            print("Generate_final_answer")

        # Generate final answer
        final_answer = generate_final_answer(
            query, relevant_evidence, context, workArticlePlusLanguageFR
        )

        if verbose:
            print(final_answer + "\n\n\n" + citations)
            print("-------------------------")

        return {
            "answer": final_answer,
            "citations": citations,
            "relevant_date": str(relevant_date),
            "evidence_count": len(relevant_evidence),
        }

    except Exception as e:
        raise Exception(f"Error processing query: {str(e)}")


# Use example
result = process_legal_query(
    query="Quand est-ce qu'un document justificatif doit être remis ?", verbose=True
)

print(result)
