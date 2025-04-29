# Standard library imports
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Third party imports
import chromadb
import networkx as nx
import numpy as np
import pandas as pd
from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel
from tqdm import tqdm


# Load environment variables
env_path = Path(__file__).resolve().parent.parent / "data" / ".env"
load_dotenv(env_path)


class ExtractDate(BaseModel):
    relevant_date: str


class AssessRelevance(BaseModel):
    # Fragment relevance
    is_fragment_relevant: bool


def get_query_embedding(query: str) -> List[float]:
    """Get embeddings for a query using Azure OpenAI's text-embedding-3-small model"""
    endpoint = os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_EMBEDDINGS_KEY")

    if not endpoint or not api_key:
        raise ValueError(
            "Missing required environment variables for Azure OpenAI embeddings"
        )

    try:
        emb_client = EmbeddingsClient(
            endpoint=endpoint, credential=AzureKeyCredential(api_key)
        )

        try:
            response = emb_client.embed(input=query, model="text-embedding-3-small")
            return response.data[0]["embedding"]
        finally:
            emb_client.close()  # Close client

    except Exception as e:
        raise Exception(f"Error getting embeddings: {str(e)}")


def extract_date_from_query(query: str) -> str:
    """Extract date from query using Azure OpenAI"""
    api_version = os.getenv("AZURE_OPENAI_RIZIV_API_VERSION")
    endpoint = os.getenv("AZURE_OPENAI_RIZIV_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_RIZIV_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_RIZIV_DEPLOYMENT")

    if not all([api_version, endpoint, api_key, deployment]):
        raise ValueError("Missing required environment variables for Azure OpenAI")

    try:
        date_extraction_client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=api_key,
        )

        try:
            today = datetime.now().strftime("%Y-%m-%d")
            completion = date_extraction_client.beta.chat.completions.parse(
                model=deployment,
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are an expert at extracting dates from user queries. 
                        Your task is to identify and extract any date mentioned or implied in a user query about Belgian legal texts.
                        If a specific date is mentioned, return it in YYYY-MM-DD format.
                        If only a year is mentioned, use January 1st of that year (YYYY-01-01).
                        If no date is mentioned or implied, return "{today}" as default.
                        Always return a valid date string in YYYY-MM-DD format.""",
                    },
                    {
                        "role": "user",
                        "content": f"""Extract the relevant date from the following query. 
                        If multiple dates are mentioned, choose the most relevant one for legal document filtering.
                        If no date is mentioned, return the default date ({today}).

                        Query: {query}""",
                    },
                ],
                response_format=ExtractDate,
            )
            return completion.choices[0].message.parsed.relevant_date
        finally:
            date_extraction_client.close()

    except Exception as e:
        raise Exception(f"Error extracting date: {str(e)}")


def get_valid_graph_at_date(G, target_date, verbose: bool = False):
    """
    Creates a subgraph containing only nodes and their connections that are valid at a specific date

    Args:
        G: Original NetworkX graph
        target_date: target date to check (string or datetime)
        verbose: If True, prints detailed information about the filtering process (default: True)
    Returns:
        G_filtered: Subgraph with only valid nodes at that date
    """
    # Convert date if needed
    target_date = pd.to_datetime(target_date)

    # Create a copy of the graph
    G_filtered = G.copy()

    # Get nodes to remove (invalid at target_date)
    nodes_to_remove = []

    # Iterate through nodes
    for node in G_filtered.nodes():
        node_data = G_filtered.nodes[node]

        # Check only text_chunk nodes that have date attributes
        if node_data.get("type_node") == "text_chunk":
            try:
                date_start = pd.to_datetime(node_data.get("date_start"))
                date_end = (
                    pd.to_datetime(node_data.get("date_end"))
                    if pd.notna(node_data.get("date_end"))
                    else None
                )

                # Check if node is invalid at target_date based only on dates
                if date_start > target_date or (
                    date_end is not None and date_end <= target_date
                ):
                    nodes_to_remove.append(node)

            except Exception as e:
                if verbose:
                    print(f"Error processing dates for node {node}: {str(e)}")
                continue

    # Remove invalid nodes
    G_filtered.remove_nodes_from(nodes_to_remove)

    if verbose:
        # Print summary
        print(f"\nGraph pruning summary for date {target_date.date()}:")
        print(f"Removed {len(nodes_to_remove)} invalid text chunks")
        print(
            f"Original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )
        print(
            f"Filtered graph: {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges"
        )

        # Print node type distribution
        node_types = {}
        for node in G_filtered.nodes():
            node_type = G_filtered.nodes[node].get("type_node")
            if node_type:
                node_types[node_type] = node_types.get(node_type, 0) + 1

        print("\nNode distribution in filtered graph:")
        for node_type, count in sorted(node_types.items()):
            print(f"- {node_type}: {count}")

    return G_filtered


def create_chroma_collection_and_retrieve_top_k(
    G: nx.Graph,
    query_embedding: List[float],
    collection_name: str = "docleg_rag",
    k: int = 10,
    node_types: List[str] = ["text_chunk", "concept"],
    verbose: bool = False,
) -> Dict:
    """
    Creates a ChromaDB collection from graph nodes and retrieves top-k similar documents

    Args:
        G (nx.Graph): Input graph with node embeddings
        query_embedding (List[float]): Query embedding vector
        collection_name (str): Name for ChromaDB collection
        k (int): Number of results to retrieve
        node_types (List[str]): Types of nodes to include in collection
        verbose (bool): If True, prints progress information (default: False)

    Returns:
        Dict: Query results containing ids, documents, and distances
    """
    # Instantiate client
    chroma_client = chromadb.Client()

    # Delete collection if exists
    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception as e:
        if verbose:
            print(f"Collection didn't exist or couldn't be deleted: {e}")

    # Create a collection
    collection = chroma_client.create_collection(name=collection_name)

    # Prepare nodes for addition
    nodes_to_add = []
    ids_to_add = []
    embeddings_to_add = []

    for node_id in G.nodes:
        node_data = G.nodes[node_id]
        if node_data.get("type_node") in node_types:
            # For concept nodes, use 'name' instead of 'text'
            if node_data.get("type_node") == "concept":
                doc_text = node_data.get("name", "")
            else:
                doc_text = node_data.get("text", "")

            doc_embedding = node_data.get("embedding", None)

            if doc_embedding is not None and doc_text:
                nodes_to_add.append(doc_text)
                ids_to_add.append(str(node_id))
                embeddings_to_add.append(
                    doc_embedding.tolist()
                    if isinstance(doc_embedding, np.ndarray)
                    else doc_embedding
                )

    if verbose:
        print(f"Adding {len(nodes_to_add)} documents to collection...")

    # Add documents one by one
    for i in tqdm(
        range(len(nodes_to_add)), desc="Adding documents", disable=not verbose
    ):
        try:
            collection.add(
                ids=[ids_to_add[i]],
                documents=[nodes_to_add[i]],
                embeddings=[embeddings_to_add[i]],
            )
        except Exception as e:
            if verbose:
                print(f"Error adding document {ids_to_add[i]}: {str(e)}")
            continue

    if verbose:
        print(f"Successfully added documents to collection")

    # Query the collection
    return collection.query(query_embeddings=[query_embedding], n_results=k)


def filter_relevant_evidence(
    query: str, evidence_texts: List[str]
) -> List[Tuple[int, str]]:
    """Filter evidence texts based on their relevance"""
    api_version = os.getenv("AZURE_OPENAI_RIZIV_API_VERSION")
    endpoint = os.getenv("AZURE_OPENAI_RIZIV_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_RIZIV_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_RIZIV_DEPLOYMENT")

    if not all([api_version, endpoint, api_key, deployment]):
        raise ValueError("Missing required environment variables for Azure OpenAI")

    try:
        llm_filter_client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=api_key,
        )

        try:
            relevant_evidence = []
            for i, evidence in enumerate(evidence_texts):
                completion = llm_filter_client.beta.chat.completions.parse(
                    model=deployment,
                    messages=[
                        {
                            "role": "system",
                            "content": """You are an expert legal evaluator specialized in Belgian legal texts in French of the RIZIV/INAMI. 
                         Your task is to assess the relevance of a given legal text fragment (evidence) in relation to a specific legal query provided by the user. 
                         For each evidence fragment, decide if it is relevant to answering the query or not. Your response must be strictly limited to a boolean 
                         result in JSON format, with a single field "relevant" that is true if the evidence is pertinent, or false otherwise. 
                         Do not include any additional commentary or information.""",
                        },
                        {
                            "role": "user",
                            "content": f"""Evaluate the relevance of the following legal text fragment for answering the legal query below. 
                         Provide your answer in the JSON format exactly as specified.

                        Query:
                        {query}

                        Legal text fragment:
                        {evidence}""",
                        },
                    ],
                    response_format=AssessRelevance,
                )
                if completion.choices[0].message.parsed.is_fragment_relevant:
                    relevant_evidence.append((i, evidence))
            return relevant_evidence
        finally:
            llm_filter_client.close()

    except Exception as e:
        raise Exception(f"Error filtering evidence: {str(e)}")


def get_document_context(G: nx.Graph, chunk_ids: List[str]) -> List[Dict]:
    """
    Get article and act context for given chunk IDs through graph traversal

    Args:
        G (nx.Graph): The graph containing document hierarchy
        chunk_ids (List[str]): List of chunk IDs to process

    Returns:
        List[Dict]: List of dictionaries containing context information for each chunk

    Example structure of returned dictionary:
    {
        'chunk_id': str,
        'article_info': {
            'article_id': int,
            'article_number': str,
            'sort_tuple': tuple
        },
        'act_info': {
            'act_id': int,
            'title': str,
            'title_short': str,
            'type': str,
            'date_document': str
        }
    }
    """
    context_info = []

    for chunk_id in chunk_ids:
        try:
            chunk_info = {"chunk_id": chunk_id, "article_info": None, "act_info": None}

            # Get chunk node
            if not G.has_node(chunk_id):
                print(f"Warning: Chunk {chunk_id} not found in graph")
                continue

            chunk_data = G.nodes[chunk_id]

            # Get sequence info directly from chunk attributes
            sequence_id = chunk_data.get("sequence_id")
            if not sequence_id:
                print(f"Warning: No sequence_id found for chunk {chunk_id}")
                continue

            # Find sequence node
            sequence_node = f"sequence_{sequence_id}"
            if not G.has_node(sequence_node):
                print(f"Warning: Sequence node {sequence_node} not found")
                continue

            # Get article info from sequence attributes
            sequence_data = G.nodes[sequence_node]
            article_id = sequence_data.get("article")
            if not article_id:
                print(f"Warning: No article_id found in sequence {sequence_node}")
                continue

            # Find article node
            article_node = f"article_{article_id}"
            if not G.has_node(article_node):
                print(f"Warning: Article node {article_node} not found")
                continue

            # Store article attributes
            article_data = G.nodes[article_node]
            chunk_info["article_info"] = {
                "article_id": article_id,
                "article_number": article_data.get("article_number_std"),
                "sort_tuple": article_data.get("sort_tuple"),
            }

            # Get act info from article attributes
            act_id = article_data.get("act")
            if not act_id:
                print(f"Warning: No act_id found in article {article_node}")
                continue

            # Find act node
            act_node = f"act_{act_id}"
            if not G.has_node(act_node):
                print(f"Warning: Act node {act_node} not found")
                continue

            # Store act attributes
            act_data = G.nodes[act_node]
            chunk_info["act_info"] = {
                "act_id": act_id,
                "title": act_data.get("Title"),
                "title_short": act_data.get("TitleShort"),
                "type": act_data.get("TypeDocument"),
                "date_document": act_data.get("DateDocument"),
            }

            context_info.append(chunk_info)

        except Exception as e:
            print(f"Error processing chunk {chunk_id}: {str(e)}")
            continue

    # Print summary
    print(f"\nProcessed {len(chunk_ids)} chunks:")
    print(f"- Found complete context for {len(context_info)} chunks")
    print(
        f"- Failed to find complete context for {len(chunk_ids) - len(context_info)} chunks"
    )

    return context_info


def get_article_url(act_id: int, article_id: int) -> str:
    """Get the URL for a specific article"""
    return f"https://webappsa.riziv-inami.fgov.be/docleg/Article?lang=fr&enforce=False&dtrf=04%2F01%2F2025%2000%3A00%3A00&actId={act_id}&articleId={article_id}&all=False"


def format_article_number(article_code: str) -> str:
    """
    Convert internal article number (XXXX.XX.XXX) to workArticlePlusLanguageFR format

    Args:
        article_code (str): Article number in format XXXX.XX.XXX

    Returns:
        str: Formatted article number
    """
    parts = article_code.split(".")
    if len(parts) != 3:
        return article_code

    main, middle, end = parts

    if end != "000":
        if middle != "00":
            return f"{main}.{middle}.{end}"
        return f"{main}.{end}"
    if middle != "00":
        return f"{main}.{middle}"
    return main


def get_article_title(article_number: str, df_articles: pd.DataFrame) -> str:
    """
    Get article title from workArticlePlusLanguageFR using formatted article number

    Args:
        article_number (str): Article number to look up
        df_articles (pd.DataFrame): DataFrame containing article information

    Returns:
        str: Article title or "(title not found)" if not found
    """
    formatted_number = format_article_number(article_number)
    matching_row = df_articles[df_articles["Number"] == formatted_number]

    if len(matching_row) == 0:
        return "(title not found)"

    return matching_row.iloc[0]["Title"]


def build_sources_citation(context: List[Dict], df_articles: pd.DataFrame) -> str:
    """
    Build a formatted string citing all sources from the context, including article titles and URLs

    Args:
        context (List[Dict]): List of context dictionaries containing act and article info
        df_articles (pd.DataFrame): DataFrame containing article information

    Returns:
        str: Formatted citation string with URLs
    """
    if not context:
        return "No sources available."

    sources = ["Références:"]
    act_articles = {}  # Dictionary to group articles by act
    act_info_map = {}  # Dictionary to keep complete act information

    # First group articles by act
    for item in context:
        act_info = item["act_info"]
        article_info = item["article_info"]

        # Create unique act key
        act_key = (act_info["act_id"], act_info["title_short"])

        # Store complete act information
        act_info_map[act_key] = act_info

        # Initialize articles list if new act
        if act_key not in act_articles:
            act_articles[act_key] = []

        # Get article title
        article_title = get_article_title(article_info["article_number"], df_articles)

        # Add article and its URL to corresponding act's list
        act_articles[act_key].append(
            (
                article_info["article_number"],
                article_title,
                article_info["article_id"],  # Adding article_id for URL generation
            )
        )

    # Then build the citation
    for act_key, articles in act_articles.items():
        act_info = act_info_map[act_key]
        # Add act information
        sources.append(
            f"- {act_info['type']} {act_info['date_document']}: {act_info['title_short']}"
        )

        # Add all articles corresponding to this act with their URLs
        for article_number, article_title, article_id in articles:
            sources.append(f"  • {article_title}")
            url = get_article_url(act_info["act_id"], article_id)
            sources.append(f"    {url}")

    return "\n".join(sources)


def generate_final_answer(
    query: str,
    relevant_evidence,
    context,
    df_articles: pd.DataFrame,
) -> str:
    """
    Generate a legal answer based on the query, relevant evidence and their context

    Args:
        query (str): User query
        relevant_evidence (List[Tuple]): List of tuples containing relevant evidence
        context (List[Dict]): Context information for each chunk from get_document_context
        df_articles (pd.DataFrame): DataFrame containing article information

    Returns:
        str: Generated answer with legal analysis based on the evidence
    """
    api_version = os.getenv("AZURE_OPENAI_RIZIV_API_VERSION")
    endpoint = os.getenv("AZURE_OPENAI_RIZIV_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_RIZIV_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_RIZIV_DEPLOYMENT")

    if not all([api_version, endpoint, api_key, deployment]):
        raise ValueError("Missing required environment variables for Azure OpenAI")

    try:
        llm_final_answer_client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=api_key,
        )

        try:
            formatted_evidence = ""

            for i in range(len(relevant_evidence)):
                piece_of_ev = relevant_evidence[i][1]
                ev_article_n = get_article_title(
                    format_article_number(context[i]["article_info"]["article_number"]),
                    df_articles,
                )
                ev_act_title = context[i]["act_info"]["title_short"]

                formatted_evidence += (
                    f"Evidence {i}\n"
                    f"{ev_article_n} - {ev_act_title}\n"
                    f"{piece_of_ev}\n\n"
                )

            completion = llm_final_answer_client.beta.chat.completions.parse(
                model=deployment,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert legal analyst specializing in Belgian legal texts from the RIZIV/INAMI corpus. 
                    Your task is to generate a precise and accurate answer to the user's query by synthesizing the provided relevant evidence. 
                    Base your response exclusively on the provided relevant evidence context and terminology specific to the RIZIV/INAMI domain. 
                    Ensure that your answer addresses the query directly while using cautious language. Do not make overly definitive statements; instead, 
                    express your analysis in terms of likelihood or appearance. Include a disclaimer at the end indicating that the answer is based on an 
                    LLM's interpretation of the legal texts and should be independently verified by a legal professional.""",
                    },
                    {
                        "role": "user",
                        "content": f"""Based on the user query below and the relevant evidence provided from the RIZIV/INAMI legal corpus, 
                    generate a final legal answer that clearly and concisely addresses the query. Use the evidence to support your answer, 
                    ensuring it is strictly grounded in the relevant legal context.

                    Query:
                    {query}
                    
                    Pieces of evidence with context:
                    {formatted_evidence}""",
                    },
                ],
            )
            return completion.choices[0].message.content

        finally:
            llm_final_answer_client.close()

    except Exception as e:
        raise Exception(f"Error generating legal answer: {str(e)}")
