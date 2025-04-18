import requests

def get_ollama_embedding(text, ollama_base_url="http://localhost:11434"):
    """
    Generate text embeddings using the Ollama API.

    Parameters:
    -----------
    text : str
        The input text for which the embedding is to be generated.
    ollama_base_url : str, optional
        The base URL of the Ollama API (default is "http://localhost:11434").

    Returns:
    --------
    dict or None
        A dictionary containing the embedding if the request is successful, otherwise None.
    """
    url = f"{ollama_base_url}/api/embeddings"
    payload = {
        "model": "bge-m3",  # Ensure this model is available in your Ollama instance
        "prompt": text
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an error for HTTP request failures
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching embedding: {e}")
        return None
