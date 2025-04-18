import pandas as pd
import numpy as np


def split_text_with_overlap(text, chunk_size=4000, overlap=250):
    """
    Split a text into overlapping chunks of specified size

    Args:
        text (str): Input text to be split
        chunk_size (int): Size of each chunk (default: 4000 characters)
        overlap (int): Number of overlapping characters between chunks (default: 250 characters)

    Returns:
        list: List of text chunks with overlap between consecutive chunks
    """
    # Input validation: Handle None, empty text or non-string inputs
    if pd.isna(text) or text == "" or not isinstance(text, str):
        return [str(text)]

    # If text is shorter than chunk_size, return it as a single chunk
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    # Calculate effective chunk size by subtracting overlap
    # This ensures proper spacing between chunk starts
    effective_chunk_size = chunk_size - overlap

    # Calculate total number of chunks needed
    # Formula accounts for overlap to avoid missing text
    n_chunks = (len(text) - overlap) // effective_chunk_size + 1

    for i in range(n_chunks):
        # Calculate start and end positions for current chunk
        start = i * effective_chunk_size
        end = start + chunk_size

        # For all chunks except first, adjust start to include overlap with previous chunk
        if i > 0:
            start = max(0, start)

        # Ensure end position doesn't exceed text length
        end = min(len(text), end)

        # Extract the chunk from text
        chunk = text[start:end]

        # Only add non-empty chunks to result
        if chunk:
            chunks.append(chunk)

    # Special handling for last chunk if text remains
    if end < len(text):
        last_start = max(end - chunk_size, 0)  # Ensure we don't get negative start
        chunks.append(text[last_start:])

    return chunks


def verify_overlap(df, expected_overlap=250):
    """
    Verify that consecutive chunks from the same original text have the expected overlap

    Args:
        df (DataFrame): DataFrame containing the chunks
        expected_overlap (int): Expected number of overlapping characters (default: 250)
    """
    errors = 0
    error_details = []

    # Check each original text separately
    for orig_idx in df["original_index"].unique():
        # Get all chunks for current text, ordered by their position
        chunks = df[df["original_index"] == orig_idx].sort_values("chunk_index")

        # Only check overlap if text was split into multiple chunks
        if len(chunks) > 1:
            # Compare each pair of consecutive chunks
            for i in range(len(chunks) - 1):
                current_chunk = chunks.iloc[i]["text"]
                next_chunk = chunks.iloc[i + 1]["text"]

                # Extract the overlapping regions
                current_end = current_chunk[-expected_overlap:]
                next_start = next_chunk[:expected_overlap]

                # Count matching characters in overlap region
                actual_overlap = sum(
                    1 for a, b in zip(current_end, next_start) if a == b
                )

                # Record error if overlap doesn't match expected size
                if actual_overlap != expected_overlap:
                    errors += 1
                    error_details.append(
                        {
                            "original_index": orig_idx,
                            "chunk_pair": f"{i}-{i+1}",
                            "expected_overlap": expected_overlap,
                            "actual_overlap": actual_overlap,
                            "current_chunk_length": len(current_chunk),
                            "next_chunk_length": len(next_chunk),
                        }
                    )

    # Report results
    print(f"\nOverlap verification completed. Found {errors} errors.")

    # Show detailed information for first 5 errors if any found
    if errors > 0:
        print("\nSample of error details:")
        for i, error in enumerate(error_details[:5]):
            print(f"\nError {i+1}:")
            for k, v in error.items():
                print(f"{k}: {v}")
