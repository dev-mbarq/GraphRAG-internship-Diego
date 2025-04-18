from pydantic import BaseModel


# Define Pydantic Schema (Structured outputs) for concept extraction
class ConceptExtraction(BaseModel):
    # Concept 1
    concept1_name: str
    concept1_category: str
    concept1_relevance: float

    # Concept 2
    concept2_name: str
    concept2_category: str
    concept2_relevance: float

    # Concept 3
    concept3_name: str
    concept3_category: str
    concept3_relevance: float

    # Concept 4
    concept4_name: str
    concept4_category: str
    concept4_relevance: float

    # Concept 5
    concept5_name: str
    concept5_category: str
    concept5_relevance: float

    # Concept 6
    concept6_name: str
    concept6_category: str
    concept6_relevance: float

    # Concept 7
    concept7_name: str
    concept7_category: str
    concept7_relevance: float

    # Concept 8
    concept8_name: str
    concept8_category: str
    concept8_relevance: float

    # Concept 9
    concept9_name: str
    concept9_category: str
    concept9_relevance: float

    # Concept 10
    concept10_name: str
    concept10_category: str
    concept10_relevance: float


class CitationExtraction(BaseModel):
    # Citation 1
    citation1: str

    # Citation 2
    citation2: str
