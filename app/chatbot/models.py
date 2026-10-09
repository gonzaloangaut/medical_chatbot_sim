"""
Domain models used by the chatbot.

Classes:
    - Document: Represents a source document in the knowledge base.
    - Chunk: Represents a retrievable fragment of a document.
    - RetrievedChunk: Represents a chunk returned by the retriever.
"""

from dataclasses import dataclass


@dataclass
class Document:
    """
    Represent a source document in the knowledge base.

    Attributes
    ----------
    document_id : str
        Unique identifier of the document.
    title : str
        Title of the document.
    source : str
        Name of the source that provides the document.
    source_url : str
        URL of the original source.
    content : str
        Full text content of the document.
    """

    document_id: str
    title: str
    source: str
    source_url: str
    content: str


@dataclass
class Chunk:
    """
    Represent a retrievable fragment of a document.

    Attributes
    ----------
    chunk_id : str
        Unique identifier of the chunk.
    document_id : str
        Identifier of the source document.
    title : str
        Title of the source document.
    source : str
        Name of the source that provides the document.
    source_url : str
        URL of the original source.
    section : str
        Section of the document represented by the chunk.
    content : str
        Text content of the chunk.
    """

    chunk_id: str
    document_id: str
    title: str
    source: str
    source_url: str
    section: str
    content: str


@dataclass
class RetrievedChunk:
    """
    Represent a chunk returned by the retriever.

    Attributes
    ----------
    chunk : Chunk
        The retrieved chunk.
    score : float
        Similarity score assigned to the chunk for the current query.
    """

    chunk: Chunk
    score: float
