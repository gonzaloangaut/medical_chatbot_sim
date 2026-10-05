from textwrap import dedent

from app.chatbot.chunking import chunk_document
from app.chatbot.models import Document

def test_chunk_document_creates_chunks_from_sections():
    """
    Test that chunk_document correctly creates chunks from document sections.
    """
    # Create a sample document with multiple sections
    document = Document(
        document_id="test_document",
        title="Test Document",
        source="Test Source",
        source_url="https://example.com",
        content=dedent(
            """
            # Test Document

            ## Overview

            Overview content.

            ## Symptoms

            Symptoms content.
            """
        ).strip(),
    )

    # Chunk the document
    chunks = chunk_document(document)

    # Assert that the correct number of chunks were created
    assert len(chunks) == 2

    # Assert that the chunks have the expected content and metadata
    assert chunks[0].chunk_id == "test_document_000"
    assert chunks[0].document_id == "test_document"
    assert chunks[0].section == "Overview"
    assert chunks[0].content == "Overview content."

    assert chunks[1].chunk_id == "test_document_001"
    assert chunks[1].document_id == "test_document"
    assert chunks[1].section == "Symptoms"
    assert chunks[1].content == "Symptoms content."


def test_chunk_document_ignores_content_before_first_section():
    """
    Test that chunk_document ignores content before the first section header.
    """
    # Create a sample document with content before the first section
    document = Document(
        document_id="test_document",
        title="Test Document",
        source="Test Source",
        source_url="https://example.com",
        content=dedent(
            """
            # Test Document

            This content should be ignored.

            ## Overview

            Overview content.
            """
        ).strip(),
    )

    # Chunk the document
    chunks = chunk_document(document)

    # Assert that only one chunk was created
    assert len(chunks) == 1

    # Assert that the chunk has the expected content and metadata
    assert chunks[0].chunk_id == "test_document_000"
    assert chunks[0].document_id == "test_document"
    assert chunks[0].section == "Overview"
    assert chunks[0].content == "Overview content."


def test_chunk_document_ignores_empty_sections():
    """
    Test that chunk_document ignores empty sections.
    """
    # Create a sample document with an empty section
    document = Document(
        document_id="test_document",
        title="Test Document",
        source="Test Source",
        source_url="https://example.com",
        content=dedent(
            """
            # Test Document

            ## Overview

            Overview content.

            ## Empty Section

            ## Symptoms

            Symptoms content.
            """
        ).strip(),
    )

    # Chunk the document
    chunks = chunk_document(document)

    # Assert that only two chunks were created (empty section should be ignored)
    assert len(chunks) == 2

    # Assert that the chunks have the expected content and metadata
    assert chunks[0].chunk_id == "test_document_000"
    assert chunks[0].document_id == "test_document"
    assert chunks[0].section == "Overview"
    assert chunks[0].content == "Overview content."

    assert chunks[1].chunk_id == "test_document_001"
    assert chunks[1].document_id == "test_document"
    assert chunks[1].section == "Symptoms"
    assert chunks[1].content == "Symptoms content."