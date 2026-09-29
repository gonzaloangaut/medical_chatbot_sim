from textwrap import dedent

import pytest

from app.chatbot.knowledge import _load_document
from app.chatbot.models import Document


def test_load_document_returns_document(tmp_path):
    """
    Test that _load_document parses a valid Markdown document correctly.
    """
    # Create a temporary Markdown document
    doc_path = tmp_path / "test_doc.md"
    doc_content = dedent(
        """
        ---
        document_id: test_document
        title: Test Document
        source: Test Source
        source_url: https://example.com
        ---

        # Test Document

        ## Overview

        Some content.
        """
    ).strip()

    doc_path.write_text(doc_content, encoding="utf-8")

    # Load the document
    document = _load_document(doc_path)

    # Verify the parsed document
    assert isinstance(document, Document)
    assert document.document_id == "test_document"
    assert document.title == "Test Document"
    assert document.source == "Test Source"
    assert document.source_url == "https://example.com"
    assert document.content == "# Test Document\n\n## Overview\n\nSome content."


def test_load_document_missing_front_matter(tmp_path):
    """
    Test that _load_document raises ValueError when front matter is missing.
    """
    # Create a document without front matter
    doc_path = tmp_path / "test_doc.md"
    doc_content = dedent(
        """
        # Test Document

        Some content.
        """
    ).strip()

    doc_path.write_text(doc_content, encoding="utf-8")

    # Verify that loading fails
    with pytest.raises(ValueError, match="missing front matter"):
        _load_document(doc_path)


def test_load_document_missing_required_metadata(tmp_path):
    """
    Test that _load_document raises ValueError when required metadata is missing.
    """
    # Create a document with incomplete metadata
    doc_path = tmp_path / "test_doc.md"
    doc_content = dedent(
        """
        ---
        document_id: test_document
        title: Test Document
        ---

        # Test Document

        Some content.
        """
    ).strip()

    doc_path.write_text(doc_content, encoding="utf-8")

    # Verify that loading fails
    with pytest.raises(ValueError, match="Missing required metadata fields"):
        _load_document(doc_path)