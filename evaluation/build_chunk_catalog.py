import json
from pathlib import Path

from app.chatbot.chunking import chunk_documents
from app.chatbot.knowledge import load_documents


def build_chunk_catalog():
    """
    Build a catalog of all chunks in the knowledge base.
    """

    # Load documents and chunk them
    documents = load_documents()
    chunks = chunk_documents(documents)

    # Build the catalog as a list of dictionaries
    catalog = []

    # Iterate over each chunk and add its details to the catalog
    for chunk in chunks:
        catalog.append(
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "title": chunk.title,
                "section": chunk.section,
                "content": chunk.content,
                "source": chunk.source,
                "source_url": chunk.source_url,
            }
        )

    return catalog


def main():
    # Build the chunk catalog
    catalog = build_chunk_catalog()

    # Validate the catalog
    if not catalog:
        raise ValueError("The knowledge base contains no chunks.")

    # Calculate chunk statistics
    word_counts = [len(item["content"].split()) for item in catalog]

    document_count = len({item["document_id"] for item in catalog})

    # Define the output path
    output_path = Path(__file__).resolve().parent / "datasets" / "chunk_catalog.json"

    # Save the catalog as JSON
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            catalog,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # Print catalog statistics
    print(f"Documents: {document_count}")
    print(f"Chunks: {len(catalog)}")
    print(f"Min words: {min(word_counts)}")
    print(f"Max words: {max(word_counts)}")
    print(f"Mean words: {sum(word_counts) / len(word_counts):.2f}")
    print(f"Catalog saved to: {output_path}")


if __name__ == "__main__":
    main()
