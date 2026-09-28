from pathlib import Path
from app.chatbot.models import Document


def load_knowledge() -> str:
    """
    Load the knowledge base from a text file.

    Returns
    -------
    knowledge : str
        The content of the knowledge base.
    """
    # Define the path to the knowledge file
    project_root = Path(__file__).resolve().parent.parent.parent

    # Define the path to the context file
    context_path = project_root / "data" / "context.txt"

    # Read and return the content of the context file
    return context_path.read_text(encoding="utf-8")


def _load_document(path: Path) -> Document:
    """
    Load a document from a Markdown file.

    Parameters
    ----------
    path : Path
        Path to the Markdown document.

    Returns
    -------
    Document
        The parsed document.

    Raises
    ------
    ValueError
        If the document has invalid front matter or required metadata
        fields are missing.
    """
    # Read the document from disk
    text = path.read_text(encoding="utf-8")

    # Validate and separate front matter from document content
    if not text.startswith("---"):
        raise ValueError(f"Document {path} is missing front matter.")

    parts = text.split("---", 2)

    if len(parts) != 3:
        raise ValueError(f"Invalid front matter in {path}.")

    metadata_text = parts[1].strip()
    content = parts[2].strip()

    # Parse the front matter metadata
    metadata = {}

    for line in metadata_text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()

    # Validate required metadata
    required_fields = {
        "document_id",
        "title",
        "source",
        "source_url",
    }

    missing_fields = required_fields - metadata.keys()

    if missing_fields:
        raise ValueError(
            f"Missing required metadata fields in {path}: "
            f"{sorted(missing_fields)}"
        )

    return Document(
        document_id=metadata["document_id"],
        title=metadata["title"],
        source=metadata["source"],
        source_url=metadata["source_url"],
        content=content,
    )

def load_documents() -> list[Document]:
    """
    Load all documents from the knowledge directory.

    Returns
    -------
    list[Document]
        The parsed documents.
    """
    # Define the path to the knowledge directory
    project_root = Path(__file__).resolve().parent.parent.parent
    documents_dir = project_root / "data" / "knowledge"

    # Load all Markdown documents in deterministic order
    documents = []

    for path in sorted(documents_dir.glob("*.md")):
        documents.append(_load_document(path))

    return documents