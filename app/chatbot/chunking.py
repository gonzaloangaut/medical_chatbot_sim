from app.chatbot.models import Chunk, Document


def chunk_document(document: Document) -> list[Chunk]:
    """
    Chunk a document into smaller parts based on section headers.

    Parameters
    ----------
    document : Document
        The document to be chunked.

    Returns
    -------
    list[Chunk]
        The chunks generated from the document sections.
    """
    # Split the document content into lines
    lines = document.content.splitlines()

    # Initialize the state used while processing sections
    current_section = None
    current_lines = []
    chunks = []

    for line in lines:
        if line.startswith("## "):
            # Save the previous section if it contains text
            content = "\n".join(current_lines).strip()

            if current_section is not None and content:
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.document_id}_{len(chunks):03d}",
                        document_id=document.document_id,
                        section=current_section,
                        content=content,
                    )
                )

            # Start a new section
            current_section = line.removeprefix("## ").strip()
            current_lines = []

        elif current_section is not None:
            current_lines.append(line)

    # Save the final section
    content = "\n".join(current_lines).strip()

    if current_section is not None and content:
        chunks.append(
            Chunk(
                chunk_id=f"{document.document_id}_{len(chunks):03d}",
                document_id=document.document_id,
                section=current_section,
                content=content,
            )
        )

    return chunks

def chunk_documents(documents: list[Document]) -> list[Chunk]:
    """
    Chunk multiple documents into a flat list of chunks.

    Parameters
    ----------
    documents : list[Document]
        The documents to be chunked.

    Returns
    -------
    list[Chunk]
        The chunks generated from all documents.
    """
    # Initialize a list to hold all chunks
    chunks = []

    # Iterate over each document and chunk it
    for document in documents:
        chunks.extend(chunk_document(document))

    return chunks