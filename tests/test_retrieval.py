from app.chatbot.retrieval import SemanticRetriever


class FakeEmbedder:
    """
    Fake Embedder class to simulate the behavior of the SentenceTransformer 
    for testing purposes.
    """
    def encode(self, texts, convert_to_tensor=True):
        """
        Simulate the embedding process by returning a fixed string.
        """
        return "EMBEDDINGS_FAKE"


def test_ingest_context_splits_keys_and_chunks():
    """
    Test that the ingest_context method correctly splits the context into 
    search keys and chunks.
    """
    embedder = FakeEmbedder()

    retriever = SemanticRetriever(
        embedder=embedder,
    )

    context = """
    fiebre temperatura alta @@@ Protocolo para fiebre.
    ###
    dolor de cabeza cefalea @@@ Protocolo para cefalea.
    """

    retriever.ingest_context(context)

    assert retriever.search_keys == [
        "fiebre temperatura alta",
        "dolor de cabeza cefalea",
    ]

    assert retriever.chunks == [
        "Protocolo para fiebre.",
        "Protocolo para cefalea.",
    ]


def test_ingest_context_uses_full_block_when_no_separator():
    """
    Test that a block without @@@ is used both as search key and chunk.
    """
    embedder = FakeEmbedder()

    retriever = SemanticRetriever(
        embedder=embedder,
    )

    context = """
    fiebre temperatura alta
    """

    retriever.ingest_context(context)

    assert retriever.search_keys == [
        "fiebre temperatura alta"
    ]

    assert retriever.chunks == [
        "fiebre temperatura alta"
    ]