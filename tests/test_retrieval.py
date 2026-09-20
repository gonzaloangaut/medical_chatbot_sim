from app.chatbot import retrieval as retrieval_module
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

    assert retriever.search_keys == ["fiebre temperatura alta"]

    assert retriever.chunks == ["fiebre temperatura alta"]


def test_retrieve_returns_best_chunk(monkeypatch):
    """
    Test that retrieve returns the chunk selected by semantic search.
    """
    embedder = FakeEmbedder()

    retriever = SemanticRetriever(
        embedder=embedder,
    )

    context = """
    fiebre @@@ Protocolo para fiebre.
    ###
    dolor de cabeza @@@ Protocolo para cefalea.
    """

    retriever.ingest_context(context)

    def fake_semantic_search(query_embedding, embeddings, top_k=1):
        return [
            [
                {
                    "score": 0.8,
                    "corpus_id": 1,
                }
            ]
        ]

    monkeypatch.setattr(
        retrieval_module.util,
        "semantic_search",
        fake_semantic_search,
    )

    result = retriever.retrieve("Tengo dolor de cabeza")

    assert result == "Protocolo para cefalea."


def test_retrieve_returns_none_when_score_is_below_threshold(monkeypatch):
    """
    Test that retrieve returns None when the similarity score
    is below the minimum threshold.
    """
    embedder = FakeEmbedder()

    retriever = SemanticRetriever(
        embedder=embedder,
    )

    context = """
    fiebre @@@ Protocolo para fiebre.
    ###
    dolor de cabeza @@@ Protocolo para cefalea.
    """

    retriever.ingest_context(context)

    def fake_semantic_search(query_embedding, embeddings, top_k=1):
        return [
            [
                {
                    "score": 0.2,
                    "corpus_id": 1,
                }
            ]
        ]

    monkeypatch.setattr(
        retrieval_module.util,
        "semantic_search",
        fake_semantic_search,
    )

    result = retriever.retrieve("Tengo dolor de cabeza")

    assert result is None
