import pytest

from app.chatbot import retrieval as retrieval_module
from app.chatbot.models import Chunk
from app.chatbot.retrieval import SemanticRetriever


class FakeEmbedder:
    """
    Fake Embedder class to simulate the behavior of the SentenceTransformer
    for testing purposes.
    """

    def __init__(self):
        self.last_input = None

    def encode(self, texts, convert_to_tensor=True):
        """
        Simulate the embedding process by returning a fixed string.
        """
        self.last_input = texts
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
        """
        Simulate the behavior of semantic_search by returning a fixed result.
        """
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
        """
        Simulate the behavior of semantic_search by returning a result
        with a score below the threshold.
        """
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


def test_search_returns_ranked_results(monkeypatch):
    """
    Test that the search method returns results ranked by similarity score.
    """
    # Create a fake embedder and retriever
    embedder = FakeEmbedder()
    retriever = SemanticRetriever(embedder=embedder)

    # Ingest some context into the retriever
    retriever.ingest_context("""
        key zero
        @@@
        chunk zero
        ###
        key one
        @@@
        chunk one
        """)

    # Monkeypatch the semantic_search function to return controlled results
    def fake_semantic_search(query_embedding, corpus_embeddings, top_k):
        """
        Simulate the behavior of semantic_search by returning a fixed set of results.
        """
        return [
            [
                {"corpus_id": 1, "score": 0.8},
                {"corpus_id": 0, "score": 0.6},
            ]
        ]

    monkeypatch.setattr(
        retrieval_module.util,
        "semantic_search",
        fake_semantic_search,
    )

    # Call the search method and check the results
    results = retriever.search("some query", top_k=2)

    assert len(results) == 2

    assert results[0]["corpus_id"] == 1
    assert results[0]["score"] == 0.8
    assert results[0]["chunk"] == "chunk one"

    assert results[1]["corpus_id"] == 0
    assert results[1]["score"] == 0.6
    assert results[1]["chunk"] == "chunk zero"


@pytest.mark.parametrize(
    "representation, expected_input",
    [
        ("search_keys", ["fever keys"]),
        ("content", ["fever protocol"]),
        (
            "search_keys_and_content",
            ["fever keys\nfever protocol"],
        ),
    ],
)
def test_document_representation_is_used_for_embedding(
    representation,
    expected_input,
):
    """
    Test that the selected document representation is passed to the embedder.

    The test verifies the three supported representations:
    search keys only, chunk content only, and search keys combined with content.
    """
    embedder = FakeEmbedder()

    retriever = SemanticRetriever(
        embedder=embedder,
        representation=representation,
    )

    retriever.ingest_context("fever keys @@@ fever protocol")

    assert embedder.last_input == expected_input


def test_invalid_representation_raises_value_error():
    """
    Test that an unsupported document representation raises a ValueError.
    """
    embedder = FakeEmbedder()

    retriever = SemanticRetriever(
        embedder=embedder,
        representation="invalid",
    )

    with pytest.raises(
        ValueError,
        match="Unknown document representation",
    ):
        retriever.ingest_context("fever keys @@@ fever protocol")


def test_build_embedding_text_uses_title_section_and_content():
    """
    Test that the _build_embedding_text method constructs the embedding text
    using the chunk's title, section, and content.
    """
    # Create a sample chunk
    chunk = Chunk(
        chunk_id="fever_000",
        document_id="fever",
        title="Fever",
        source="Test Source",
        source_url="https://example.com/fever",
        section="Common Causes",
        content="Infections are a common cause of fever.",
    )

    # Create a retriever instance without initializing the embedder
    retriever = SemanticRetriever.__new__(SemanticRetriever)

    # Call the _build_embedding_text method
    text = retriever._build_embedding_text(chunk)

    # Assert that the constructed text includes the title, section, and content
    assert text == (
        "Fever\n\n" "Common Causes\n\n" "Infections are a common cause of fever."
    )


def test_ingest_chunks_embeds_chunk_representations():
    """
    Test that ingest_chunks correctly builds the embedding text for each chunk
    and passes it to the embedder for embedding.
    """
    # Create sample chunks
    chunks = [
        Chunk(
            chunk_id="fever_000",
            document_id="fever",
            title="Fever",
            source="Test Source",
            source_url="https://example.com/fever",
            section="Overview",
            content="Overview content.",
        ),
        Chunk(
            chunk_id="fever_001",
            document_id="fever",
            title="Fever",
            source="Test Source",
            source_url="https://example.com/fever",
            section="Symptoms",
            content="Symptoms content.",
        ),
    ]

    # Create a fake embedder that records the input it receives
    fake_model = FakeEmbedder()

    # Create a retriever instance with the fake embedder
    retriever = SemanticRetriever.__new__(SemanticRetriever)
    retriever.embedder = fake_model

    # Ingest the sample chunks
    retriever.ingest_chunks(chunks)

    # Verify that the original chunks are stored
    assert retriever.chunks == chunks

    # Verify the text representations passed to the embedder
    assert fake_model.last_input == [
        "Fever\n\nOverview\n\nOverview content.",
        "Fever\n\nSymptoms\n\nSymptoms content.",
    ]

    # Verify that the returned embeddings are stored
    assert retriever.embeddings == "EMBEDDINGS_FAKE"
