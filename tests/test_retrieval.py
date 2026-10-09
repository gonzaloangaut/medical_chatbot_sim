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


def test_retrieve_returns_ranked_chunks(monkeypatch):
    """
    Test that retrieve returns ranked chunks when the top result
    passes the similarity threshold.
    """
    # Create a fake embedder and retriever
    embedder = FakeEmbedder()
    retriever = SemanticRetriever(
        embedder=embedder,
        threshold=0.3,
    )

    # Ingest sample chunks into the retriever
    chunks = [
        Chunk(
            chunk_id="document_000",
            document_id="document",
            title="Document",
            source="Test Source",
            source_url="https://example.com",
            section="Section Zero",
            content="Chunk zero.",
        ),
        Chunk(
            chunk_id="document_001",
            document_id="document",
            title="Document",
            source="Test Source",
            source_url="https://example.com",
            section="Section One",
            content="Chunk one.",
        ),
    ]

    # Ingest the chunks into the retriever
    retriever.ingest_chunks(chunks)

    # Monkeypatch the semantic_search function to return a fixed ranking of chunks
    def fake_semantic_search(
        query_embedding,
        corpus_embeddings,
        top_k,
    ):
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

    # Call the retrieve method with a query and top_k=2
    results = retriever.retrieve("some query", top_k=2)

    # Verify that the results are ranked by score and match the expected chunks
    assert len(results) == 2

    assert results[0].chunk == chunks[1]
    assert results[0].score == 0.8

    assert results[1].chunk == chunks[0]
    assert results[1].score == 0.6


def test_retrieve_returns_empty_list_when_score_is_below_threshold(
    monkeypatch,
):
    """
    Test that retrieve returns an empty list when the top similarity
    score is below the threshold.
    """
    # Create a fake embedder and retriever with a threshold of 0.3
    embedder = FakeEmbedder()
    retriever = SemanticRetriever(
        embedder=embedder,
        threshold=0.3,
    )

    # Ingest a sample chunk into the retriever
    chunks = [
        Chunk(
            chunk_id="document_000",
            document_id="document",
            title="Document",
            source="Test Source",
            source_url="https://example.com",
            section="Overview",
            content="Some content.",
        )
    ]

    # Ingest the chunk into the retriever
    retriever.ingest_chunks(chunks)

    # Monkeypatch the semantic_search function to return a score 
    # below the threshold
    def fake_semantic_search(
        query_embedding,
        corpus_embeddings,
        top_k,
    ):
        return [
            [
                {
                    "corpus_id": 0,
                    "score": 0.2,
                }
            ]
        ]

    monkeypatch.setattr(
        retrieval_module.util,
        "semantic_search",
        fake_semantic_search,
    )

    # Call the retrieve method with a query
    results = retriever.retrieve("some query")

    # Verify that the results are an empty list since the score is 
    # below the threshold
    assert results == []


def test_search_returns_ranked_results(monkeypatch):
    """
    Test that search returns chunks ranked by similarity score.
    """
    # Create a fake embedder and retriever
    embedder = FakeEmbedder()
    retriever = SemanticRetriever(embedder=embedder)

    # Ingest sample chunks into the retriever
    chunks = [
        Chunk(
            chunk_id="document_000",
            document_id="document",
            title="Document",
            source="Test Source",
            source_url="https://example.com",
            section="Section Zero",
            content="Chunk zero.",
        ),
        Chunk(
            chunk_id="document_001",
            document_id="document",
            title="Document",
            source="Test Source",
            source_url="https://example.com",
            section="Section One",
            content="Chunk one.",
        ),
    ]

    retriever.ingest_chunks(chunks)

    # Monkeypatch the semantic_search function to return a fixed ranking of chunks
    def fake_semantic_search(query_embedding, corpus_embeddings, top_k):
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


    # Call the search method with a query and top_k=2
    results = retriever.search("some query", top_k=2)

    # Verify that the results are ranked by score and match the expected chunks
    assert len(results) == 2

    assert results[0].score == 0.8
    assert results[0].chunk == chunks[1]

    assert results[1].score == 0.6
    assert results[1].chunk == chunks[0]


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
