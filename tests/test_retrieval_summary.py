import pytest

from evaluation.summarize_retrieval_results import (
    calculate_query_metrics,
    summarize_metrics,
)


def test_calculate_query_metrics():
    """
    Test metric calculation for a multi-evidence query.
    """
    query_result = {
        "query_id": "q_test",
        "query": "Test query",
        "category": "multi_evidence",
        "answerable": True,
        "relevant_chunk_ids": ["A", "B"],
        "retrieved_chunks": [
            {"chunk_id": "A", "rank": 1, "score": 0.9},
            {"chunk_id": "C", "rank": 2, "score": 0.8},
            {"chunk_id": "D", "rank": 3, "score": 0.7},
            {"chunk_id": "B", "rank": 4, "score": 0.6},
            {"chunk_id": "E", "rank": 5, "score": 0.5},
        ],
    }

    metrics = calculate_query_metrics(query_result)

    assert metrics["hit_at_1"] == 1
    assert metrics["recall_at_1"] == 0.5
    assert metrics["recall_at_3"] == 0.5
    assert metrics["recall_at_5"] == 1.0
    assert metrics["complete_evidence_at_1"] == 0
    assert metrics["complete_evidence_at_3"] == 0
    assert metrics["complete_evidence_at_5"] == 1
    assert metrics["reciprocal_rank"] == 1.0


def test_summarize_metrics():
    """
    Test summarization of metrics across multiple queries.
    """
    metrics = []

    # First query: multi-evidence
    query_result_1 = {
        "query_id": "q_test_1",
        "query": "First test query",
        "category": "multi_evidence",
        "answerable": True,
        "relevant_chunk_ids": ["A", "B"],
        "retrieved_chunks": [
            {"chunk_id": "A", "rank": 1, "score": 0.9},
            {"chunk_id": "C", "rank": 2, "score": 0.8},
            {"chunk_id": "D", "rank": 3, "score": 0.7},
            {"chunk_id": "B", "rank": 4, "score": 0.6},
            {"chunk_id": "E", "rank": 5, "score": 0.5},
        ],
    }

    metrics.append(calculate_query_metrics(query_result_1))

    # Second query: single-evidence
    query_result_2 = {
        "query_id": "q_test_2",
        "query": "Second test query",
        "category": "single_evidence",
        "answerable": True,
        "relevant_chunk_ids": ["A"],
        "retrieved_chunks": [
            {"chunk_id": "C", "rank": 1, "score": 0.9},
            {"chunk_id": "A", "rank": 2, "score": 0.8},
            {"chunk_id": "B", "rank": 3, "score": 0.7},
            {"chunk_id": "D", "rank": 4, "score": 0.6},
            {"chunk_id": "E", "rank": 5, "score": 0.5},
        ],
    }

    metrics.append(calculate_query_metrics(query_result_2))

    # Calculate aggregated metrics
    summary = summarize_metrics(metrics)

    # Verify the number of queries
    assert summary["n_queries"] == 2

    # Verify mean ranking metrics
    assert summary["hit_at_1"] == pytest.approx(0.5)
    assert summary["recall_at_1"] == pytest.approx(0.25)
    assert summary["recall_at_3"] == pytest.approx(0.75)
    assert summary["recall_at_5"] == pytest.approx(1.0)

    # Verify mean complete evidence metrics
    assert summary["complete_evidence_at_1"] == 0
    assert summary["complete_evidence_at_3"] == pytest.approx(0.5)
    assert summary["complete_evidence_at_5"] == 1

    # Verify Mean Reciprocal Rank
    assert summary["mrr_at_5"] == pytest.approx(0.75)


def test_unsupported_query():
    """
    Test that ranking metrics reject an unsupported query.
    """
    query_result = {
        "query_id": "q_unsupported",
        "query": "How much does a blood test cost?",
        "category": "in_domain_unsupported",
        "answerable": False,
        "relevant_chunk_ids": [],
        "retrieved_chunks": [
            {"chunk_id": "A", "rank": 1, "score": 0.7},
            {"chunk_id": "B", "rank": 2, "score": 0.5},
        ],
    }

    with pytest.raises(
        ValueError,
        match="Ranking metrics require an answerable query",
    ):
        calculate_query_metrics(query_result)


def test_summarize_empty_metrics():
    """
    Test that an empty metric list raises a ValueError.
    """
    with pytest.raises(
        ValueError,
        match="Cannot summarize an empty list of query metrics",
    ):
        summarize_metrics([])
