import pytest

from evaluation.retrieval_metrics import (
    complete_evidence_at_k,
    hit_at_1,
    recall_at_k,
    reciprocal_rank,
)


def test_first_retrieved_is_relevant():
    """
    Test that hit_at_1 returns 1 when the first retrieved ID is relevant.
    """
    # Test retrieved IDs
    retrieved_ids = ["id1", "id2", "id3"]
    # Relevant IDs
    relevant_ids = ["id1", "id4"]
    # Verify that the first retrieved ID is relevant
    assert hit_at_1(retrieved_ids, relevant_ids) == 1


def test_first_retrieved_is_not_relevant():
    """
    Test that hit_at_1 returns 0 when the first retrieved ID is not relevant.
    """
    # Test retrieved IDs
    retrieved_ids = ["id2", "id3", "id4"]
    # Relevant IDs
    relevant_ids = ["id1", "id4"]
    # Verify that the first retrieved ID is not relevant
    assert hit_at_1(retrieved_ids, relevant_ids) == 0


def test_no_retrieved_ids():
    """
    Test that hit_at_1 returns 0 when there are no retrieved IDs.
    """
    # Test retrieved IDs
    retrieved_ids = []
    # Relevant IDs
    relevant_ids = ["id1", "id4"]
    # Verify that there are no retrieved IDs
    assert hit_at_1(retrieved_ids, relevant_ids) == 0


def test_recall_at_k_with_relevant_ids():
    """
    Test that recall_at_k returns the correct recall value when there are
    relevant IDs.
    """
    # Test retrieved IDs
    retrieved_ids = ["id1", "id2", "id3"]
    # Relevant IDs
    relevant_ids = ["id1", "id4"]
    # Verify that the recall value is correct
    assert recall_at_k(retrieved_ids, relevant_ids, k=2) == 0.5


def test_recall_at_k_total_relevant_ids():
    """
    Test that recall_at_k returns 1.0 when all relevant IDs are retrieved
    within Top-k.
    """
    # Test retrieved IDs
    retrieved_ids = ["id1", "id2", "id3"]
    # Relevant IDs
    relevant_ids = ["id1", "id2"]
    # Verify that all relevant IDs are retrieved within Top-k
    assert recall_at_k(retrieved_ids, relevant_ids, k=3) == 1.0


def test_recall_at_k_with_empty_retrieval():
    """
    Test that Recall@k is zero when no chunks are retrieved.
    """
    retrieved_ids = []
    relevant_ids = ["id1", "id2"]

    assert recall_at_k(retrieved_ids, relevant_ids, k=3) == 0.0


def test_complete_evidence_at_k():
    """
    Test that complete_evidence_at_k returns the correct value.
    """
    # Test retrieved IDs
    retrieved_ids = ["id1", "id2", "id3"]
    # Relevant IDs
    relevant_ids = ["id1", "id2"]
    # Verify that the complete evidence value is correct
    assert complete_evidence_at_k(retrieved_ids, relevant_ids, k=3) == 1


def test_complete_evidence_at_k_not_all_retrieved():
    """
    Test that complete_evidence_at_k returns 0 when not all relevant IDs
    are retrieved within Top-k.
    """
    # Test retrieved IDs
    retrieved_ids = ["id1", "id3", "id4"]
    # Relevant IDs
    relevant_ids = ["id1", "id2"]
    # Verify that not all relevant IDs are retrieved within Top-k
    assert complete_evidence_at_k(retrieved_ids, relevant_ids, k=3) == 0


def test_complete_evidence_at_k_with_limited_k():
    """
    Test that evidence outside Top-k does not count as retrieved.
    """
    retrieved_ids = ["id1", "id3", "id2"]
    relevant_ids = ["id1", "id2"]

    assert complete_evidence_at_k(retrieved_ids, relevant_ids, k=2) == 0
    assert complete_evidence_at_k(retrieved_ids, relevant_ids, k=3) == 1


@pytest.mark.parametrize("metric", [recall_at_k, complete_evidence_at_k])
@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k(metric, k):
    """
    Test that ranking metrics reject non-positive k values.
    """
    retrieved_ids = ["id1", "id2"]
    relevant_ids = ["id1"]

    with pytest.raises(ValueError, match="k must be a positive integer"):
        metric(retrieved_ids, relevant_ids, k=k)


@pytest.mark.parametrize("metric", [recall_at_k, complete_evidence_at_k])
def test_relevant_ids_empty(metric):
    """
    Test that ranking metrics reject empty ground truth.
    """
    # Test retrieved IDs
    retrieved_ids = ["id1", "id2", "id3"]

    # Relevant IDs
    relevant_ids = []

    # Verify that a ValueError is raised for empty relevant_ids
    with pytest.raises(ValueError, match="Relevant IDs cannot be empty"):
        metric(retrieved_ids, relevant_ids, k=2)


def test_reciprocal_rank_first_result():
    """
    Test that reciprocal rank is 1.0 when the first result is relevant.
    """
    retrieved_ids = ["id1", "id2", "id3"]
    relevant_ids = ["id1", "id3"]

    assert reciprocal_rank(retrieved_ids, relevant_ids) == 1.0


def test_reciprocal_rank_second_result():
    """
    Test that reciprocal rank is 0.5 when the first relevant result
    appears in second position.
    """
    retrieved_ids = ["id2", "id1", "id3"]
    relevant_ids = ["id1", "id3"]

    assert reciprocal_rank(retrieved_ids, relevant_ids) == 0.5


def test_reciprocal_rank_no_relevant_results():
    """
    Test that reciprocal rank is zero when no relevant chunks are retrieved.
    """
    retrieved_ids = ["id2", "id4", "id5"]
    relevant_ids = ["id1", "id3"]

    assert reciprocal_rank(retrieved_ids, relevant_ids) == 0.0


def test_reciprocal_rank_empty_ground_truth():
    """
    Test that reciprocal rank rejects an empty ground truth.
    """
    retrieved_ids = ["id1", "id2"]
    relevant_ids = []

    with pytest.raises(ValueError, match="Relevant IDs cannot be empty"):
        reciprocal_rank(retrieved_ids, relevant_ids)
