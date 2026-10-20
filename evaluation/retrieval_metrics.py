def hit_at_1(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> int:
    """
    Return 1 if the first retrieved chunk is relevant, otherwise 0.
    """
    if not relevant_ids:
        raise ValueError("Relevant IDs cannot be empty.")

    if not retrieved_ids:
        return 0

    return int(retrieved_ids[0] in relevant_ids)


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """
    Calculate Recall@k for a single query.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    if not relevant_ids:
        raise ValueError("Relevant IDs cannot be empty.")

    retrieved_at_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)

    relevant_retrieved = retrieved_at_k & relevant

    return len(relevant_retrieved) / len(relevant)


def complete_evidence_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> int:
    """
    Return 1 if all required evidence is retrieved within Top-k.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    if not relevant_ids:
        raise ValueError("Relevant IDs cannot be empty.")

    retrieved_at_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)

    return int(relevant.issubset(retrieved_at_k))


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> float:
    """
    Calculate reciprocal rank for a single query.

    It considers only the first relevant document retrieved.
    If no relevant document is retrieved, it returns 0.
    """
    if not relevant_ids:
        raise ValueError("Relevant IDs cannot be empty.")

    for rank, retrieved_id in enumerate(retrieved_ids, start=1):
        if retrieved_id in relevant_ids:
            return 1.0 / rank

    return 0.0
