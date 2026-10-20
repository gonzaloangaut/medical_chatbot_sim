from evaluation.evaluate_retrieval_threshold import evaluate_threshold


def test_evaluate_threshold():
    """
    Test the four possible outcomes of an acceptance decision.
    """
    results = [
        {
            "answerable": True,
            "retrieved_chunks": [{"score": 0.8}],
        },
        {
            "answerable": True,
            "retrieved_chunks": [{"score": 0.2}],
        },
        {
            "answerable": False,
            "retrieved_chunks": [{"score": 0.7}],
        },
        {
            "answerable": False,
            "retrieved_chunks": [{"score": 0.1}],
        },
    ]

    metrics = evaluate_threshold(results, threshold=0.5)

    assert metrics["tp"] == 1
    assert metrics["fn"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1


def test_score_equal_to_threshold():
    """
    Test that a score equal to the threshold is accepted.
    """
    results = [
        {
            "answerable": True,
            "retrieved_chunks": [{"score": 0.5}],
        },
    ]

    metrics = evaluate_threshold(results, threshold=0.5)

    assert metrics["tp"] == 1
    assert metrics["fn"] == 0
