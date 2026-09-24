import json
from pathlib import Path

from app.chatbot.retrieval import SemanticRetriever

# Define paths for the evaluation data and results
ROOT_DIR = Path(__file__).resolve().parent.parent
EVALUATION_DIR = Path(__file__).resolve().parent

CONTEXT_PATH = ROOT_DIR / "data" / "context.txt"
DATASET_PATH = EVALUATION_DIR / "retrieval_dataset.json"
CATALOG_PATH = EVALUATION_DIR / "retrieval_catalog.json"
RESULTS_PATH = EVALUATION_DIR / "retrieval_results.json"


def load_json(path: Path):
    """
    Load JSON data from a file.
    """
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_retriever(
    retriever: SemanticRetriever,
    dataset: list[dict],
    catalog: dict[int, str],
) -> list[dict]:
    """
    Evaluate a configured retriever on the retrieval benchmark.

    Parameters
    ----------
    retriever : SemanticRetriever
        The retriever to evaluate. It must already contain the indexed
        knowledge base.
    dataset : list[dict]
        Evaluation cases containing queries and expected results.
    catalog : dict[int, str]
        Mapping from corpus IDs to semantic document IDs.

    Returns
    -------
    list[dict]
        Per-query retrieval results, rankings, scores, and correctness metrics.
    """
    results = []

    for case in dataset:
        ranking = retriever.search(case["query"])

        best_result = ranking[0]
        predicted_id = catalog[best_result["corpus_id"]]
        best_score = best_result["score"]

        if best_score >= retriever.threshold:
            retrieved_id = predicted_id
        else:
            retrieved_id = None

        expected_rank = None

        if case["expected_id"] is not None:
            for rank, result in enumerate(ranking, start=1):
                result_id = catalog[result["corpus_id"]]

                if result_id == case["expected_id"]:
                    expected_rank = rank
                    break

        result = {
            "case_id": case["case_id"],
            "query": case["query"],
            "expected_id": case["expected_id"],
            "should_retrieve": case["should_retrieve"],
            "query_type": case["query_type"],
            "difficulty": case["difficulty"],
            "predicted_id": predicted_id,
            "best_score": best_score,
            "retrieved_id": retrieved_id,
            "expected_rank": expected_rank,
            "retrieval_correct": retrieved_id == case["expected_id"],
            "decision_correct": ((retrieved_id is not None) == case["should_retrieve"]),
            "ranking": [
                {
                    "rank": rank,
                    "id": catalog[item["corpus_id"]],
                    "score": item["score"],
                }
                for rank, item in enumerate(ranking, start=1)
            ],
        }

        results.append(result)

    return results


def main():
    """
    Run the baseline retrieval evaluation and save the results.
    """
    dataset = load_json(DATASET_PATH)
    catalog_data = load_json(CATALOG_PATH)

    catalog = {
        item["corpus_id"]: item["id"]
        for item in catalog_data
    }

    context_text = CONTEXT_PATH.read_text(
        encoding="utf-8"
    )

    retriever = SemanticRetriever()
    retriever.ingest_context(context_text)

    results = evaluate_retriever(
        retriever=retriever,
        dataset=dataset,
        catalog=catalog,
    )

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Evaluated {len(results)} cases.")
    print(f"Results saved to: {RESULTS_PATH}")

if __name__ == "__main__":
    main()
