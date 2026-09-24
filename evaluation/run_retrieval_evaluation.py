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


def main():
    """
    Run the retrieval evaluation using the provided dataset and catalog.
    """
    # Load evaluation data
    dataset = load_json(DATASET_PATH)
    catalog_data = load_json(CATALOG_PATH)

    # Convert catalog to a dictionary:
    # {0: "fever", 1: "headache", ...}
    catalog = {item["corpus_id"]: item["id"] for item in catalog_data}

    # Load and index the knowledge base
    context_text = CONTEXT_PATH.read_text(encoding="utf-8")

    # Initialize the semantic retriever and ingest the context
    retriever = SemanticRetriever()
    retriever.ingest_context(context_text)

    # Evaluate each case in the dataset
    results = []

    for case in dataset:
        # Search for the most similar pieces of text to the query
        ranking = retriever.search(case["query"])
        # Get the best result
        best_result = ranking[0]

        # Take the predicted id from the catalog
        predicted_id = catalog[best_result["corpus_id"]]
        # And the score
        best_score = best_result["score"]

        # Apply the current retrieval policy
        if best_score >= retriever.threshold:
            retrieved_id = predicted_id
        else:
            retrieved_id = None

        # Find the rank of the expected document
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

    # Store the results
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
