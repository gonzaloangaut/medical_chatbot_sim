import json

from app.chatbot.retrieval import SemanticRetriever
from evaluation.run_retrieval_evaluation import (
    CATALOG_PATH,
    CONTEXT_PATH,
    DATASET_PATH,
    EVALUATION_DIR,
    evaluate_retriever,
    load_json,
)

RESULTS_PATH = EVALUATION_DIR / "representation_experiment_results.json"

REPRESENTATIONS = [
    "search_keys",
    "content",
    "search_keys_and_content",
]


def main():
    """
    Compare different document representations on the retrieval benchmark.
    """
    dataset = load_json(DATASET_PATH)
    catalog_data = load_json(CATALOG_PATH)

    catalog = {item["corpus_id"]: item["id"] for item in catalog_data}

    context_text = CONTEXT_PATH.read_text(encoding="utf-8")

    experiment_results = []

    for representation in REPRESENTATIONS:
        print(f"Evaluating: {representation}")

        retriever = SemanticRetriever(
            representation=representation,
        )
        retriever.ingest_context(context_text)

        results = evaluate_retriever(
            retriever=retriever,
            dataset=dataset,
            catalog=catalog,
        )

        for result in results:
            result["representation"] = representation

        experiment_results.extend(results)

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            experiment_results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Evaluated {len(REPRESENTATIONS)} representations "
        f"on {len(dataset)} queries."
    )
    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
