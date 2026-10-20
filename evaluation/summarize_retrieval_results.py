import argparse
import json
from pathlib import Path
from statistics import mean

from evaluation.retrieval_metrics import (
    complete_evidence_at_k,
    hit_at_1,
    recall_at_k,
    reciprocal_rank,
)


def calculate_query_metrics(query_result: dict) -> dict:
    """
    Calculate retrieval metrics for a single answerable query.
    """
    if not query_result["answerable"]:
        raise ValueError("Ranking metrics require an answerable query.")

    # Extract retrieved and relevant chunk IDs
    retrieved_ids = [item["chunk_id"] for item in query_result["retrieved_chunks"]]

    relevant_ids = query_result["relevant_chunk_ids"]

    # Calculate metrics
    metrics = {
        "query_id": query_result["query_id"],
        "query": query_result["query"],
        "category": query_result["category"],
        "answerable": query_result["answerable"],
        "retrieved_ids": retrieved_ids,
        "relevant_ids": relevant_ids,
        "hit_at_1": hit_at_1(retrieved_ids, relevant_ids),
        "recall_at_1": recall_at_k(retrieved_ids, relevant_ids, k=1),
        "recall_at_3": recall_at_k(retrieved_ids, relevant_ids, k=3),
        "recall_at_5": recall_at_k(retrieved_ids, relevant_ids, k=5),
        "complete_evidence_at_1": complete_evidence_at_k(
            retrieved_ids, relevant_ids, k=1
        ),
        "complete_evidence_at_3": complete_evidence_at_k(
            retrieved_ids, relevant_ids, k=3
        ),
        "complete_evidence_at_5": complete_evidence_at_k(
            retrieved_ids, relevant_ids, k=5
        ),
        "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant_ids),
    }

    return metrics


def summarize_metrics(query_metrics: list[dict]) -> dict:
    """
    Calculate mean retrieval metrics across answerable queries.
    """
    if not query_metrics:
        raise ValueError("Cannot summarize an empty list of query metrics.")

    summary = {
        "n_queries": len(query_metrics),
        "hit_at_1": mean(qm["hit_at_1"] for qm in query_metrics),
        "recall_at_1": mean(qm["recall_at_1"] for qm in query_metrics),
        "recall_at_3": mean(qm["recall_at_3"] for qm in query_metrics),
        "recall_at_5": mean(qm["recall_at_5"] for qm in query_metrics),
        "complete_evidence_at_1": mean(
            qm["complete_evidence_at_1"] for qm in query_metrics
        ),
        "complete_evidence_at_3": mean(
            qm["complete_evidence_at_3"] for qm in query_metrics
        ),
        "complete_evidence_at_5": mean(
            qm["complete_evidence_at_5"] for qm in query_metrics
        ),
        "mrr_at_5": mean(qm["reciprocal_rank"] for qm in query_metrics),
    }

    return summary


def main():
    """
    Load retrieval results, calculate metrics, and save a summary.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Summarize retrieval benchmark results."
    )

    parser.add_argument(
        "--dataset",
        choices=("dev", "test"),
        required=True,
        help="Dataset split to summarize.",
    )

    args = parser.parse_args()

    # Define input and output paths
    evaluation_dir = Path(__file__).resolve().parent

    results_path = evaluation_dir / "results" / f"retrieval_{args.dataset}.json"

    output_path = evaluation_dir / "results" / f"retrieval_summary_{args.dataset}.json"

    # Load benchmark results
    with results_path.open("r", encoding="utf-8") as file:
        benchmark_data = json.load(file)

    # Verify the retrieval configuration
    if benchmark_data["top_k"] != 5:
        raise ValueError("This summary expects a Top-5 retrieval benchmark.")

    results = benchmark_data["results"]

    # Filter answerable queries
    answerable_results = [item for item in results if item["answerable"]]

    # Calculate individual query metrics
    query_metrics = [calculate_query_metrics(item) for item in answerable_results]

    # Calculate overall metrics
    summary = {
        "overall": summarize_metrics(query_metrics),
    }

    # Calculate metrics by category
    for category in ("single_evidence", "multi_evidence"):
        category_metrics = [
            item for item in query_metrics if item["category"] == category
        ]

        if category_metrics:
            summary[category] = summarize_metrics(category_metrics)

    # Build evaluation report
    report = {
        "model_name": benchmark_data["model_name"],
        "dataset": benchmark_data["dataset"],
        "top_k": benchmark_data["top_k"],
        "n_queries": len(results),
        "n_answerable": len(answerable_results),
        "n_unsupported": len(results) - len(answerable_results),
        "summary": summary,
        "query_metrics": query_metrics,
    }

    # Save evaluation report
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    # Display results
    print("\nRetrieval Benchmark Summary")
    print(f"Model: {report['model_name']}")
    print(f"Queries: {report['n_queries']}")
    print(f"Answerable: {report['n_answerable']}")
    print(f"Unsupported: {report['n_unsupported']}")

    for category, metrics in summary.items():
        print(f"\n{category} ({metrics['n_queries']} queries)")

        for metric_name, value in metrics.items():
            if metric_name != "n_queries":
                print(f"  {metric_name}: {value:.4f}")

    print(f"\nSummary saved to: {output_path}")


if __name__ == "__main__":
    main()
