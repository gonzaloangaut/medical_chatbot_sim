import argparse
import json
from pathlib import Path


def evaluate_threshold(results: list[dict], threshold: float) -> dict:
    """
    Evaluate acceptance and rejection decisions for a given threshold.
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for result in results:
        top_score = result["retrieved_chunks"][0]["score"]

        predicted_answerable = top_score >= threshold
        actually_answerable = result["answerable"]

        # Update counts based on predictions and actual labels
        if predicted_answerable and actually_answerable:
            tp += 1
        elif not predicted_answerable and not actually_answerable:
            tn += 1
        elif predicted_answerable and not actually_answerable:
            fp += 1
        elif not predicted_answerable and actually_answerable:
            fn += 1

    return {
        "threshold": threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main():
    """Compare thresholds on dev, or report the frozen 0.30 gate on test."""
    parser = argparse.ArgumentParser(description="Evaluate retrieval threshold.")
    parser.add_argument("--dataset", choices=("dev", "test"), required=True)
    args = parser.parse_args()
    # Define file paths
    evaluation_dir = Path(__file__).resolve().parent

    results_path = evaluation_dir / "results" / f"retrieval_{args.dataset}.json"
    output_path = (
        evaluation_dir / "results" / f"retrieval_threshold_{args.dataset}.json"
    )

    # Load benchmark results
    with results_path.open("r", encoding="utf-8") as file:
        benchmark_data = json.load(file)

    results = benchmark_data["results"]

    # Define candidate thresholds
    thresholds = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6] if args.dataset == "dev" else [0.3]

    threshold_results = []

    for threshold in thresholds:
        # Calculate confusion matrix
        metrics = evaluate_threshold(results, threshold)

        tp = metrics["tp"]
        tn = metrics["tn"]
        fp = metrics["fp"]
        fn = metrics["fn"]

        # Calculate total positive and negative cases
        n_answerable = tp + fn
        n_unsupported = tn + fp

        if n_answerable == 0 or n_unsupported == 0:
            raise ValueError(
                "Threshold evaluation requires both answerable "
                "and unsupported queries."
            )

        # Calculate classification metrics
        metrics["sensitivity"] = tp / n_answerable
        metrics["specificity"] = tn / n_unsupported
        metrics["false_acceptance_rate"] = fp / n_unsupported
        metrics["false_rejection_rate"] = fn / n_answerable

        threshold_results.append(metrics)

    # Build the evaluation report
    report = {
        "model_name": benchmark_data["model_name"],
        "dataset": benchmark_data["dataset"],
        "decision_rule": "Accept if top-1 score >= threshold",
        "n_queries": len(results),
        "threshold_results": threshold_results,
    }

    # Save results
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    # Display results
    print("\nRetrieval Threshold Evaluation")
    print(f"Model: {report['model_name']}")
    print(f"Dataset: {report['dataset']}")

    print(
        f"\n{'Threshold':>10} "
        f"{'TP':>5} {'FN':>5} {'TN':>5} {'FP':>5} "
        f"{'TPR':>8} {'TNR':>8} {'FPR':>8}"
    )

    for metrics in threshold_results:
        print(
            f"{metrics['threshold']:>10.2f} "
            f"{metrics['tp']:>5} "
            f"{metrics['fn']:>5} "
            f"{metrics['tn']:>5} "
            f"{metrics['fp']:>5} "
            f"{metrics['sensitivity']:>8.3f} "
            f"{metrics['specificity']:>8.3f} "
            f"{metrics['false_acceptance_rate']:>8.3f}"
        )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
