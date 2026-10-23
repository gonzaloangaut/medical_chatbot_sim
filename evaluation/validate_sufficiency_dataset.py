"""Validate structural consistency and frozen retrieval provenance for a sufficiency dataset.

Usage from repository root::

    python -m evaluation.validate_sufficiency_dataset

This checks references and label logic. It CANNOT verify that a passage truly
supports a requirement: semantic annotations need independent manual review.
"""

import argparse
import json
import hashlib
import math
import re
from collections import Counter
from pathlib import Path

EVALUATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALUATION_DIR.parent

LABELS = {"SUFFICIENT", "PARTIAL", "INSUFFICIENT"}
REQUIREMENT_LABELS = {"SUPPORTED", "UNSUPPORTED"}
EVIDENCE_ORIGINS = {"saved_dev_retrieval", "controlled"}


class DatasetValidationError(ValueError):
    """Raised for inconsistent annotations or source references."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetValidationError(message)


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def nonempty_string(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def unique_lookup(rows: list, key: str, source: str) -> dict:
    require(isinstance(rows, list), f"{source} must be a list.")
    result = {}
    for row in rows:
        require(isinstance(row, dict), f"{source} must contain objects.")
        identifier = row.get(key)
        require(nonempty_string(identifier), f"{source} has an invalid {key}.")
        require(identifier not in result, f"Duplicate {key}={identifier!r} in {source}.")
        result[identifier] = row
    return result


def strings_without_duplicates(values, context: str) -> None:
    require(isinstance(values, list), f"{context} must be a list.")
    require(all(nonempty_string(value) for value in values), f"{context} contains an invalid ID.")
    require(len(values) == len(set(values)), f"{context} contains duplicate IDs.")


def validate(dataset: dict, catalog: list, dev_queries: list, rankings: dict) -> Counter:
    require(dataset.get("dataset_id") == "sufficiency_dev_v1", "Unexpected dataset_id.")
    require(dataset.get("annotation_status") in {"draft_pending_manual_review", "reviewed_by_project_author"}, "Unknown annotation review status.")

    config = dataset.get("retrieval_configuration")
    require(isinstance(config, dict), "Missing retrieval configuration.")
    require(config.get("top_k") == 5, "Expected frozen top_k=5.")
    threshold = config.get("top1_acceptance_threshold")
    require(type(threshold) in (int, float) and float(threshold) == 0.30, "Expected frozen threshold=0.30.")

    require(isinstance(rankings, dict) and rankings.get("top_k") == 5, "Saved rankings must be Top-5.")
    chunks = unique_lookup(catalog, "chunk_id", "chunk catalog")
    dev = unique_lookup(dev_queries, "query_id", "development queries")
    saved = unique_lookup(rankings.get("results"), "query_id", "saved development rankings")
    cases = unique_lookup(dataset.get("cases"), "case_id", "sufficiency cases")

    require(len(cases) == 30, f"Expected 30 diagnostic cases, got {len(cases)}.")
    counts = Counter()
    origins = Counter()
    for case_id, case in cases.items():
        prefix = f"{case_id}: "
        require(re.fullmatch(r"suff_dev_\d{3}", case_id) is not None, prefix + "invalid case_id format.")
        require(nonempty_string(case.get("query")), prefix + "empty query.")
        require(nonempty_string(case.get("diagnostic_category")), prefix + "missing diagnostic category.")
        require(nonempty_string(case.get("annotation_rationale")), prefix + "missing annotation rationale.")

        label = case.get("expected_sufficiency")
        require(label in LABELS, prefix + "unknown sufficiency label.")
        origin = case.get("evidence_origin")
        require(origin in EVIDENCE_ORIGINS, prefix + "unknown evidence origin.")

        evidence_ids = case.get("evidence_chunk_ids")
        strings_without_duplicates(evidence_ids, prefix + "evidence_chunk_ids")
        require(set(evidence_ids).issubset(chunks), prefix + "unknown evidence chunk ID(s).")

        source_query_id = case.get("source_query_id")
        if source_query_id is not None:
            require(source_query_id in dev, prefix + "unknown source_query_id.")
            require(case["query"] == dev[source_query_id]["query"], prefix + "query differs from development source.")
        if origin == "saved_dev_retrieval":
            require(nonempty_string(source_query_id), prefix + "frozen retrieval case must identify its source query.")
            require(source_query_id in saved, prefix + "source query missing in frozen rankings.")
            saved_row = saved[source_query_id]
            require(saved_row["query"] == case["query"], prefix + "frozen query mismatch.")
            ranked = saved_row["retrieved_chunks"]
            require(isinstance(ranked, list) and len(ranked) > 0, prefix + "missing ranked chunks.")
            scores = [row["score"] for row in ranked]
            require(all(type(s) in (int, float) and math.isfinite(s) for s in scores), prefix + "invalid retrieval scores.")
            require(all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), prefix + "unsorted saved retrieval scores.")
            expected_ids = (
                [row["chunk_id"] for row in ranked[:5]]
                if scores[0] >= threshold else []
            )
            require(evidence_ids == expected_ids, prefix + "evidence differs from frozen Top-5/gate output.")

        requirements = case.get("requirements")
        require(isinstance(requirements, list) and requirements, prefix + "missing information requirements.")
        seen_requirements = set()
        supported = 0
        for index, item in enumerate(requirements, start=1):
            require(isinstance(item, dict), prefix + "requirement must be an object.")
            req_id = item.get("requirement_id")
            require(req_id == f"r{index}", prefix + f"expected requirement_id r{index}.")
            require(req_id not in seen_requirements, prefix + "duplicate requirement ID.")
            seen_requirements.add(req_id)
            require(nonempty_string(item.get("description")), prefix + f"empty {req_id} description.")
            status = item.get("status")
            require(status in REQUIREMENT_LABELS, prefix + f"invalid {req_id} status.")
            supporting = item.get("supporting_chunk_ids")
            strings_without_duplicates(supporting, prefix + f"{req_id} supporting_chunk_ids")
            require(set(supporting).issubset(evidence_ids), prefix + f"{req_id} points outside provided evidence.")
            require(bool(supporting) == (status == "SUPPORTED"), prefix + f"{req_id} status/support mismatch.")
            supported += status == "SUPPORTED"

        expected_label = (
            "SUFFICIENT" if supported == len(requirements)
            else "INSUFFICIENT" if supported == 0
            else "PARTIAL"
        )
        require(label == expected_label, prefix + f"label inconsistent with requirements: expected {expected_label}.")
        counts[label] += 1
        origins[origin] += 1

    require(counts == Counter({label: 10 for label in LABELS}), f"Expected 10 cases per label; observed {dict(counts)}.")
    require(origins == Counter({"saved_dev_retrieval": 19, "controlled": 11}), f"Unexpected evidence-origin distribution: {dict(origins)}.")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=EVALUATION_DIR / "datasets" / "sufficiency_dev.json")
    parser.add_argument("--catalog", type=Path, default=EVALUATION_DIR / "datasets" / "chunk_catalog.json")
    parser.add_argument("--dev-queries", type=Path, default=EVALUATION_DIR / "datasets" / "retrieval_dev.json")
    parser.add_argument("--retrieval-results", type=Path, default=EVALUATION_DIR / "results" / "retrieval_dev.json")
    args = parser.parse_args()
    try:
        dataset = read_json(args.dataset)
        expected_hashes = dataset.get("reference_sha256", {})
        require(isinstance(expected_hashes, dict), "Missing source fingerprints.")
        for key, path in (
            ("catalog", args.catalog),
            ("development_queries", args.dev_queries),
            ("frozen_rankings", args.retrieval_results),
        ):
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
            require(observed == expected_hashes.get(key), f"Source file changed: {path}. Check dataset version and provenance.")
        counts = validate(
            dataset,
            read_json(args.catalog),
            read_json(args.dev_queries),
            read_json(args.retrieval_results),
        )
    except (OSError, json.JSONDecodeError, DatasetValidationError, KeyError, TypeError) as error:
        parser.exit(1, f"Validation failed: {error}\n")

    print("Sufficiency dataset: structural and frozen-retrieval checks passed.")
    print("Cases: 30 (" + ", ".join(f"{label}: {counts[label]}" for label in sorted(LABELS)) + ")")
    print("Evidence origin: 19 frozen development Top-5/gate, 11 controlled contexts.")
    print(f"Annotation status: {dataset['annotation_status']}")
    if dataset["annotation_status"] == "draft_pending_manual_review":
        print("IMPORTANT: Semantic support labels are draft annotations and still require manual review.")
    else:
        print("Structural validation does not independently verify semantic labels.")


if __name__ == "__main__":
    main()
