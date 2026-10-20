import json
from pathlib import Path


def validate_dataset(dataset: list[dict], catalog: list[dict]) -> None:
    """
    Validate the retrieval evaluation dataset against the chunk catalog.

    Raises
    ------
    ValueError
        If the dataset contains invalid annotations.
    """

    # Validate that the dataset and catalog are not empty
    if not dataset:
        raise ValueError("The evaluation dataset is empty.")

    if not catalog:
        raise ValueError("The chunk catalog is empty.")

    # Get the available chunk IDs
    chunk_ids = {chunk["chunk_id"] for chunk in catalog}

    if len(chunk_ids) != len(catalog):
        raise ValueError("The chunk catalog contains duplicate IDs.")

    # Define the allowed query categories
    allowed_categories = {
        "single_evidence",
        "multi_evidence",
        "in_domain_unsupported",
        "out_of_domain",
    }

    required_fields = {
        "query_id",
        "query",
        "category",
        "answerable",
        "relevant_chunk_ids",
    }

    # Keep track of previously seen query IDs
    seen_query_ids = set()

    # Validate each query
    for item in dataset:

        # Check required fields
        missing_fields = required_fields - item.keys()

        if missing_fields:
            raise ValueError(f"Missing fields: {sorted(missing_fields)}")

        query_id = item["query_id"]
        category = item["category"]
        answerable = item["answerable"]
        relevant_ids = item["relevant_chunk_ids"]

        # Check query ID uniqueness
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError("Query IDs must be non-empty strings.")

        if query_id in seen_query_ids:
            raise ValueError(f"Duplicate query ID: {query_id}")

        seen_query_ids.add(query_id)

        # Check query text
        if not isinstance(item["query"], str) or not item["query"].strip():
            raise ValueError(f"{query_id}: Query text is empty or invalid.")

        # Check category
        if category not in allowed_categories:
            raise ValueError(f"{query_id}: Invalid category '{category}'.")

        # Check answerability type
        if not isinstance(answerable, bool):
            raise TypeError(f"{query_id}: 'answerable' must be a boolean.")

        # Check relevant chunk IDs
        if not isinstance(relevant_ids, list) or not all(
            isinstance(chunk_id, str) for chunk_id in relevant_ids
        ):
            raise ValueError(
                f"{query_id}: 'relevant_chunk_ids' must be a list of strings."
            )

        if len(relevant_ids) != len(set(relevant_ids)):
            raise ValueError(f"{query_id}: Duplicate relevant chunk IDs.")

        # Check that referenced chunks exist
        invalid_ids = set(relevant_ids) - chunk_ids

        if invalid_ids:
            raise ValueError(f"{query_id}: Unknown chunks: {sorted(invalid_ids)}")

        # Validate single-evidence queries
        if category == "single_evidence":
            if not answerable or len(relevant_ids) != 1:
                raise ValueError(
                    f"{query_id}: Single-evidence queries require "
                    "exactly one relevant chunk."
                )

        # Validate multi-evidence queries
        elif category == "multi_evidence":
            if not answerable or len(relevant_ids) < 2:
                raise ValueError(
                    f"{query_id}: Multi-evidence queries require "
                    "at least two relevant chunks."
                )

        # Validate unsupported queries
        else:
            if answerable or relevant_ids:
                raise ValueError(
                    f"{query_id}: Unsupported queries must have "
                    "answerable=False and no relevant chunks."
                )


def main():
    # Define input file paths
    evaluation_dir = Path(__file__).resolve().parent

    catalog_path = evaluation_dir / "chunk_catalog_v2.json"
    dataset_path = evaluation_dir / "retrieval_dataset_pilot_v2.json"

    # Load the chunk catalog
    with catalog_path.open("r", encoding="utf-8") as file:
        catalog = json.load(file)

    # Load the evaluation dataset
    with dataset_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    # Validate the dataset against the catalog
    validate_dataset(dataset, catalog)

    # Print validation results
    print("Dataset validation passed.")
    print(f"Catalog chunks: {len(catalog)}")
    print(f"Queries: {len(dataset)}")


if __name__ == "__main__":
    main()
