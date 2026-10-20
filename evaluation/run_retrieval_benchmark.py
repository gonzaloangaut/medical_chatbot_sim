import json
from pathlib import Path

from app.chatbot.chunking import chunk_documents
from app.chatbot.knowledge import load_documents
from app.chatbot.retrieval import SemanticRetriever
from evaluation.validate_retrieval_dataset import validate_dataset


def main():
    # Define evaluation file paths
    evaluation_dir = Path(__file__).resolve().parent

    catalog_path = evaluation_dir / "chunk_catalog_v2.json"
    dataset_path = evaluation_dir / "retrieval_dataset_pilot_v2.json"

    # Load evaluation data
    with catalog_path.open("r", encoding="utf-8") as file:
        catalog = json.load(file)

    with dataset_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    # Validate the evaluation dataset
    validate_dataset(dataset, catalog)

    # Load documents and generate chunks
    documents = load_documents()
    chunks = chunk_documents(documents)

    # Verify that the catalog matches the current knowledge base
    catalog_ids = {item["chunk_id"] for item in catalog}
    current_ids = {chunk.chunk_id for chunk in chunks}

    if catalog_ids != current_ids:
        raise ValueError("Chunk catalog does not match the knowledge base.")

    # Initialize and populate the retriever
    retriever = SemanticRetriever()
    retriever.ingest_chunks(chunks)

    # Run retrieval for all evaluation queries
    benchmark_results = []

    for item in dataset:
        query = item["query"]

        # Retrieve the top-5 most similar chunks
        results = retriever.search(query, top_k=5)

        # Store ranked results and similarity scores
        retrieved_chunks = []

        for rank, result in enumerate(results, start=1):
            retrieved_chunks.append(
                {
                    "rank": rank,
                    "chunk_id": result.chunk.chunk_id,
                    "score": result.score,
                }
            )

        # Store the query and its retrieval results
        benchmark_results.append(
            {
                "query_id": item["query_id"],
                "query": query,
                "category": item["category"],
                "answerable": item["answerable"],
                "relevant_chunk_ids": item["relevant_chunk_ids"],
                "retrieved_chunks": retrieved_chunks,
            }
        )

        print(f"Processed {item['query_id']}: {query}")

    # Define the output path
    output_path = evaluation_dir / "retrieval_results_pilot_v2.json"

    # Store benchmark metadata and results
    benchmark_data = {
        "model_name": retriever.model_name,
        "top_k": 5,
        "dataset": dataset_path.name,
        "results": benchmark_results,
    }

    # Save benchmark results
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            benchmark_data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nBenchmark completed: {len(benchmark_results)} queries.")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
