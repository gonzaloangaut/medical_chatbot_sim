"""Compare fixed Top-5 evidence with oracle evidence for three development queries.

Run ``python -m evaluation.run_oracle_context_pilot --help`` from the
repository root.

This is a qualitative diagnostic, NOT a held-out evaluation. It does not
recompute embeddings or rerun retrieval. Both conditions use the same
MedicalAssistance prompt, Qwen model, and greedy decoding. Oracle evidence
is selected using gold chunk IDs, which are unavailable in production.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from app.chatbot.llm import QwenLLM
from app.chatbot.models import Chunk, RetrievedChunk
from app.chatbot.service import MedicalAssistance


EVALUATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALUATION_DIR.parent
CATALOG_PATH = EVALUATION_DIR / "datasets" / "chunk_catalog.json"
DEV_PATH = EVALUATION_DIR / "datasets" / "retrieval_dev.json"
RANKINGS_PATH = EVALUATION_DIR / "results" / "retrieval_dev.json"
DEFAULT_OUTPUT_PATH = (
    EVALUATION_DIR / "results" / "generation" / "oracle_context_pilot_dev.json"
)
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_CASE_IDS = ("q_001", "q_047", "q_049")
TOP_K = 5
THRESHOLD = 0.30  # Frozen baseline; not tuned in this experiment.


class FixedRetriever:
    """Return a predetermined list of chunks for one specific question."""

    def __init__(self, query: str, results: list[RetrievedChunk]):
        self.query = query
        self.results = results

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        if query != self.query:
            raise ValueError("Unexpected query passed to FixedRetriever.")
        return self.results


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe_evidence(chunks: list[RetrievedChunk], original_ranks: dict) -> list[dict]:
    return [
        {
            "original_top5_rank": original_ranks[item.chunk.chunk_id],
            "score": item.score,
            "chunk": asdict(item.chunk),
        }
        for item in chunks
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--case-ids",
        nargs="+",
        default=list(DEFAULT_CASE_IDS),
        help="Answerable development query IDs with all gold evidence in Top-5.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="JSON output path (defaults to evaluation/results/generation/).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. "
            "Choose --output or pass --overwrite explicitly."
        )

    catalog = load_json(CATALOG_PATH)
    development = load_json(DEV_PATH)
    saved_retrieval = load_json(RANKINGS_PATH)

    if saved_retrieval["top_k"] != TOP_K:
        raise ValueError("Saved retrieval was not computed with Top-5.")

    chunk_by_id = {entry["chunk_id"]: Chunk(**entry) for entry in catalog}
    dev_by_id = {entry["query_id"]: entry for entry in development}
    ranked_by_id = {
        entry["query_id"]: entry for entry in saved_retrieval["results"]
    }

    prepared_cases = []
    for query_id in args.case_ids:
        case = dev_by_id[query_id]
        ranking = ranked_by_id[query_id]
        if case["query"] != ranking["query"]:
            raise ValueError(f"Query mismatch for {query_id}.")
        if not case["answerable"] or not case["relevant_chunk_ids"]:
            raise ValueError(f"No oracle evidence defined for {query_id}.")

        ranked_chunks = ranking["retrieved_chunks"][:TOP_K]
        if len(ranked_chunks) != TOP_K or ranked_chunks[0]["score"] < THRESHOLD:
            raise ValueError(f"Expected an accepted Top-5 ranking for {query_id}.")
        if len({entry["chunk_id"] for entry in ranked_chunks}) != TOP_K:
            raise ValueError(f"Duplicate retrieved chunks in {query_id}.")

        gold_ids = set(case["relevant_chunk_ids"])
        retrieved_ids = {entry["chunk_id"] for entry in ranked_chunks}
        if not gold_ids.issubset(retrieved_ids):
            raise ValueError(f"Gold evidence not fully present in Top-5: {query_id}.")

        top5 = [
            RetrievedChunk(
                chunk=chunk_by_id[entry["chunk_id"]],
                score=float(entry["score"]),
            )
            for entry in ranked_chunks
        ]
        # Keep the original retrieval order among the annotated gold chunks.
        oracle = [item for item in top5 if item.chunk.chunk_id in gold_ids]
        original_ranks = {
            entry["chunk_id"]: entry["rank"] for entry in ranked_chunks
        }
        prepared_cases.append((case, top5, oracle, original_ranks))

    # Load the model only once; both conditions use greedy decoding.
    llm = QwenLLM(model_name=args.model_name, do_sample=False)
    output = {
        "study": "oracle_evidence_diagnostic",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "model_name": args.model_name,
        "decoding": {"do_sample": False, "max_new_tokens": 256},
        "same_service_prompt_for_both_conditions": True,
        "retrieval": {
            "source": str(RANKINGS_PATH.relative_to(EVALUATION_DIR)),
            "top_k": TOP_K,
            "frozen_top1_threshold": THRESHOLD,
            "notes": "Read saved rankings; did not recompute embeddings or rerun retrieval.",
        },
        "sha256": {
            "catalog": sha256(CATALOG_PATH),
            "development_queries": sha256(DEV_PATH),
            "saved_retrieval": sha256(RANKINGS_PATH),
            "service_py": sha256(PROJECT_ROOT / "app" / "chatbot" / "service.py"),
            "llm_py": sha256(PROJECT_ROOT / "app" / "chatbot" / "llm.py"),
        },
        "cases": [],
    }

    for case, top5, oracle, original_ranks in prepared_cases:
        print(f"\n=== {case['query_id']} ===\n{case['query']}", flush=True)
        print(
            f"Top-5: {[x.chunk.chunk_id for x in top5]}\n"
            f"Oracle: {[x.chunk.chunk_id for x in oracle]}",
            flush=True,
        )

        responses = {}
        for condition, evidence in (("top5", top5), ("oracle", oracle)):
            bot = MedicalAssistance(
                llm=llm,
                retriever=FixedRetriever(case["query"], evidence),
            )
            result = bot.generate_response(
                [{"role": "user", "content": case["query"]}]
            )
            responses[condition] = {
                "response": result.response,
                "sources": [asdict(source) for source in result.sources],
                "evidence": describe_evidence(evidence, original_ranks),
            }
            print(f"\n{condition.upper()}:\n{result.response}", flush=True)

        output["cases"].append(
            {
                "query_id": case["query_id"],
                "query": case["query"],
                "category": case["category"],
                "gold_relevant_chunk_ids": case["relevant_chunk_ids"],
                "conditions": responses,
            }
        )
        # Preserve completed cases if the run is interrupted.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(output, file, ensure_ascii=False, indent=2)

    print(f"\nCompleted {len(output['cases'])} cases.")
    print(f"Saved results to: {output_path}")
    print("Inspect answers against evidence; do not treat this pilot as a test score.")


if __name__ == "__main__":
    main()
