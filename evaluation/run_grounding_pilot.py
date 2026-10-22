"""Compare historical and structured RAG generation using frozen development rankings.

Run ``python -m evaluation.run_grounding_pilot --help`` from the repository root.
Neither embeddings nor retrieval are recomputed; this is an exploratory
development diagnostic, not a held-out or clinical evaluation.
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
CATALOG_PATH = EVALUATION_DIR / "datasets" / "chunk_catalog.json"
DEV_PATH = EVALUATION_DIR / "datasets" / "retrieval_dev.json"
RANKINGS_PATH = EVALUATION_DIR / "results" / "retrieval_dev.json"
DEFAULT_OUTPUT_PATH = (
    EVALUATION_DIR / "results" / "generation" / "grounding_pilot_dev.json"
)
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
RETRIEVAL_THRESHOLD = 0.30  # Frozen retrieval baseline; not retuned here.

# Previously inspected development examples; NOT an independent test set.
DEFAULT_CASE_IDS = ("q_001", "q_047", "q_049", "q_052", "q_057")

PARTIAL_CASE = {
    "query_id": "pilot_partial_001",
    "query": (
        "En Demo Health Clinic, ¿con cuánta anticipación debo cancelar "
        "un turno y cuánto me cobran si cancelo tarde?"
    ),
    "category": "synthetic_partial_evidence",
    "answerable": False,  # Not fully answerable as asked.
    "relevant_chunk_ids": ["demo_clinic_appointment_policy_001"],
}

# Historical service prompts before the structured-grounding refactor.
BASELINE_SYSTEM_PROMPT = "Eres un asistente médico útil y preciso."
BASELINE_NO_CONTEXT = (
    "No hay información específica en la base de datos."
    "Respuesta obligatoria: Lo siento, no hay información al respecto."
)


class FixedRetriever:
    """Return frozen evidence for exactly one expected question."""

    def __init__(self, query: str, results: list[RetrievedChunk]):
        self.query = query
        self.results = results

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        if query != self.query:
            raise ValueError("FixedRetriever received an unexpected query.")
        return self.results


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_baseline_context(results: list[RetrievedChunk]) -> str:
    """Reproduce the historical context format (without chunk IDs)."""
    blocks = []
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        blocks.append(
            f"[Context {index}]\n"
            f"Title: {chunk.title}\n"
            f"Section: {chunk.section}\n"
            f"Source: {chunk.source}\n"
            f"Content:\n"
            f"{chunk.content}"
        )
    return "\n\n".join(blocks)


def build_baseline_messages(
    query: str, results: list[RetrievedChunk]
) -> list[dict[str, str]]:
    """Reproduce the old service's prompting, including its empty-context path."""
    if results:
        context_to_use = build_baseline_context(results)
    else:
        context_to_use = BASELINE_NO_CONTEXT

    # Preserve the original formatting of the historical triple-quoted prompt.
    full_prompt = f"""
        Instrucciones: Eres un asistente médico.
        Responde DIRECTAMENTE al paciente (usa "usted") basándote SOLO en el 
        TEXTO OFICIAL.
        Usa oraciones completas.
        Sé empático con el paciente.

        TEXTO OFICIAL:
        {context_to_use}

        ---
        PREGUNTA DEL PACIENTE: {query}
        
        TU RESPUESTA AL PACIENTE:
        """

    return [
        {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
        {"role": "user", "content": full_prompt},
    ]


def frozen_results(
    case: dict,
    rank_record: dict,
    chunk_by_id: dict[str, Chunk],
) -> list[RetrievedChunk]:
    """Apply the already frozen Top-1 gate to previously saved Top-5 rankings."""
    if case["query"] != rank_record["query"]:
        raise ValueError(f"Query mismatch for {case['query_id']}.")

    ranked = rank_record["retrieved_chunks"]
    if not ranked or ranked[0]["score"] < RETRIEVAL_THRESHOLD:
        return []

    return [
        RetrievedChunk(
            chunk=chunk_by_id[item["chunk_id"]],
            score=float(item["score"]),
        )
        for item in ranked[:5]
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--case-ids",
        nargs="+",
        default=list(DEFAULT_CASE_IDS),
        help="Development query IDs to run (the synthetic partial case is separate).",
    )
    parser.add_argument(
        "--no-partial",
        action="store_true",
        help="Omit the manually selected partial-evidence case.",
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


def save_output(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    output_path = args.output
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. "
            "Choose --output or pass --overwrite explicitly."
        )

    catalog = load_json(CATALOG_PATH)
    dev_records = load_json(DEV_PATH)
    ranked_data = load_json(RANKINGS_PATH)

    if ranked_data["top_k"] != 5:
        raise ValueError("Saved retrieval results are not Top-5.")

    dev_by_id = {item["query_id"]: item for item in dev_records}
    rank_by_id = {item["query_id"]: item for item in ranked_data["results"]}
    chunk_by_id = {
        item["chunk_id"]: Chunk(**item)
        for item in catalog
    }

    cases = []
    for query_id in args.case_ids:
        case = dev_by_id[query_id]
        results = frozen_results(case, rank_by_id[query_id], chunk_by_id)
        cases.append((case, results, "saved_dev_retrieval"))

    # The synthetic partial-answer case is diagnostic, not a retrieval result.
    if not args.no_partial:
        partial_results = [
            RetrievedChunk(
                chunk=chunk_by_id["demo_clinic_appointment_policy_001"],
                score=0.0,  # Placeholder only; no similarity score was computed.
            )
        ]
        cases.append((PARTIAL_CASE, partial_results, "manually_selected_evidence"))

    # One model instance for both variants, with identical greedy decoding.
    llm = QwenLLM(model_name=args.model_name, do_sample=False)

    output = {
        "study": "grounded_generation_prompt_comparison",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "model_name": args.model_name,
        "decoding": {"do_sample": False, "max_new_tokens": 256},
        "retrieval": {
            "source": str(RANKINGS_PATH.relative_to(EVALUATION_DIR)),
            "catalog": str(CATALOG_PATH.relative_to(EVALUATION_DIR)),
            "top_k": 5,
            "frozen_top1_threshold": RETRIEVAL_THRESHOLD,
            "catalog_sha256": hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest(),
            "notes": "No embedding inference or retrieval reruns performed.",
        },
        "comparison": (
            "Old service prompt/context versus new service prompt/context. "
            "If retrieval is empty, the old LLM fallback is compared with the "
            "deterministic service fallback; that case is NOT prompt-only."
        ),
        "cases": [],
    }

    for case, results, evidence_origin in cases:
        print(f"\n=== {case['query_id']} ===", flush=True)
        print(case["query"], flush=True)

        # Historical baseline: LLM is invoked even when retrieval is empty.
        baseline_messages = build_baseline_messages(case["query"], results)
        baseline_response = llm.generate(baseline_messages)
        print(f"\nBaseline:\n{baseline_response}", flush=True)

        # Current service: its empty-context branch returns without invoking LLM.
        bot = MedicalAssistance(
            llm=llm,
            retriever=FixedRetriever(case["query"], results),
        )
        grounded_result = bot.generate_response(
            [{"role": "user", "content": case["query"]}]
        )
        print(f"\nGrounded:\n{grounded_result.response}", flush=True)

        output["cases"].append(
            {
                "query_id": case["query_id"],
                "query": case["query"],
                "category": case["category"],
                "answerable": case["answerable"],
                "gold_relevant_chunk_ids": case["relevant_chunk_ids"],
                "evidence_origin": evidence_origin,
                "evidence": [
                    {
                        "rank": rank,
                        "chunk_id": item.chunk.chunk_id,
                        "score": item.score if evidence_origin == "saved_dev_retrieval" else None,
                        "chunk": asdict(item.chunk),
                    }
                    for rank, item in enumerate(results, start=1)
                ],
                "baseline_messages": baseline_messages,
                "baseline_response": baseline_response,
                "grounded_response": grounded_result.response,
                "grounded_sources": [asdict(source) for source in grounded_result.sources],
                "grounded_llm_invoked": bool(results),
                "comparison_scope": (
                    "service_fallback" if not results else "generation_prompts"
                ),
            }
        )
        # Save each completed case to retain partial results if the run is interrupted.
        save_output(output, output_path)

    print(f"\nCompleted {len(output['cases'])} cases.")
    print(f"Saved to: {output_path}")
    print("Review faithfulness, completeness and abstention manually;")
    print("this pilot does not estimate generalization performance.")


if __name__ == "__main__":
    main()
