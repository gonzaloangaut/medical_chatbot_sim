# Grounded Generation — Exploratory Evaluation

**Project:** Medical Chatbot Simulation
**Status:** Exploratory development diagnostics, 22 October 2026
**Scope:** Generation behavior only; **not** clinical validation or a held-out end-to-end evaluation.

## 1. Questions

1. Does the revised grounded prompt/context format improve answers over the historical service format?
2. Does selecting only annotated supporting evidence (`oracle`) improve responses compared with the retrieved Top-5?
3. Does Qwen2.5-1.5B-Instruct address generation failures observed with Qwen2.5-0.5B-Instruct?
4. Can the models abstain or answer partially when relevant material does not support the requested fact?

## 2. Experimental setup

- **Models:** `Qwen/Qwen2.5-0.5B-Instruct` and `Qwen/Qwen2.5-1.5B-Instruct`, run locally on CPU.
- **Decoding:** `do_sample=False`, `max_new_tokens=256`; greedy decoding for both variants. This changes the original application's sampling behavior for experimental control.
- **Inputs:** Spanish questions and English section-level chunks from the fixed 15-document, 54-chunk knowledge base.
- **Retrieval:** Existing development Top-5 rankings from `results/retrieval_dev.json`, with the previously frozen Top-1 acceptance threshold of `0.30`. No embeddings were recomputed and the retriever was not retuned.
- **Prompt comparison:** Original service instructions/context vs. revised structured system prompt/context, holding evidence and decoding fixed. Both prompt instructions and context formatting changed; results cannot be attributed to system-prompt wording alone.
- **Oracle diagnostic:** Same grounded service and model with either Top-5 or only the gold annotated chunks (`relevant_chunk_ids`) for `q_001`, `q_047`, and `q_049`, preserving their original relative retrieval order. Oracle chunks are **not available to the deployed system**.
- **Unsupported/partial pilot:** `q_052` uses stored development retrieval; `pilot_partial_001` uses one manually selected fictional clinic-policy chunk and **does not measure retrieval**. The 1.5B follow-up covers these two cases, not the whole original six-case pilot.
- **No-context case:** `q_057` compares the old LLM-generated fallback with the new deterministic service fallback; this is a **service behavior** comparison, not a prompt-only comparison.

Saved result JSONs contain the queries, responses, chunk contents, and available provenance/configuration metadata. The oracle-pilot JSONs also record SHA-256 hashes for several input and code files. Model revisions and library versions were not fully pinned in these runs; future evaluations should record them.

## 3. Qualitative observations

| Case | Qwen 0.5B | Qwen 1.5B | Main limitation |
| --- | --- | --- | --- |
| `q_001`: Is 37 °C necessarily fever? | Top-5 answers incorrectly; oracle gets the central point but adds unsupported facts. | Top-5 gives the correct overall answer with a contradictory detail; oracle is substantially more faithful. | Extra context and generation fidelity. |
| `q_047`: Headache triggers **and** warning signs | Top-5 and oracle produce serious errors, with repetitive oracle output. | Top-5 mixes in dehydration signs; oracle identifies warning signs but **omits triggers**. | Evidence contamination and incomplete multi-part answers. |
| `q_049`: Appointment cancellation and record processing | Extracts `24` and `7` but loses **business days**, sometimes reverses the seven-day condition. | Preserves business days more often; top-5 adds unsupported delay explanations, oracle misstates certain details/translates *secure portal* incorrectly. | Conditions, qualifiers, and translation accuracy. |
| `q_052`: Patient-specific allergies absent from KB | Invents personal allergens in both baseline and grounded prompts. | Baseline invents personal allergens; grounded admits missing details **after** making an unsupported personal allergy claim. | In-domain unsupported question; false personal inference. |
| `pilot_partial_001`: Cancellation notice and unspecified late fee | Both prompts invent different exact monetary penalties. | Baseline invents a fee; grounded incorrectly **asserts that no fee applies**. | Missing information is not evidence that a fee exists or does not exist. |
| `q_057`: No relevant retrieval | Old fallback invokes LLM and produces a wrong unrelated capital. | Not rerun for 1.5B. | New service fallback is deterministic and does not invoke an LLM. |

### Important examples

- `q_001`: The evidence says a reading of 37 °C **by itself does not establish fever**. A response claiming the patient has a fever contradicts the source.
- `q_047`: Both gold chunks are in Top-5, at ranks 3 and 4. Oracle improves Qwen 1.5B's warning signs but fails to answer the triggers half of the question.
- `q_049`: The policy aims to process a **complete, verified request within seven business days**. Saying **at least seven days** changes the intended direction of the condition.
- `q_052`: General allergen examples cannot determine an individual patient's allergies; no personal medical record is provided.
- `pilot_partial_001`: The fictional policy defines **no fee schedule**; neither `a late fee is charged` nor `no late fee is charged` follows from that document. The supported partial answer is: cancellation requested at least **24 hours** in advance; fee information is unspecified.

## 4. Interpretation

1. The structured prompt and explicit grounding instructions are not sufficient to guarantee factual grounding, even with correct evidence available.
2. Restricting context to annotated oracle chunks improves parts of some answers but does not eliminate hallucination, missing sub-answers, or distorted conditions. Therefore, better retrieval/context selection alone is not a demonstrated solution.
3. Qwen 1.5B appears more capable than Qwen 0.5B on these development examples, particularly with oracle context. However, it still produces unsupported and incomplete claims. No general performance advantage or deployment readiness is established by such a small, selected pilot.
4. The deterministic empty-context fallback is a reliable **programmatic** improvement; it must be assessed separately from prompt quality.
5. Source metadata in API output records retrieved context **provenance**, not verified claim-level supporting citations.

## 5. Limitations and follow-up

- Cases were selected/inspected as **development diagnostics**, not sampled to estimate generalization. **No accuracy percentage, hypothesis test, or confidence interval** is warranted.
- The retrieval held-out test split was **not used** for this generation study. A future end-to-end evaluation will need a new independent set with answer-level reference expectations and claim/evidence assessment.
- Manual oracle selection and the fictional-clinic partial case are diagnostic setups, not deployable retrieval functionality.
- There was no systematic latency, memory, or throughput comparison for 0.5B versus 1.5B. Neither model is clinically validated.
- **Next engineering priority:** investigate evidence-sufficiency decisions and abstention; evaluate unsupported and partially supported questions without confusing semantic similarity with answerability.
- **Later evaluation:** design a separate end-to-end evaluation, then compare candidate generators/context selection approaches under a fixed, recorded protocol.

## 6. Artifacts

Scripts: `evaluation/run_grounding_pilot.py`, `evaluation/run_oracle_context_pilot.py`.

Suggested committed output location: `evaluation/results/generation/`:

- `grounding_pilot_qwen0_5b_dev.json`: original six-case old-vs-new pilot.
- `oracle_context_pilot_qwen0_5b_dev.json`: Top-5 vs. oracle on three cases.
- `oracle_context_pilot_qwen1_5b_dev.json`: same diagnostic on the larger model.
- `grounding_pilot_qwen1_5b_unsupported_dev.json`: old-vs-new comparison on `q_052` and the synthetic partial case.

The archived JSON files are retained **byte-for-byte**, including their original
metadata identifiers, to preserve the experimental record. New runs use
research-oriented study IDs.

### Reproduction

Run from the repository root with installed application dependencies. Results are
not expected to be byte-identical across different model revisions, library
versions, or hardware. Use new output filenames to preserve archived runs.

```bash
python -m evaluation.run_grounding_pilot \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --output evaluation/results/generation/reproduced_grounding_qwen0_5b_dev.json

python -m evaluation.run_oracle_context_pilot \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --output evaluation/results/generation/reproduced_oracle_qwen0_5b_dev.json

python -m evaluation.run_oracle_context_pilot \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --output evaluation/results/generation/reproduced_oracle_qwen1_5b_dev.json

python -m evaluation.run_grounding_pilot \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --case-ids q_052 \
  --output evaluation/results/generation/reproduced_grounding_qwen1_5b_unsupported_dev.json
```

`run_grounding_pilot.py` includes the synthetic partial-evidence example by
default; `--no-partial` skips it. Both runners reject an existing output path
unless `--overwrite` is explicitly supplied. Existing outputs are qualitative
development observations and **must not** be interpreted as held-out test scores.
