# Retrieval Benchmark v2 — Draft annotation and evaluation protocol

**Status:** Semantically reviewed candidate annotation set; **not yet frozen or independently double-annotated**. Human semantic review required before evaluation. The corpus is educational/synthetic and **not suitable for clinical use**.

## Scope and version

- Corpus: the 15-document Knowledge Base v2; 54 section-based chunks in `evaluation/chunk_catalog_v2.json`.
- Starting repository commit: `38142c166f7e4fc10100833f3eb53fd6e50dd2e0` (`Add pilot retrieval benchmark and evaluation summary`).
- Existing pilot: `q_001`–`q_015`, included **unchanged** in development.
- Additional candidate questions: `q_016`–`q_120`, created from the source text *without running the retriever on the new questions*.
- Primary goal: assess **chunk-level evidence retrieval**, and later separately evaluate retrieval acceptance/rejection of unsupported queries.

## Dataset design

| Category | Development | Final test | Total |
| --- | ---: | ---: | ---: |
| `single_evidence` | 35 | 35 | 70 |
| `multi_evidence` | 10 | 10 | 20 |
| `in_domain_unsupported` | 8 | 7 | 15 |
| `out_of_domain` | 7 | 8 | 15 |
| **Total** | **60** | **60** | **120** |

- `retrieval_dataset_dev_v2_reviewed.json` contains 60 questions; use for debugging or any calibration.
- `retrieval_dataset_test_v2_reviewed.json` contains 60 **held-out** questions; do not use their retrieval outputs, scores, or metrics until all design choices are frozen.
- Both files use the same five required keys as the pilot: `query_id`, `query`, `category`, `answerable`, `relevant_chunk_ids`. They can be passed to the existing `validate_dataset()` function, independently, without changing its schema.
- Questions are mostly in Spanish, while the source documents are in English. This examines multilingual retrieval, not a representative estimate of real user traffic.
- The split is **not document-disjoint or fully clinical-intent-disjoint**. Some sections and subquestions appear in both sets; e.g. some multi-evidence test queries reuse components present in development. Thus test is held-out at the *question* level, **not** the document, fact, or clinical-intent level. Avoid claims that test estimates real-world generalization.

## Annotation instructions

1. Read the full question, then identify the specific facts it requests.
2. Read the KB **without viewing retriever predictions**. Find the smallest set of chunks sufficient to provide those facts.
3. Use `single_evidence` and one ID when a single chunk suffices; `multi_evidence` and two or more IDs when all are needed.
4. Label `in_domain_unsupported` when the request concerns the clinic or medical domain but the KB lacks the requested facts (e.g., a personal lab result, live availability, price, exact medical dose). Use `out_of_domain` for unrelated requests. Both use `answerable=false` and `relevant_chunk_ids=[]`.
5. `answerable` means **the requested factual information can be provided from these documents**, not that a patient can receive a diagnosis or individualized treatment recommendation. A refusal or safety redirection is not a substitute for missing factual evidence.
6. If another chunk or a different combination of chunks is equally sufficient, **flag the annotation for review**. The current JSON encodes just **one required evidence set**; it cannot represent alternative valid sets. Rewrite ambiguous questions or extend the annotation model before trusting Complete Evidence@k.
7. If multiple chunks are merely on topic but do not together support the full requested facts, the question is unsupported, not multi-evidence.
8. Treat the clinic policies as **fictional**; questions about its service targets must not be interpreted as real-world medical or legal facts.
9. If manual review changes a query, document the change *before running retrieval*. Do not adjust gold labels to match model predictions.

## Quality gates before benchmarking

- Run the existing `validate_dataset()` independently on both candidate JSON files against the v2 catalog. Verify exactly 60 queries each and no `query_id` overlap.
- Review **every** `multi_evidence` annotation to confirm all IDs are necessary, not just related.
- Spot-check every unsupported label against the full KB; validate that no relevant factual answer is present elsewhere.
- Check that each supported query can be answered fully and unambiguously using its assigned evidence.
- Check paraphrase/intent overlap across development and test; if too similar, rewrite or reassign **before the test run**.
- Freeze corpus content as well as IDs: ID equality alone will not reveal changed section text. Record code commit and exact embedding model revision in experimental metadata if feasible.

## Evaluation protocol

1. Keep the 15-question pilot as exploratory development evidence; do **not** quote its accuracy as the final result.
2. On development only, run `SemanticRetriever.search(..., top_k=5)`, compute macro Hit@1, Recall@1/3/5, Complete Evidence@1/3/5, and MRR@5 using answerable questions. Break down by category.
3. For acceptance/rejection, compare Top-1 similarity score distributions across answerable and unsupported development questions. At the candidate threshold(s), report supported acceptance, unsupported rejection, false acceptance and false rejection. Avoid treating cosine similarity as a calibrated probability.
4. Choose and **freeze** the retrieval Top-k, threshold, any model/chunking changes, and the scoring/annotation protocol using development only.
5. Evaluate test **once** after freezing the configuration. Report the same ranking metrics, acceptance metrics, and error examples. If the test reveals weaknesses, treat them as findings and create a *new* holdout for any subsequent tuning rather than reusing test as development.
6. The benchmark evaluates retrieval **only**, not completeness or clinical safety of generated chatbot responses. End-to-end evaluation is a later class.

## Review priorities

- `q_009`, `q_010`, `q_011` and the new multi-evidence questions: do alternative single chunks or chunk combinations already fully answer the question?
- In-domain unsupported questions about personal data: the KB contains only general knowledge and fictional policies, not personal health records.
- Questions with overlapping concepts (fever vs heatstroke, RSV vs influenza, sore throat vs strep): check sufficiency rather than mere topical relevance.
- The development/test split is deliberately balanced by category, **not** a prevalence-weighted estimate of real-world traffic. The small 15-document corpus limits true independence between questions.

## Next implementation task

After annotation review, update `run_retrieval_benchmark.py` to accept a dataset path (and output path) rather than hardcoding the pilot; first run only the development dataset. The existing `summarize_retrieval_results.py` can likewise accept results paths later. Keep `search()` distinct from `retrieve()` to measure ranking separately from the acceptance policy.

## Semantic audit and corrections (20 October 2026)

The 60 test entries were inspected against the 54 KB chunks, checking factual sufficiency, category, and possible alternative evidence. The 60 development questions were checked for cross-split overlap; the earlier Q043–Q049 review was also revisited. Lexical similarities using word/character TF-IDF were used **only to flag candidate overlaps**; the final decision was manual semantic comparison, without looking at retrieval predictions for the new questions. The corpus is small enough that genuine complete intent-disjointness would be unrealistic.

**Revised annotation files:** `retrieval_dataset_dev_v2_reviewed.json` and `retrieval_dataset_test_v2_reviewed.json`.

| Query | Modification | Reason |
| --- | --- | --- |
| Q046 (dev) | Add `medlineplus_diarrhea_003` and ask about fluid replacement | Include one 3-chunk evidence task in development |
| Q069 | Change wording to 'help evaluate' instead of implying CBC diagnoses by itself | Avoid unsupported medical claim |
| Q096 | Explicitly caution that overlapping allergy/cold symptoms do not diagnose the cause | Clarify evidence scope |
| Q097 | Compare influenza and RSV prevention, replacing the common-cold vaccine component | Reduce substantial overlap with development Q022 |
| Q098 | Specify that the clinic's turnaround starts after receipt of the finalized report | More faithful to fictional policy |
| Q099 | Add dehydration causes, signs in infants, and severe-care options; 3 required chunks | Include one 3-chunk task in test |
| Q101 | Compare fluids in RSV care vs diarrhea treatment | Replace composite of development Q031 and Q037 |
| Q102 | Ask for signs described for each condition, not signs 'common to both' | Remove ambiguity |
| Q105 | Ask about testing indications and follow-up after a negative rapid test | Avoid repeating development Q042's meaning of a positive test |
| Q112 | Ask for last week's confirmed RSV case count in Argentina | Replace paraphrase of development Q013's personal electrolyte result |

**Structural checks after edits:** 60 queries per split; 120 distinct IDs; category quotas unchanged; all supported labels have evidence and unsupported labels have none; 10 multi-evidence cases per split, including one 3-chunk question per split. Changed chunk IDs were matched to actual KB v2 section labels.

**Remaining limits:** The same medical topics/chunks recur across splits, and some multi-evidence test cases contain individual information needs previously seen in development. This is a held-out question benchmark, not an independent clinical-intent benchmark. Many questions are directly derived from the KB rather than actual patient utterances. Alternative valid evidence sets may still be missed; a second independent annotation pass would increase confidence. No embedding model was run on the new candidate test questions during this review.

**Do not tune on test.** Once the annotated datasets are frozen, develop settings solely using the development split. Record the final corpus/content fingerprint and embedding model revision, then evaluate test once.
