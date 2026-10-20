# Retrieval benchmark — Protocol and final results

**Project:** Medical Chatbot Simulation  
**Status:** Completed held-out retrieval evaluation (20 October 2026)  
**Baseline frozen before test:** [`d29d9b917ab4b3943aab6ef1c82cb7d6f833a9c2`](https://github.com/gonzaloangaut/medical_chatbot_sim/commit/d29d9b917ab4b3943aab6ef1c82cb7d6f833a9c2)  
**Intended use:** Educational evaluation only; **not clinical validation**.

## 1. Research questions

This benchmark evaluates the **retrieval stage** of a small multilingual retrieval-augmented generation (RAG) prototype:

1. Can the retriever find the annotated supporting evidence for Spanish-language questions in an English-language knowledge base (KB)?
2. How does evidence coverage change from Top-1 to Top-3 to Top-5, especially for multi-evidence questions?
3. Can the highest cosine-similarity score reliably distinguish answerable questions from questions whose requested facts are absent from the KB?

This benchmark does **not** evaluate the accuracy, factual grounding, citation correctness, or clinical safety of generated answers.

## 2. Corpus and frozen configuration

| Component | Configuration |
| --- | --- |
| Corpus | 15 English-language Markdown documents, 54 section-based chunks |
| Content | Educational medical information adapted from MedlinePlus and three fictional clinic-policy documents |
| Chunking | One retrievable unit per non-empty `##` section |
| Embedding model | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (pretrained; no fine-tuning) |
| Chunk representation | `title + "\n\n" + section + "\n\n" + content` |
| Ranking | Cosine similarity through Sentence Transformers semantic search |
| Retrieval depth | **Top-5** |
| Initial acceptance gate | **Accept if Top-1 cosine similarity >= 0.30** |
| Query language | Primarily Spanish |
| Document language | English |
| Configuration selected on | Development split only |

`SemanticRetriever.search()` computes a ranked list **without a threshold**; this is what the ranking benchmark evaluates. `SemanticRetriever.retrieve()` separately applies the Top-1 similarity gate before supplying context to the chatbot. The similarity score is not a calibrated probability or a guarantee of answerability.

The corpus catalog is stored in `evaluation/datasets/chunk_catalog.json`. The benchmark runner checks the **full catalog records** against the chunks currently generated from the KB; changing source text without updating the catalog causes the run to fail.

## 3. Question sets and annotation

Two question-level splits contain **60 queries each (120 total)**:

| Category | Development | Held-out test |
| --- | ---: | ---: |
| `single_evidence` | 35 | 35 |
| `multi_evidence` | 10 | 10 |
| `in_domain_unsupported` | 8 | 7 |
| `out_of_domain` | 7 | 8 |
| **Total** | **60** | **60** |

Both splits have 45 answerable and 15 unsupported questions. One multi-evidence question per split requires **three** distinct chunks; the remaining multi-evidence examples require two. The category counts are deliberately balanced for diagnostic evaluation and do **not** represent natural user-query prevalence.

Each JSON record contains `query_id`, `query`, `category`, `answerable`, and `relevant_chunk_ids`.

**Gold annotation rules:**

- Identify the facts needed to answer the full question from the fixed KB, independently of model rankings.
- `single_evidence` requires one sufficient chunk; `multi_evidence` requires two or more annotated chunks to cover all requested information.
- For `in_domain_unsupported` and `out_of_domain`, set `answerable=false` and `relevant_chunk_ids=[]`.
- General information about a condition does **not** support claims about a particular patient's records, current clinic availability, specific prices, exact individualized dosing, or live surveillance figures.
- Clinic policies are fictional. They do not describe actual providers, legal requirements, or service guarantees.
- The JSON records **one required evidence set per question**. It does not represent all possible alternative sufficient evidence sets.

Annotations and cross-split overlap were manually reviewed before the held-out run. The set was **not independently double-annotated**. No gold labels were edited to fit test predictions.

**Split limitation:** The test questions are new, but both splits reuse KB documents, topics, facts, and some related information needs. This is **question-level holdout**, not document-disjoint, clinical-intent-disjoint, or real-patient evaluation.

## 4. Metrics and evaluation procedure

**Ranking metrics** are computed only over the 45 answerable queries per split:

- **Hit@1:** whether the first retrieved chunk is among the annotated relevant chunks.
- **Recall@k:** fraction of annotated relevant chunks among the top `k` results, averaged across queries (macro Recall@k).
- **Complete Evidence@k:** fraction of queries for which **all** annotated required chunks appear in Top-k.
- **MRR@5:** mean reciprocal rank of the **first** annotated relevant chunk within Top-5. For multi-evidence questions, MRR does not measure completeness.

**Retrieve-or-reject metrics** use all 60 queries and the Top-1 score. A query is a positive example if its requested facts are marked answerable. We report TP/FN/TN/FP, sensitivity (TPR), specificity (TNR), false acceptance rate (FPR), and false rejection rate (FNR). A retrieval-gate false positive is **not** a measured LLM hallucination.

**Procedure:**

1. Create and validate the chunk catalog and gold annotations from the fixed KB.
2. Evaluate Top-5 ranking on development. Analyze Complete Evidence@k, score distributions, errors, and several candidate thresholds on development only.
3. Freeze MiniLM, chunking/representation, Top-5, acceptance threshold **0.30**, and the evaluation rules; commit [`d29d9b9`](https://github.com/gonzaloangaut/medical_chatbot_sim/commit/d29d9b917ab4b3943aab6ef1c82cb7d6f833a9c2).
4. Evaluate held-out test without further tuning. Evaluate acceptance on test **only at 0.30**.
5. Report all observed failures. Do not select a new threshold, change the model, or revise test gold labels in response to test performance and then reuse this test as an independent estimate.

## 5. Ranking results

| Metric (45 answerable questions per split) | Development | Held-out test |
| --- | ---: | ---: |
| Hit@1 | 82.22% | 82.22% |
| Recall@1 | 74.07% | 71.85% |
| Recall@3 | 93.33% | 88.89% |
| **Recall@5** | **100.00%** | **95.56%** |
| Complete Evidence@1 | 66.67% | 62.22% |
| Complete Evidence@3 | 88.89% | 82.22% |
| **Complete Evidence@5** | **100.00% (45/45)** | **93.33% (42/45)** |
| MRR@5 | 0.8970 | 0.8907 |

### Complete evidence by question type

| Question type | Dev CE@3 | Test CE@3 | Dev CE@5 | Test CE@5 |
| --- | ---: | ---: | ---: | ---: |
| Single-evidence (n=35/split) | 97.14% | 94.29% | 100.00% | 97.14% |
| Multi-evidence (n=10/split) | 60.00% | 40.00% | 100.00% | 80.00% |

Top-5 was selected on development because it recovered complete annotated evidence for more questions than Top-3. The held-out test supports using Top-5 as a **retrieval-coverage baseline**, but confirms it is not perfect. Its effect on generated-answer quality and context distraction remains untested.

Three answerable test cases lack at least one annotated required chunk in Top-5:

- **Q069 — CBC monitoring:** `medlineplus_complete_blood_count_001` is absent, despite related CBC and diagnostic chunks being returned.
- **Q102 — Headache and heatstroke warnings:** the heatstroke section appears, but `medlineplus_headache_002` does not.
- **Q104 — Clinic report access and medical-record delivery:** the laboratory-access section appears, but `demo_clinic_medical_records_policy_002`, the annotated record-delivery section, does not.

These are errors **under the frozen gold annotations**. Other retrieved text may be topically related; the benchmark does not automatically treat topical relevance as evidence sufficient to answer every requested fact.

## 6. Similarity-gate results

At the **preselected threshold 0.30**, both splits produced the same aggregate confusion-matrix counts:

| Actual label | Predicted reject | Predicted accept |
| --- | ---: | ---: |
| Unsupported | TN = 6 | FP = 9 |
| Answerable | FN = 0 | TP = 45 |

| Gate metric | Development | Held-out test |
| --- | ---: | ---: |
| Sensitivity / TPR | 100.00% | 100.00% |
| Specificity / TNR | 40.00% | 40.00% |
| **False acceptance rate / FPR** | **60.00%** | **60.00%** |
| False rejection rate / FNR | 0.00% | 0.00% |

**Held-out unsupported categories:** all **7/7 in-domain unsupported** questions are incorrectly accepted, while **6/8 out-of-domain** questions are correctly rejected (2/8 accepted). The development split similarly accepted all 8/8 in-domain unsupported questions.

For example, the held-out Q108 asks for an exact late-cancellation charge. The retriever assigns high similarity to the cancellation-policy chunk, although that policy does not specify a monetary charge. **High semantic similarity is not evidence sufficiency.**

Development-only threshold experiments showed that increasing the threshold trades false acceptance for false rejection; no single threshold can perfectly separate these labeled questions. The value **0.30** is deliberately retained as a permissive **low-similarity filter**, not an independent safety gate.

## 7. Limitations and interpretation

- **Small and source-derived dataset:** 120 largely synthetic questions over 54 chunks; estimates are noisy, particularly for the 15 unsupported questions per split.
- **Question-level rather than topic-level holdout:** the test shares documents and some underlying facts with development. Do not infer performance on unseen clinical topics or real patients.
- **Single gold evidence set:** alternatives that are sufficient but unannotated can make chunk-ID metrics pessimistic. Annotation review and more flexible evidence evaluation are future work.
- **No independent second annotator:** a second blinded evidence audit would improve annotation reliability.
- **No calibrated confidence:** cosine similarity is used for ranking and low-relevance filtering, not as a probability of correctness or answerability.
- **No end-to-end validation:** these results measure retrieval and its initial acceptance gate, not the LLM's factual accuracy, abstention, groundedness, or user safety.
- **Reproducibility scope:** the repository commit and stored result JSONs identify the experimental baseline; the current outputs include the model name but **not an exact immutable model revision or full dependency lockfile**. The corpus-content check prevents a particular class of drift but does not establish full environment reproducibility.

This system is an **educational medical-information prototype**, not a medical device, diagnostic system, or clinical advisor.

## 8. Reproduce the evaluation

From the repository root, after installing `requirements.txt` (plus `pytest` for tests):

```bash
pytest -q

python -m evaluation.run_retrieval_benchmark --dataset dev
python -m evaluation.summarize_retrieval_results --dataset dev
python -m evaluation.evaluate_retrieval_threshold --dataset dev

# Reproduce the already-reported held-out results without retuning:
python -m evaluation.run_retrieval_benchmark --dataset test
python -m evaluation.summarize_retrieval_results --dataset test
python -m evaluation.evaluate_retrieval_threshold --dataset test
```

The threshold command evaluates **several candidate thresholds only for `dev`** and exactly **0.30 for `test`**. Development and test artifacts are stored in `evaluation/results/` and the frozen inputs in `evaluation/datasets/`.

See [`retrieval_analysis.ipynb`](retrieval_analysis.ipynb) for figures, failure cases and interpretation. That notebook analyzes stored results rather than recalculating embeddings.

**Next research question:** Can a separate, explicitly evaluated evidence-sufficiency and grounded-generation stage reduce unsupported claims without rejecting answerable requests unnecessarily?
