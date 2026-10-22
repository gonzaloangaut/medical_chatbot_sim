# Medical Chatbot Simulation

An **educational medical-information chatbot** built with FastAPI, a local language model, and a structured multilingual Retrieval-Augmented Generation (RAG) pipeline. The project focuses on modular software design, source provenance, reproducible evaluation, and understanding the difference between **retrieving related information** and **having sufficient evidence to answer**.

> **Medical disclaimer:** This is a research/portfolio prototype, **not a medical device**. It is not clinically validated and must not be used for diagnosis, prescribing, emergency triage, or treatment decisions. The demo clinic and its operational policies are fictional.

## What the project demonstrates

- **Structured knowledge ingestion:** English Markdown documents with source metadata, split into section-level chunks.
- **Cross-lingual semantic retrieval:** Spanish-language questions are matched to English-language reference documents using pretrained multilingual sentence embeddings.
- **Modular RAG pipeline:** knowledge loading, chunking, retrieval, prompt construction, generation, and API responses have separate responsibilities.
- **Programmatic source provenance:** API responses include metadata for the chunks provided as context, rather than relying on the LLM to invent citations. These source lists are **not claim-level citation verification**.
- **Evaluation-driven design:** a 120-query annotated benchmark with separate development and held-out test splits, ranking metrics, threshold analysis, and qualitative failure analysis.
- **Exploratory generator diagnostics:** controlled prompt, evidence-selection, and model-size comparisons, with preserved failure cases and an explicit limitation analysis.
- **Unit testing:** test doubles cover application behavior without needing to initialize the real embedding or generation models in every test.

## Architecture

```text
English Markdown knowledge base (15 documents)
                │
                ▼
         Document loader
                │
                ▼
      Section-based chunking
          (54 chunks)
                │
                ▼
        Multilingual MiniLM
       (chunk/query embeddings)
                │
                ▼
    Cosine-similarity retrieval
              Top-5
                │
                ▼
 Top-1 similarity gate (>= 0.30)
                │
                ▼
     Structured LLM context
                │
                ▼
    Local Qwen 2.5 0.5B Instruct
                │
                ▼
  FastAPI response + source list
```

The current similarity gate is **not an evidence-sufficiency detector**. A query can pass it even when the KB lacks the requested fact. Explicit evidence sufficiency, reliable abstention, and response-level groundedness are future evaluation targets.

### Knowledge base and provenance

The KB lives in `data/knowledge/` and contains educational medical summaries adapted from MedlinePlus alongside **three fictional clinic-policy documents**. Every document has front matter identifying its document ID, title, source, and source URL. Non-empty `##` sections become chunks that retain their parent metadata.

The fixed benchmark uses **15 documents and 54 chunks**. The chunk representation sent to the embedding model is:

```text
title + "\n\n" + section + "\n\n" + content
```

### Retrieval

Embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. This is a **pretrained model**; the project does not train or fine-tune it. The retriever ranks chunks by cosine similarity.

The baseline settings were selected on the development split and frozen before held-out evaluation:

- **Top-k:** 5
- **Top-1 acceptance threshold:** 0.30

`search()` provides a ranked list for evaluation. `retrieve()` applies the initial acceptance gate before downstream generation. The similarity score is not a calibrated confidence value.

### Generation and API

Generation defaults to the local `Qwen/Qwen2.5-0.5B-Instruct` model on CPU; `QwenLLM` also accepts an alternative model name. `MedicalAssistance` places the retrieved chunks in a structured evidence block and supplies grounding instructions to the model. When no chunks pass retrieval acceptance, the service returns a deterministic fallback without invoking the LLM. FastAPI exposes the response together with retrieved-context provenance.

Example response shape:

```json
{
  "response": "Generated answer",
  "sources": [
    {
      "title": "Fever",
      "section": "Overview",
      "source": "MedlinePlus - National Library of Medicine",
      "source_url": "https://medlineplus.gov/fever.html"
    }
  ]
}
```

**Current limitation:** The generator has not yet been rigorously evaluated for unsupported claims, evidence-based abstention, or clinical safety. A returned source list only establishes which sources were retrieved and provided, **not** that every claim in the answer is supported.

## Retrieval evaluation

The benchmark contains **120 manually reviewed, mostly synthetic Spanish-language queries** against the fixed English-language KB: 60 development and 60 held-out test questions. Each split contains 35 single-evidence, 10 multi-evidence, and 15 unsupported queries. The test configuration was frozen at commit [`d29d9b9`](https://github.com/gonzaloangaut/medical_chatbot_sim/commit/d29d9b917ab4b3943aab6ef1c82cb7d6f833a9c2) **before** inspecting test retrieval results.

### Ranking: 45 answerable queries per split

| Metric | Development | Held-out test |
| --- | ---: | ---: |
| Hit@1 | 82.22% | 82.22% |
| Recall@3 | 93.33% | 88.89% |
| **Recall@5** | **100.00%** | **95.56%** |
| Complete Evidence@3 | 88.89% | 82.22% |
| **Complete Evidence@5** | **100.00% (45/45)** | **93.33% (42/45)** |
| MRR@5 | 0.8970 | 0.8907 |

For the **10 multi-evidence queries** in each split, Complete Evidence@5 falls from **100% (development)** to **80% (test)**. Top-5 therefore improves evidence coverage over Top-3 but does **not** guarantee complete context.

### Acceptance gate: all 60 queries per split

With the fixed threshold `0.30`, the held-out test gate accepted all **45/45 answerable** queries but also incorrectly accepted **9/15 unsupported** queries (60% false acceptance rate). All **7/7 in-domain unsupported** test queries passed the gate; 6/8 out-of-domain queries were correctly rejected.

**Main finding:** Semantic similarity finds **related** text; it does not establish that the retrieved text contains the **specific evidence required** to answer a question. The next stage will investigate grounded generation, evidence sufficiency, and abstention as distinct measurable behaviors.

**Interpretation limits:** The KB is small; the questions are mostly curated from the source material rather than sampled from clinical interactions; and development/test share documents, topics, and some underlying facts. This is a held-out **question-level retrieval benchmark**, not proof of real-world multilingual robustness, diagnostic correctness, or patient safety. Only one gold evidence set is annotated per question.

**Explore the analysis:** [`evaluation/retrieval_analysis.ipynb`](evaluation/retrieval_analysis.ipynb)  
**Methodology and full results:** [`evaluation/RETRIEVAL_BENCHMARK_PROTOCOL.md`](evaluation/RETRIEVAL_BENCHMARK_PROTOCOL.md)

## Exploratory generation evaluation

A **development-only qualitative pilot** compares the historical and structured
RAG prompts, retrieved Top-5 versus annotated *oracle* evidence, and local
Qwen2.5 0.5B versus 1.5B. Both models used greedy decoding and frozen
retrieval results; the held-out retrieval test queries were **not** used.
The experiments are diagnostic examples, **not representative accuracy scores**.

The larger model generally handled some supported facts better, but still
omitted parts of multi-evidence questions and made unsupported assertions
about individual allergies and unspecified cancellation fees. Restricting
context to gold chunks improved some answers but did not guarantee grounding.
The no-context fallback improved through deterministic application logic,
not prompt wording. Neither model has been validated for clinical use.

**Methodology, observed responses, limitations, and reproduction commands:**
[`evaluation/GENERATION_PILOT_REPORT.md`](evaluation/GENERATION_PILOT_REPORT.md).
Saved development outputs are in [`evaluation/results/generation/`](evaluation/results/generation/).

## Repository layout

```text
.
├── app/
│   ├── main.py                  # Composition root
│   ├── api.py                   # FastAPI integration
│   ├── schemas.py               # API models
│   └── chatbot/
│       ├── knowledge.py         # Markdown loading
│       ├── chunking.py          # Section chunks
│       ├── retrieval.py         # MiniLM and semantic search
│       ├── service.py           # Context, generation, provenance
│       ├── llm.py               # Local Qwen interface
│       └── models.py            # Domain models
├── data/knowledge/             # English-language source documents
├── evaluation/
│   ├── RETRIEVAL_BENCHMARK_PROTOCOL.md
│   ├── GENERATION_PILOT_REPORT.md
│   ├── retrieval_analysis.ipynb
│   ├── build_chunk_catalog.py
│   ├── validate_retrieval_dataset.py
│   ├── retrieval_metrics.py
│   ├── run_retrieval_benchmark.py
│   ├── run_grounding_pilot.py
│   ├── run_oracle_context_pilot.py
│   ├── summarize_retrieval_results.py
│   ├── evaluate_retrieval_threshold.py
│   ├── datasets/
│   │   ├── chunk_catalog.json
│   │   ├── retrieval_dev.json
│   │   └── retrieval_test.json
│   └── results/
│       ├── retrieval_dev.json
│       ├── retrieval_test.json
│       ├── retrieval_summary_dev.json
│       ├── retrieval_summary_test.json
│       ├── retrieval_threshold_dev.json
│       ├── retrieval_threshold_test.json
│       └── generation/          # Exploratory generation pilot outputs
├── tests/
├── chat_test.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Run locally

Python 3.12 was used during development.

```bash
git clone https://github.com/gonzaloangaut/medical_chatbot_sim.git
cd medical_chatbot_sim
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

API endpoint: `http://127.0.0.1:8000`  
Interactive API docs: `http://127.0.0.1:8000/docs`

Terminal client:

```bash
python chat_test.py
```

### Docker

```bash
docker build -t medical-chatbot .
docker run -p 8000:8000 medical-chatbot
```

### Tests and benchmark

Install test and notebook tooling separately from the application dependencies if needed:

```bash
pip install pytest pandas numpy matplotlib jupyter
pytest -q
```

To reproduce the saved retrieval experiments **without retraining models**:

```bash
python -m evaluation.run_retrieval_benchmark --dataset dev
python -m evaluation.summarize_retrieval_results --dataset dev
python -m evaluation.evaluate_retrieval_threshold --dataset dev

python -m evaluation.run_retrieval_benchmark --dataset test
python -m evaluation.summarize_retrieval_results --dataset test
python -m evaluation.evaluate_retrieval_threshold --dataset test
```

The `test` threshold command evaluates **only the previously selected value, 0.30**; do not retune on its results. The corpus is validated against the saved catalog before running the benchmark. The notebook works with the saved JSON outputs without loading MiniLM again.

## Next steps

Next priorities are **evidence-sufficiency and abstention controls**, followed by independent end-to-end evaluation of generated claims, answer completeness, and provenance. The present generation pilots are diagnostic development results; clinical safety is neither claimed nor established.
