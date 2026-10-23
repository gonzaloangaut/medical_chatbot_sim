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

The current similarity gate is **not an evidence-sufficiency detector**. A query can pass it even when the KB lacks the requested fact. An opt-in experimental LLM evidence gate is now available, but reliable complete-answer detection and response-level groundedness remain unvalidated.

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

**Main finding:** Semantic similarity finds **related** text; it does not establish that the retrieved text contains the **specific evidence required** to answer a question. Development sufficiency diagnostics and grounding pilots are documented below; end-to-end answer evaluation is still pending.

**Interpretation limits:** The KB is small; the questions are mostly curated from the source material rather than sampled from clinical interactions; and development/test share documents, topics, and some underlying facts. This is a held-out **question-level retrieval benchmark**, not proof of real-world multilingual robustness, diagnostic correctness, or patient safety. Only one gold evidence set is annotated per question.

**Explore the analysis:** [`evaluation/retrieval_analysis.ipynb`](evaluation/retrieval_analysis.ipynb)  
**Methodology and full results:** [`evaluation/retrieval_analysis.ipynb`](evaluation/retrieval_analysis.ipynb)

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
[`evaluation/generation_analysis.ipynb`](evaluation/generation_analysis.ipynb).
Saved development outputs are in [`evaluation/results/generation/`](evaluation/results/generation/).

## Experimental evidence-completeness assessment

An **optional, disabled-by-default** binary evidence gate is available between
retrieval and generation (`app/chatbot/evidence.py`). It requires support for
**every essential fact requested by the question** and returns only
`SUFFICIENT` or `INSUFFICIENT`; a partially supported question must be rejected
under this strict full-answer policy. Invalid evaluator output fails closed;
model execution failures remain service errors.

The gate was evaluated with Qwen2.5-1.5B, greedy decoding, on development-only,
purposefully constructed evidence cases. On **19 selected saved-retrieval
examples**, the binary gate rejected 9/9 insufficient contexts but rejected
3/10 sufficient ones. On the **30-case diagnostic**, it incorrectly accepted
6/20 questions without full support (all originally labeled `PARTIAL`). It
also falsely accepted 4/8 negative variants in a small fixed-question study.
The gold annotations remain pending author sign-off. These are **not**
independent generalization, clinical-safety, or grounded-generation estimates.

Enable the experimental gate with the same evaluated model settings:

```bash
MEDICAL_EVIDENCE_GATE_ENABLED=1 \
MEDICAL_LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
MEDICAL_LLM_DO_SAMPLE=0 \
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

For a valid end-to-end comparison, fix model/sampling/retrieval settings and
change only the gate flag. The no-gate application behavior is still the
default.

**Evaluation, methodology, case-level audit, charts and architectural decision:**
[`evaluation/sufficiency_analysis.ipynb`](evaluation/sufficiency_analysis.ipynb).
The accompanying historical inference JSON files are in
[`evaluation/results/sufficiency/`](evaluation/results/sufficiency/).
The notebook opens and analyzes saved results without loading the LLM.

## Repository layout

```text
.
├── app/
│   ├── main.py                     # Composition root and opt-in gate flag
│   ├── api.py                      # FastAPI endpoints
│   └── chatbot/                    # Retrieval, grounded generation, evidence gate
├── data/knowledge/                # Frozen English educational and demo-policy KB
├── evaluation/
│   ├── retrieval_analysis.ipynb    # Ranking, threshold and holdout evaluation
│   ├── generation_analysis.ipynb   # Qualitative Qwen prompt / oracle pilots
│   ├── sufficiency_analysis.ipynb  # Binary gate and counterfactual diagnostics
│   ├── validate_sufficiency_dataset.py
│   ├── datasets/                   # Frozen annotated queries / chunk snapshots
│   └── results/
│       ├── retrieval/              # Frozen Top-5 rankings (dev and test)
│       ├── generation/             # Historical Qwen responses
│       └── sufficiency/            # Full development and counterfactual outputs
├── tests/                          # Unit tests for the running chatbot and validation
├── requirements.txt
├── Dockerfile
└── README.md
```

Each notebook contains its own experimental protocol, derived metrics, result visualizations, limitations and conclusions. Notebooks load saved JSON by default rather than repeatedly invoking language models. The sufficiency annotations are still **draft, pending author sign-off**.

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

### Tests and evaluation notebooks

```bash
python -m pytest -q
python -m evaluation.validate_sufficiency_dataset
jupyter lab
```

Run the three notebooks from the repository root or `evaluation/`:

- [`retrieval_analysis.ipynb`](evaluation/retrieval_analysis.ipynb): ranking and threshold statistics are recalculated from `evaluation/results/retrieval/retrieval_dev.json` and `retrieval_test.json`. A disabled-by-default cell can recompute embeddings with MiniLM and save **new** ranking artifacts. Candidate thresholds are development-only; the held-out test remains frozen at 0.30.
- [`generation_analysis.ipynb`](evaluation/generation_analysis.ipynb): qualitative prompt/oracle comparisons computed from archived Qwen responses.
- [`sufficiency_analysis.ipynb`](evaluation/sufficiency_analysis.ipynb): the three-class and binary assessment comparison and evidence counterfactuals, with opt-in binary model replication.

The analysis notebooks use Jupyter, pandas, numpy, matplotlib and IPython in addition to application dependencies. Historical model outputs are not guaranteed byte-reproducible because exact model revisions and dependency versions were not all pinned.

## Next steps

Next priorities are **independent end-to-end evaluation** of the base pipeline and opt-in evidence gate, assessing unsupported claims, answer completeness, abstention behavior, and provenance. The present generation pilots are diagnostic development results; clinical safety is neither claimed nor established.
