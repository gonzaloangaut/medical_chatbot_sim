# Medical Chatbot Simulation

A lightweight medical-information chatbot built as an educational project to explore API design, software architecture, testing, and Retrieval-Augmented Generation (RAG).

The application retrieves relevant information from a structured knowledge base, provides the retrieved context to a local language model, and returns both the generated answer and the sources used during retrieval.

> **Disclaimer:** This project is an educational prototype. It is not a medical device and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

## Current Architecture

The current RAG pipeline is:

```text
Markdown knowledge documents
        ↓
Document loading + metadata
        ↓
Structure-aware chunking
        ↓
Semantic embeddings
        ↓
Top-k retrieval
        ↓
Retrieved chunks
        ↓
Structured LLM context
        ↓
Generated answer + provenance
        ↓
FastAPI response
```

The application intentionally keeps each responsibility separate:

```text
knowledge.py
    Markdown files → Document objects

chunking.py
    Documents → Chunk objects

retrieval.py
    Chunks → ranked RetrievedChunk objects

service.py
    Retrieval → context construction → generation → provenance

api.py
    Domain response → HTTP response

main.py
    Application composition root
```

## Main Features

### Structured knowledge base

Knowledge is stored as Markdown documents under:

```text
data/knowledge/
```

Each document contains front matter with provenance metadata:

```yaml
---
document_id: medlineplus_fever
title: Fever
source: MedlinePlus - National Library of Medicine
source_url: https://medlineplus.gov/fever.html
---
```

The Markdown body is divided into sections using level-two headings (`##`).

### Structure-aware chunking

Each non-empty `##` section becomes a retrievable `Chunk`.

A chunk preserves:

- document identity;
- title;
- section;
- source;
- source URL;
- textual content.

This keeps retrieval units separate from source documents while preserving provenance.

### Semantic retrieval

The project uses:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Each chunk is embedded using:

```text
title + section + content
```

Queries are ranked using cosine similarity.

The retriever currently returns the top 3 candidates by default and applies a similarity threshold to the highest-ranked result as a simple retrieve-or-reject decision.

These retrieval settings are working defaults and will be evaluated more systematically as the knowledge base grows.

### Local language model

Generation currently uses:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

The model runs locally on CPU.

Retrieved chunks are converted into structured context before being sent to the language model.

### Provenance

The API does not ask the language model to generate citations.

Instead, provenance is propagated programmatically from the retrieved chunks to the final API response.

A response has the form:

```json
{
  "response": "Generated response",
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

Retrieval scores are intentionally kept internal and are not exposed as confidence values.

### Testing

The project uses `pytest` and test doubles to validate the main application contracts without loading real ML models during unit tests.

Tests cover areas including:

- API validation and error handling;
- document loading;
- Markdown chunking;
- semantic retrieval orchestration;
- retrieve-or-reject behavior;
- context construction;
- provenance propagation;
- service-level generation behavior.

## Technology Stack

- **Python 3.12**
- **FastAPI**
- **Pydantic**
- **Uvicorn**
- **Sentence Transformers**
- **Transformers**
- **PyTorch**
- **Qwen2.5-0.5B-Instruct**
- **pytest**
- **Docker**

## Repository Structure

```text
.
├── app/
│   ├── main.py
│   ├── api.py
│   ├── schemas.py
│   └── chatbot/
│       ├── chunking.py
│       ├── exceptions.py
│       ├── knowledge.py
│       ├── llm.py
│       ├── models.py
│       ├── retrieval.py
│       └── service.py
│
├── data/
│   └── knowledge/
│       └── fever.md
│
├── evaluation/
│   ├── analyze_retrieval_results.ipynb
│   ├── representation_experiment_results.json
│   ├── retrieval_catalog.json
│   ├── retrieval_dataset.json
│   └── retrieval_results.json
│
├── tests/
│   ├── test_api.py
│   ├── test_chunking.py
│   ├── test_knowledge.py
│   ├── test_retrieval.py
│   └── test_service.py
│
├── chat_test.py
├── Dockerfile
├── requirements.txt
└── README.md
```

The files under `evaluation/` preserve experiments performed during an earlier retrieval iteration. They provide historical evidence for retrieval-design decisions but should not be interpreted as validation of the current multi-document RAG architecture.

## Running Locally

Clone the repository:

```bash
git clone https://github.com/gonzaloangaut/medical_chatbot_sim.git
cd medical_chatbot_sim
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI documentation is available at:

```text
http://127.0.0.1:8000/docs
```

To run the terminal client:

```bash
python chat_test.py
```

## Running with Docker

Build the image:

```bash
docker build -t medical-chatbot .
```

Run the container:

```bash
docker run -p 8000:8000 medical-chatbot
```

Then use the API at:

```text
http://127.0.0.1:8000
```

## Running the Tests

```bash
pytest
```

## Current Scope and Next Steps

The current knowledge base is intentionally small. Its purpose is to validate the structured RAG architecture before scaling the corpus.

The next development stage will focus on:

- expanding the knowledge base with multiple traceable documents;
- building a new retrieval evaluation dataset for the current architecture;
- evaluating Top-k retrieval and retrieve-or-reject behavior;
- analyzing out-of-domain queries;
- measuring grounded generation and unsupported claims;
- improving the prompt and generation behavior based on observed failures;
- evaluating whether more advanced retrieval techniques are actually necessary.

The project is being developed incrementally: additional infrastructure is introduced only when evaluation results provide a clear reason for it.
