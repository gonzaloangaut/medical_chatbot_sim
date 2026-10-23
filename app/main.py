import os

from app.api import create_app
from app.chatbot.chunking import chunk_documents
from app.chatbot.evidence import (
    BinaryEvidenceGate,
    parse_evidence_gate_enabled,
    parse_zero_one_setting,
)
from app.chatbot.knowledge import load_documents
from app.chatbot.llm import QwenLLM
from app.chatbot.retrieval import SemanticRetriever
from app.chatbot.service import MedicalAssistance

# Load structured knowledge documents
documents = load_documents()

# Split documents into retrievable chunks
chunks = chunk_documents(documents)

# Load retriever and ingest chunks
retriever = SemanticRetriever()
retriever.ingest_chunks(chunks)

# Explicitly opt in to the experimental gate; default behavior stays unchanged.
use_evidence_gate = parse_evidence_gate_enabled(
    os.getenv("MEDICAL_EVIDENCE_GATE_ENABLED", "0")
)
model_name = os.getenv(
    "MEDICAL_LLM_MODEL",
    "Qwen/Qwen2.5-1.5B-Instruct"
    if use_evidence_gate else "Qwen/Qwen2.5-0.5B-Instruct",
)

# Preserve historical sampling when the gate is off, and use the pilot's
# greedy decoding when it is on. Explicitly pin both values in A/B comparisons.
do_sample = parse_zero_one_setting(
    os.getenv("MEDICAL_LLM_DO_SAMPLE", "0" if use_evidence_gate else "1"),
    "MEDICAL_LLM_DO_SAMPLE",
)

# Share one model instance for gate and answer generation when enabled.
llm = QwenLLM(model_name=model_name, do_sample=do_sample)
evidence_gate = BinaryEvidenceGate(llm) if use_evidence_gate else None

# Load the chatbot
bot = MedicalAssistance(
    llm=llm,
    retriever=retriever,
    evidence_gate=evidence_gate,
)

# Create the FastAPI app
app = create_app(bot)
