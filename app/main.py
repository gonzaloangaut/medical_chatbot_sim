from app.api import create_app
from app.chatbot.chunking import chunk_documents
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

# Load the LLM
llm = QwenLLM()

# Load the chatbot
bot = MedicalAssistance(
    llm=llm,
    retriever=retriever,
)

# Create the FastAPI app
app = create_app(bot)