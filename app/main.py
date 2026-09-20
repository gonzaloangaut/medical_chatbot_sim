from app.api import create_app
from app.chatbot.knowledge import load_knowledge
from app.chatbot.llm import QwenLLM
from app.chatbot.retrieval import SemanticRetriever
from app.chatbot.service import MedicalAssistance

# Read the context file
knowledge = load_knowledge()

# Load retriever and ingest context
retriever = SemanticRetriever()
retriever.ingest_context(knowledge)

# Load the LLM
llm = QwenLLM()

# Load the chatbot
bot = MedicalAssistance(
    llm=llm,
    retriever=retriever,
)

# Create the FastAPI app
app = create_app(bot)
