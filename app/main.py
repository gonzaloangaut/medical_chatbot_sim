from fastapi import FastAPI

from logic import MedicalAssistance
from app.schemas import ChatRequest
from app.chatbot.llm import QwenLLM

# Create the APP
app = FastAPI()

# Load the LLM
llm = QwenLLM()

# Load the chatbot
bot = MedicalAssistance(llm=llm)

# Read the context file
try:
    with open("context.txt", "r", encoding="utf-8") as f:
        context = f.read()
except FileNotFoundError:
    context = "No context available."

# Get to check the status of the service
@app.get("/health")
def health():
    return {"status": "ok"}

# Post petition
@app.post("/predict")
def predict(request: ChatRequest):
    chat_history_dicts = [m.model_dump() for m in request.messages]
    # Get the answer from the bot
    response = bot.generate_response(chat_history_dicts, context)

    return {"response": response}
