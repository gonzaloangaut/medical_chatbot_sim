from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict

from logic import MedicalAssistance

# Create the APP
app = FastAPI()

# Load the chatbot
bot = MedicalAssistance()


# Give structure to the API
class Message(BaseModel):
    role: str = Field(default="user", description="Quién envía el mensaje")
    content: str = Field(..., examples=["Tengo fiebre."])


class ChatRequest(BaseModel):
    messages: List[Message]


# Read the context file
try:
    with open("context.txt", "r", encoding="utf-8") as f:
        context = f.read()
except FileNotFoundError:
    context = "No context available."


# Post petition
@app.post("/predict")
def predict(request: ChatRequest):
    chat_history_dicts = [m.model_dump() for m in request.messages]
    # Get the answer from the bot
    response = bot.generate_response(chat_history_dicts, context)

    return {"response": response}
