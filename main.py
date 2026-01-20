from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

from logic import MedicalAssistance 

# Create the APP
app = FastAPI()

# Load the chatbot
bot = MedicalAssistance() 

# Accept only lists of dictionaries
class ChatRequest(BaseModel):
    messages: List[Dict[str, str]] 

# Post petition
@app.post("/predict")
def predict(request: ChatRequest):
    # Read the context file
    try:
        with open("context.txt", "r", encoding="utf-8") as f:
            context = f.read()
    except FileNotFoundError:
        context = "No context available."

    
    # Get the answer from the bot
    response = bot.generate_response(request.messages, context)
    
    return {"response": response}