from pydantic import BaseModel, Field
from typing import List


# Give structure to the API
class Message(BaseModel):
    role: str = Field(default="user", description="Quién envía el mensaje")
    content: str = Field(..., examples=["Tengo fiebre."])


class ChatRequest(BaseModel):
    messages: List[Message]
