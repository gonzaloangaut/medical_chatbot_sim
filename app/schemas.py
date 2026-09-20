from pydantic import BaseModel, Field


# Give structure to the API
class Message(BaseModel):
    role: str = Field(default="user", description="Quién envía el mensaje")
    content: str = Field(..., examples=["Tengo fiebre."])


class ChatRequest(BaseModel):
    messages: list[Message] = Field(
        ...,
        min_length=1,
    )
