from pydantic import BaseModel, Field


# Give structure to the API
class Message(BaseModel):
    """
    A single message in the chat history.
    """
    
    role: str = Field(default="user", description="Quién envía el mensaje")
    content: str = Field(..., examples=["Tengo fiebre."])


class ChatRequest(BaseModel):
    """
    Request body for the chatbot API.
    """

    messages: list[Message] = Field(
        ...,
        min_length=1,
    )

class SourceResponse(BaseModel):
    """
    Source metadata returned by the API.
    """

    title: str
    section: str
    source: str
    source_url: str


class ChatResponse(BaseModel):
    """
    Response returned by the chatbot API.
    """

    response: str
    sources: list[SourceResponse]