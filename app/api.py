import logging

from fastapi import FastAPI, HTTPException

from app.chatbot.exceptions import GenerationError
from app.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


def create_app(bot):
    """
    Factory function to create a FastAPI app with the given bot instance.
    """
    app = FastAPI()

    @app.get("/health")
    def health():
        """
        Health check endpoint to verify that the API is running.
        """
        return {"status": "ok"}

    @app.post(
        "/predict",
        response_model=ChatResponse,
    )
    def predict(request: ChatRequest):
        """
        Predict endpoint that takes a chat request and returns a response generated
        by the bot.
        """
        chat_history_dicts = [
            message.model_dump()
            for message in request.messages
        ]

        try:
            result = bot.generate_response(chat_history_dicts)

        except GenerationError as error:
            logger.exception("Response generation failed.")

            raise HTTPException(
                status_code=503,
                detail="Response generation is temporarily unavailable.",
            ) from error

        return {
            "response": result.response,
            "sources": [
                {
                    "title": source.title,
                    "section": source.section,
                    "source": source.source,
                    "source_url": source.source_url,
                }
                for source in result.sources
            ],
        }

    return app
