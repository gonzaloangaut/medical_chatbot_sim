import pytest

from fastapi.testclient import TestClient

from app.api import create_app
from app.chatbot.exceptions import GenerationError


class FakeBot:
    """
    Fake Bot class to simulate the behavior of the MedicalAssistance bot for testing purposes.
    """
    def __init__(self):
        self.received_history = None

    def generate_response(self, chat_history):
        self.received_history = chat_history
        return "RESPUESTA_FAKE"

class FailingBot:
    """
    Fake bot that simulates a generation failure.
    """

    def generate_response(self, chat_history):
        raise GenerationError("Generation failed.")
    

@pytest.fixture
def api_setup():
    """
    Fixture to set up the FastAPI test client and the fake bot for testing the API endpoints.
    """
    bot = FakeBot()
    app = create_app(bot)
    client = TestClient(app)

    return client, bot


def test_health_returns_ok(api_setup):
    """
    Test that the /health endpoint returns a 200 status code and the expected JSON response.
    """
    client, _ = api_setup
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_bot_response(api_setup):
    """
    Test that the /predict endpoint returns a 200 status code and the expected JSON response.
    """
    client, _ = api_setup
    response = client.post(
        "/predict",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Tengo fiebre"
                }
            ]
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "response": "RESPUESTA_FAKE"
    }


def test_predict_rejects_message_without_content(api_setup):
    """
    Test that the /predict endpoint returns a 422 status code when a message is missing the 'content' field.
    """
    client, _ = api_setup
    response = client.post(
        "/predict",
        json={
            "messages": [
                {
                    "role": "user"
                }
            ]
        },
    )

    assert response.status_code == 422


def test_predict_uses_default_user_role(api_setup):
    """
    Test that the /predict endpoint uses the default role 'user' when the role is not provided in the message.
    """
    client, bot = api_setup
    response = client.post(
        "/predict",
        json={
            "messages": [
                {
                    "content": "Tengo fiebre"
                }
            ]
        },
    )

    assert response.status_code == 200
    assert bot.received_history[0]["role"] == "user"
    assert bot.received_history[0]["content"] == "Tengo fiebre"


def test_predict_rejects_empty_messages(api_setup):
    """
    Test that /predict rejects a request with an empty messages list.
    """
    client, _ = api_setup

    response = client.post(
        "/predict",
        json={
            "messages": []
        },
    )

    assert response.status_code == 422

def test_predict_returns_503_when_generation_fails():
    """
    Test that the /predict endpoint returns a 503 status code when the bot fails to generate a response.
    """
    bot = FailingBot()
    app = create_app(bot)
    client = TestClient(app)

    response = client.post(
        "/predict",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Tengo fiebre",
                }
            ]
        },
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Response generation is temporarily unavailable."
    }