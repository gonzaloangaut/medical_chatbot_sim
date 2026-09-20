from fastapi.testclient import TestClient

from app.api import create_app


class FakeBot:
    """
    Fake Bot class to simulate the behavior of the MedicalAssistance bot for testing purposes.
    """
    def __init__(self):
        self.received_history = None
    def generate_response(self, chat_history):
        self.received_history = chat_history
        return "RESPUESTA_FAKE"


# Create a FastAPI app with the FakeBot for testing
bot = FakeBot()
app = create_app(bot)

client = TestClient(app)


def test_health_returns_ok():
    """
    Test that the /health endpoint returns a 200 status code and the expected JSON response.
    """
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_bot_response():
    """
    Test that the /predict endpoint returns a 200 status code and the expected JSON response.
    """
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


def test_predict_rejects_message_without_content():
    """
    Test that the /predict endpoint returns a 422 status code when a message is missing the 'content' field.
    """
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


def test_predict_uses_default_user_role():
    """
    Test that the /predict endpoint uses the default role 'user' when the role is not provided in the message.
    """
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