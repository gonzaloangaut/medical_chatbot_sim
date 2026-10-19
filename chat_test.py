import requests

# URL for Uvicorn
URL = "http://127.0.0.1:8000/predict"


def main():
    print("\n Asistente Médico")
    print("Escribir 'salir' para terminar.\n")

    # Load chat history for memory
    history = []

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["salir", "exit"]:
                break

            # Add message to the history
            history.append({"role": "user", "content": user_input})

            # Prepare it
            payload = {"messages": history}

            # Send it to the API
            response = requests.post(URL, json=payload)

            # Get the answer
            if response.status_code == 200:
                data = response.json()

                bot_reply = data["response"]
                sources = data["sources"]

                print(f"BOT: {bot_reply}\n")

                if sources:
                    print("SOURCES:")
                    for source in sources:
                        print(
                            f"- {source['source']} — "
                            f"{source['title']} / {source['section']}"
                        )
                        print(f"  {source['source_url']}")
                    print()

                history.append(
                    {
                        "role": "assistant",
                        "content": bot_reply,
                    }
                )
            else:
                print(f"Error {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            print("ERROR: No se puede conectar a la API")
            break


if __name__ == "__main__":
    main()
