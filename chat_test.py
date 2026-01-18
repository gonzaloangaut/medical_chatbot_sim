import requests
import json

# URL local donde va a estar escuchando Uvicorn
URL = "http://127.0.0.1:8000/predict"

def main():
    print("\n Asistente Médico")
    print("Escribir 'salir' para terminar.\n")
    
    # Historial local para mantener la memoria de la charla
    history = []

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["salir", "exit"]:
                break
            
            # Agregamos tu mensaje
            history.append({"role": "user", "content": user_input})
            
            # Preparamos el paquete
            payload = {"messages": history}

            # Enviamos a la API
            response = requests.post(URL, json=payload)
            
            if response.status_code == 200:
                bot_reply = response.json()["response"]
                print(f"BOT: {bot_reply}\n")
                
                # Guardamos la respuesta del bot en el historial
                history.append({"role": "assistant", "content": bot_reply})
            else:
                print(f"Error {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            print("ERROR: No se puede conectar a la API. ¿Corriste uvicorn?")
            break

if __name__ == "__main__":
    main()