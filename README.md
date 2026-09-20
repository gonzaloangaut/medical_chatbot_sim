# Medical Chatbot Simulation

Este proyecto es una simulación de un Asistente Médico, cuya tarea es responder preguntas sobre síntomas y administración de turnos, basándose en una base de conocimiento en texto plano (`context.txt`).

---

## Características Principales

* **Local y liviano:** Utiliza el modelo liviano `Qwen 2.5 (0.5B)` corriendo en CPU (esto ayuda a que el Docker no sea tan pesado).
* **RAG:** Busca por síntoma para encontrar similitudes con la pregunta del usuario y luego contesta con protocolo. El contexto está separado de la siguiente forma:
  ```plaintext
  Palabras Clave (Síntomas y Sinónimos) 
  @@@ 
  Protocolo Oficial (Respuesta Técnica del Bot)
  ###
  ```
  para poder separar cada tema con "###" y los síntomas del protocolo mediante "@@@".
* **Embedding:** Usa el embedding `paraphrase-multilingual-MiniLM-L12-v2` que es liviano y sirve para español. El embedding busca similitud entre la pregunta del usuario y los síntomas.
* **Control:** Filtros de umbral de similitud para descartar preguntas que no tengan que ver con un tema médico.
* **Arquitectura Dockerizada:** Listo para desplegar en cualquier entorno compatible con Docker.

## Stack Tecnológico

* **Lenguaje:** Python 3.9+
* **LLM (Small Language Model):** Qwen/Qwen2.5-0.5B-Instruct
* **Embeddings:** Sentence-Transformers
* **Backend:** FastAPI / Uvicorn
* **Container:** Docker

---

## Estructura del repositorio

```text
app/
├── main.py              # Ensambla los componentes de la aplicación
├── api.py               # Define la API FastAPI
├── schemas.py           # Modelos y validación de requests
└── chatbot/
    ├── service.py       # Coordina retrieval, prompt y generación
    ├── llm.py           # Implementación del modelo Qwen
    ├── retrieval.py     # Retrieval semántico
    ├── knowledge.py     # Carga de la base de conocimiento
    └── exceptions.py    # Excepciones propias de la aplicación

data/
└── context.txt          # Base de conocimiento

tests/
├── test_api.py
├── test_retrieval.py
└── test_service.py
---

## Instrucciones para la ejecución

Primero que nada, se debe clonar el repositorio:

 ```bash
 git clone <repo_url>
 ```

Luego, hay 2 opciones: con o sin Docker.

### Opción 1: Usando Docker (Recomendado para aislar el entorno)

**1. Requisitos**
Tener instalado Docker.

**2. Construir la imagen**
La construcción puede tardar algunos minutos debido a la instalación de PyTorch y las dependencias del proyecto. La primera ejecución puede tardar más mientras se descargan los modelos necesarios.
Para esto, correr en la terminal:

```bash
docker build -t medical_chatbot .
```


**3. Correr el contenedor**
Esto iniciará el servidor API en el puerto 8000.

```bash
docker run -p 8000:8000 medical_chatbot
```


**4. Probar el Chatbot**
En otra terminal, ejecuta el script cliente para iniciar el chat:

```bash
python chat_test.py
```

### Opción 2: Ejecución Local (Sin Docker)

También, se puede correr local con un entorno de Python.

**1. Crear un entorno virtual**
Hay muchas formas, en Linux la más común es desde la terminal con el comando:

```bash
python -m venv venv
source venv/bin/activate
```


**2. Instalar dependencias**

```bash
pip install -r requirements.txt
```


**3. Iniciar la API**

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**4. Probar el Chatbot**
En otra terminal, ejecuta el script cliente para iniciar el chat:

```bash
python chat_test.py
```


## Ejemplo de interacción

Debajo se encuentra un pequeño ejemplo de una conversación con el asistente usando `chat_test.py`:
<img width="2731" height="1369" alt="Captura desde 2026-01-21 17-33-09" src="https://github.com/user-attachments/assets/e4e6f1fe-b64c-4bfb-9190-f1e412263de6" />

