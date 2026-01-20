FROM python:3.12-slim

WORKDIR /app

# 1. Instalar herramientas básicas y limpiar basura
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. TRUCO DE SENIOR: Instalar PyTorch CPU explícitamente PRIMERO.
# Esto baja la versión liviana que tiene aceleración matemática (MKL) para CPU.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 3. Copiar el resto de las librerías
COPY requirements.txt .

# 4. Instalar el resto. Como Torch ya está puesto, Pip no baja el pesado.
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar tu código
COPY . .

# 6. Configurar puerto y arranque
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]