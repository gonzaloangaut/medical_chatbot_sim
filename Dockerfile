FROM python:3.12-slim

WORKDIR /app

# Install the tools and clean trash
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Load PyTorch for CPU (lighter)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy every requirment
COPY requirements.txt .

# Install the libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy the code
COPY . .

# Configure port and boot
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]