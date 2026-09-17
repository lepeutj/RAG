FROM python:3.11-slim

WORKDIR /app

# Minimal system dependencies for compiling some machine-learning wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/

# The sentence-transformers model is cached in this volume after first startup.
VOLUME ["/app/storage", "/root/.cache"]

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
