FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface

COPY requirements.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/storage "$HF_HOME" \
    && chown -R app:app /app/storage /home/app

COPY --chown=app:app src/ ./src/
COPY --chown=app:app scripts/ ./scripts/

USER 10001:10001

CMD ["sh", "-c", "exec uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --no-server-header"]
