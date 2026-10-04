# Build from the repository root:  docker build -f backend/Dockerfile -t prism-api .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt psycopg2-binary

COPY backend backend
COPY data/replay/sample_news.jsonl data/replay/sample_news.jsonl

WORKDIR /app/backend
EXPOSE 8000
CMD ["uvicorn", "prism.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
