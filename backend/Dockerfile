FROM python:3.11-slim
WORKDIR /app/backend
COPY backend/pyproject.toml ./
COPY backend/app ./app
COPY backend/migrations ./migrations
COPY knowledge_base_seed /app/knowledge_base_seed
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir ".[documents,signals,ai,inference,retrieval]"
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
