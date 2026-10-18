FROM python:3.11-slim

WORKDIR /app

# System deps occasionally needed by faiss / torch wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Pre-download the bge-m3 model into the image so first request is fast.
# Set HF cache so it is reused at runtime.
ENV HF_HOME=/models
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

ENV KNOWLEDGE_BASE_DIR=/knowledge_base
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
