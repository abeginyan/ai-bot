# RAG service

Python FastAPI microservice that answers questions about the **Эйтрин** universe
using the docs in `../knowledge_base/universe_docs/`.

- **Retrieval:** `BAAI/bge-m3` embeddings (sentence-transformers) + a FAISS
  inner-product index over L2-normalized vectors (cosine similarity). The index
  is built once at startup, one chunk per document.
- **Generation:** OpenAI chat completions, grounded strictly on the retrieved
  docs. If `OPENAI_API_KEY` is unset the service runs **retrieval-only** —
  it returns matching sources with `answer: null`, `generated: false`.

## API

`POST /rag`
```json
{ "query": "Кто такой Ферро-Страж?", "topK": 4 }
```
→
```json
{
  "answer": "…",
  "generated": true,
  "sources": [
    { "id": "hero_ferro_strazh", "title": "ФЕРРО-СТРАЖ", "score": 0.87, "snippet": "…" }
  ]
}
```

`GET /health` → `{ status, documents, embeddingModel, generationEnabled }`

The React frontend calls this service **directly** from the browser (`POST /rag`
on `:8000`), so CORS is enabled for the frontend origins — see `CORS_ORIGINS`.

## Configuration (env vars)

| Variable | Default | Purpose |
|---|---|---|
| `KNOWLEDGE_BASE_DIR` | `/knowledge_base` | Dir containing `universe_docs/*.txt` |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | sentence-transformers model |
| `RAG_TOP_K` | `4` | Default number of docs retrieved |
| `OPENAI_API_KEY` | _(empty)_ | Enables answer generation |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `OPENAI_BASE_URL` | _(empty)_ | Optional OpenAI-compatible endpoint |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:3000` | Comma-separated browser origins allowed to call directly |

## Run locally (without Docker)

```bash
pip install -r requirements.txt
export KNOWLEDGE_BASE_DIR=../knowledge_base
export OPENAI_API_KEY=sk-...        # optional
uvicorn app.main:app --port 8000
```

> First start downloads the bge-m3 model (~2.3 GB). The Docker image bakes it in
> at build time so container startup is fast.
