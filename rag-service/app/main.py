from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import settings
from .rag import engine


class RagRequest(BaseModel):
    query: str = Field(..., min_length=1)
    topK: int = Field(default=settings.default_top_k, ge=1, le=20)
    # Per-request toggles for the three prompt-injection defense layers.
    layer1: bool = True  # hardened system prompt + <context> delimiters
    layer2: bool = True  # sanitize retrieved documents
    layer3: bool = True  # scrub the generated answer


class SourceModel(BaseModel):
    id: str
    title: str
    score: float
    snippet: str


class RagResponse(BaseModel):
    answer: str | None
    sources: list[SourceModel]
    generated: bool
    generationError: str | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Build the FAISS index once at startup (loads the bge-m3 model).
    engine.build()
    yield


app = FastAPI(title="Эйтрин RAG service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if engine.ready else "starting",
        "documents": len(engine.documents),
        "embeddingModel": settings.embedding_model,
        "generationEnabled": bool(settings.openai_api_key),
    }


@app.post("/rag", response_model=RagResponse)
def rag(req: RagRequest) -> RagResponse:
    if not engine.ready:
        raise HTTPException(status_code=503, detail="RAG engine is still starting")

    sources = engine.search(req.query, req.topK)
    answer, error = engine.generate(
        req.query,
        sources,
        layer1=req.layer1,
        layer2=req.layer2,
        layer3=req.layer3,
    )
    return RagResponse(
        answer=answer,
        sources=[SourceModel(**s.__dict__) for s in sources],
        generated=answer is not None,
        generationError=error,
    )
