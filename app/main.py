from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.rag import RAGService


settings = get_settings()
service = RAGService(settings)
static_dir = Path(__file__).parent / "static"

app = FastAPI(
    title="Corrective RAG",
    description="Cloud deployable Corrective RAG API and web interface.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: Literal["advanced", "standard"] = "advanced"
    namespace: str | None = None


class QueryResponse(BaseModel):
    answer: str
    mode: str
    rewritten_query: str | None
    retries: int
    sources: list[dict]


class HealthResponse(BaseModel):
    ok: bool
    configured: bool
    missing_env: list[str]
    configuration_errors: list[str]
    index_name: str
    llm_provider: str
    llm_model: str
    gemini_model: str
    embedding_model: str
    reranker_model: str


def _namespace(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        ok=True,
        configured=settings.configured,
        missing_env=settings.missing_env,
        configuration_errors=settings.configuration_errors,
        index_name=settings.pinecone_index_name,
        llm_provider=settings.llm_provider,
        llm_model=settings.active_llm_model,
        gemini_model=settings.gemini_model,
        embedding_model=settings.embedding_model,
        reranker_model=settings.reranker_model,
    )


@app.post("/api/ingest")
async def ingest_pdf(
    file: UploadFile = File(...),
    namespace: str | None = Form(None),
    chunk_size: int = Form(800),
    chunk_overlap: int = Form(100),
):
    filename = file.filename or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    target = settings.upload_dir / f"{uuid.uuid4().hex}.pdf"
    try:
        with target.open("wb") as output:
            shutil.copyfileobj(file.file, output)

        result = await run_in_threadpool(
            service.ingest_pdf,
            target,
            _namespace(namespace),
            chunk_size,
            chunk_overlap,
        )
        return {
            "pages": result.pages,
            "chunks": result.chunks,
            "namespace": result.namespace,
            "index_name": result.index_name,
        }
    except (RuntimeError, ValueError, TimeoutError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()
        target.unlink(missing_ok=True)


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    try:
        result = await run_in_threadpool(
            service.ask,
            request.query,
            request.mode,
            _namespace(request.namespace),
        )
        return QueryResponse(
            answer=result.answer,
            mode=result.mode,
            rewritten_query=result.rewritten_query,
            retries=result.retries,
            sources=result.sources,
        )
    except (RuntimeError, ValueError, TimeoutError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
