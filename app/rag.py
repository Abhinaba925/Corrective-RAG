from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from math import fsum
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel, Field, ValidationError
from pypdf import PdfReader
import requests

from app.config import Settings


class GradeDocuments(BaseModel):
    """Binary relevance decision used by the CRAG router."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


@dataclass
class RAGParams:
    standard_top_k: int = 15
    advanced_broad_k: int = 40
    advanced_final_k: int = 5
    max_retries: int = 2
    temperature: float = 0
    relevance_threshold: float | None = None
    enable_reranking: bool = True
    enable_query_rewrite: bool = True
    use_fallback: bool = True


@dataclass
class QueryMetrics:
    total_ms: float = 0
    rewrite_ms: float = 0
    retrieval_ms: float = 0
    rerank_ms: float = 0
    grading_ms: float = 0
    generation_ms: float = 0
    retrieved_docs: int = 0
    final_docs: int = 0
    top_score: float | None = None
    avg_score: float | None = None
    accepted_context: bool = False
    rewrite_triggered: bool = False
    no_answer_detected: bool = False
    fallback_used: bool = False
    fallback_reason: str | None = None


@dataclass
class RAGResult:
    answer: str
    sources: list[dict[str, Any]]
    mode: str
    provider: str
    model: str
    metrics: dict[str, Any]
    params: dict[str, Any]
    rewritten_query: str | None = None
    retries: int = 0


@dataclass
class IngestResult:
    pages: int
    chunks: int
    namespace: str | None
    index_name: str


@dataclass
class ChatResponse:
    content: str


class GroqChatModel:
    """Small OpenAI-compatible Groq adapter for the calls used in this app."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        temperature: float = 0,
        timeout: int = 90,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout

    def invoke(self, prompt: str) -> ChatResponse:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer directly and do not include hidden reasoning, "
                        "chain-of-thought, or <think> blocks."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code == 429:
            retry_after = response.headers.get("retry-after")
            suffix = f" Retry after {retry_after}s." if retry_after else ""
            raise RuntimeError(f"Groq rate limit exceeded.{suffix}")
        if response.status_code >= 400:
            raise RuntimeError(f"Groq API error {response.status_code}: {response.text}")

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return ChatResponse(content=_strip_thinking(content or ""))

    def with_structured_output(self, schema: type[BaseModel]) -> "StructuredGroqChatModel":
        return StructuredGroqChatModel(self, schema)


class StructuredGroqChatModel:
    def __init__(self, llm: GroqChatModel, schema: type[BaseModel]):
        self.llm = llm
        self.schema = schema

    def invoke(self, prompt: str) -> BaseModel:
        structured_prompt = (
            "Return only a valid JSON object matching this schema. "
            "Do not include markdown or extra text.\n"
            f"Schema fields: {list(self.schema.model_fields.keys())}\n\n"
            f"{prompt}"
        )
        content = self.llm.invoke(structured_prompt).content.strip()
        try:
            return self.schema.model_validate_json(content)
        except ValidationError:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if match:
                try:
                    return self.schema.model_validate(json.loads(match.group(0)))
                except (json.JSONDecodeError, ValidationError):
                    pass

        # The CRAG router only needs a conservative yes/no relevance signal.
        lowered = content.lower()
        if "yes" in lowered and "no" not in lowered:
            return self.schema(binary_score="yes")
        return self.schema(binary_score="no")


def _strip_thinking(content: str) -> str:
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def _require_config(settings: Settings) -> None:
    errors = settings.configuration_errors
    if errors:
        raise RuntimeError(" ".join(errors))

    missing = settings.missing_env
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variables: {joined}")


def _install_env(settings: Settings) -> None:
    # LangChain integrations also read these environment variables internally.
    if settings.google_api_key:
        os.environ["GOOGLE_API_KEY"] = settings.google_api_key
    if settings.groq_api_key:
        os.environ["GROQ_API_KEY"] = settings.groq_api_key
    if settings.pinecone_api_key:
        os.environ["PINECONE_API_KEY"] = settings.pinecone_api_key


def ensure_pinecone_index(settings: Settings) -> None:
    _require_config(settings)
    _install_env(settings)

    pc = Pinecone(api_key=settings.pinecone_api_key)
    if not _has_pinecone_index(pc, settings.pinecone_index_name):
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )

    deadline = time.time() + 120
    while time.time() < deadline:
        description = pc.describe_index(settings.pinecone_index_name)
        dimension = _index_value(description, "dimension")
        if dimension and int(dimension) != settings.embedding_dimension:
            raise ValueError(
                f"Pinecone index '{settings.pinecone_index_name}' has dimension "
                f"{dimension}, but EMBEDDING_DIMENSION is {settings.embedding_dimension}."
            )

        status = description.status
        if isinstance(status, dict):
            ready = status.get("ready")
        else:
            ready = getattr(status, "ready", None)
            if ready is None:
                ready = status["ready"]
        if ready:
            return
        time.sleep(1)

    raise TimeoutError(
        f"Pinecone index '{settings.pinecone_index_name}' was not ready after 120s."
    )


def _has_pinecone_index(pc: Pinecone, index_name: str) -> bool:
    if hasattr(pc, "has_index"):
        return bool(pc.has_index(index_name))

    indexes = pc.list_indexes()
    if hasattr(indexes, "names"):
        return index_name in indexes.names()

    for item in indexes:
        if isinstance(item, dict) and item.get("name") == index_name:
            return True
        if getattr(item, "name", None) == index_name:
            return True
    return False


def _index_value(description: Any, key: str) -> Any:
    if isinstance(description, dict):
        return description.get(key)
    return getattr(description, key, None)


def extract_pdf_documents(
    pdf_path: Path, source_name: str | None = None
) -> tuple[list[Document], int]:
    reader = PdfReader(str(pdf_path))
    documents: list[Document] = []
    source = source_name or pdf_path.name

    for index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"page": index + 1, "source": source},
                )
            )

    return documents, len(reader.pages)


def split_documents(
    documents: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(documents)


def serialize_sources(documents: list[Document]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for doc in documents:
        metadata = dict(doc.metadata or {})
        score = metadata.get("relevance_score")
        sources.append(
            {
                "page": metadata.get("page"),
                "source": metadata.get("source"),
                "score": round(float(score), 4) if score is not None else None,
                "preview": doc.page_content[:700],
                "content": doc.page_content,
            }
        )
    return sources


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def _looks_like_no_answer(answer: str) -> bool:
    lowered = answer.strip().lower()
    markers = (
        "context lacks",
        "context does not",
        "not provided in the context",
        "not mentioned in the context",
        "i don't know",
        "i do not know",
        "cannot determine",
        "missing",
        "unknown",
    )
    return any(marker in lowered for marker in markers)


def _score_summary(documents: list[Document]) -> tuple[float | None, float | None]:
    scores = [
        float(doc.metadata["relevance_score"])
        for doc in documents
        if doc.metadata.get("relevance_score") is not None
    ]
    if not scores:
        return None, None
    return round(max(scores), 4), round(fsum(scores) / len(scores), 4)


def _validate_params(params: RAGParams) -> RAGParams:
    params.standard_top_k = max(1, int(params.standard_top_k))
    params.advanced_broad_k = max(1, int(params.advanced_broad_k))
    params.advanced_final_k = max(1, min(int(params.advanced_final_k), params.advanced_broad_k))
    params.max_retries = max(0, int(params.max_retries))
    params.temperature = min(1.5, max(0, float(params.temperature)))
    return params


class StandardRAG:
    def __init__(
        self,
        llm: Any,
        vector_store: PineconeVectorStore,
        params: RAGParams,
        provider: str,
        model: str,
    ):
        self.llm = llm
        self.vector_store = vector_store
        self.params = params
        self.provider = provider
        self.model = model

    def ask(self, query: str) -> RAGResult:
        metrics = QueryMetrics()
        total_start = time.perf_counter()

        retrieval_start = time.perf_counter()
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.params.standard_top_k}
        )
        context = retriever.invoke(query)
        metrics.retrieval_ms = _elapsed_ms(retrieval_start)
        metrics.retrieved_docs = len(context)
        metrics.final_docs = len(context)

        context_str = "\n\n".join(
            [
                f"[Page {doc.metadata.get('page', '?')}]: {doc.page_content}"
                for doc in context
            ]
        )
        prompt = (
            "Answer the user's question using only the context below. "
            "If the context lacks the answer, state that it is missing.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {query}\nAnswer:"
        )

        generation_start = time.perf_counter()
        answer = self.llm.invoke(prompt).content
        metrics.generation_ms = _elapsed_ms(generation_start)
        metrics.total_ms = _elapsed_ms(total_start)
        metrics.accepted_context = bool(context)
        metrics.no_answer_detected = _looks_like_no_answer(answer)

        return RAGResult(
            answer=answer,
            sources=serialize_sources(context),
            mode="standard",
            provider=self.provider,
            model=self.model,
            metrics=asdict(metrics),
            params=asdict(self.params),
        )


class AdvancedRAG:
    """Corrective RAG with query rewriting, reranking, and relevance grading."""

    def __init__(
        self,
        llm: Any,
        reranker: Any,
        vector_store: PineconeVectorStore,
        params: RAGParams,
        provider: str,
        model: str,
    ):
        self.llm = llm
        self.grader_llm = llm.with_structured_output(GradeDocuments)
        self.reranker = reranker
        self.vector_store = vector_store
        self.params = params
        self.provider = provider
        self.model = model

    def _rewrite_query(self, query: str) -> str:
        prompt = (
            "Extract keywords and concepts from this question to optimize it "
            "for vector search. Do not answer it.\n"
            f"Question: {query}\nOptimized:"
        )
        rewritten = self.llm.invoke(prompt).content.strip()
        return rewritten or query

    def _retrieve_and_rerank(
        self, query: str, metrics: QueryMetrics
    ) -> list[Document]:
        retrieval_start = time.perf_counter()
        docs = self.vector_store.as_retriever(
            search_kwargs={"k": self.params.advanced_broad_k}
        ).invoke(query)
        metrics.retrieval_ms += _elapsed_ms(retrieval_start)
        metrics.retrieved_docs += len(docs)

        if not docs:
            return []

        if self.params.enable_reranking:
            rerank_start = time.perf_counter()
            pairs = [[query, doc.page_content] for doc in docs]
            scores = self.reranker.predict(pairs)
            for doc, score in zip(docs, scores):
                doc.metadata["relevance_score"] = float(score)
            docs.sort(key=lambda item: item.metadata["relevance_score"], reverse=True)
            metrics.rerank_ms += _elapsed_ms(rerank_start)

        return docs[: self.params.advanced_final_k]

    def _context_is_relevant(
        self, query: str, context: list[Document], metrics: QueryMetrics
    ) -> bool:
        if not context:
            return False

        top_score, avg_score = _score_summary(context)
        metrics.top_score = top_score
        metrics.avg_score = avg_score
        if (
            self.params.relevance_threshold is not None
            and top_score is not None
            and top_score < self.params.relevance_threshold
        ):
            return False

        context_str = "\n\n".join([doc.page_content for doc in context])
        prompt = (
            "Does this context relate to the question? "
            "Return only the structured binary score.\n"
            f"Q: {query}\nContext: {context_str}"
        )
        grading_start = time.perf_counter()
        grade = self.grader_llm.invoke(prompt)
        metrics.grading_ms += _elapsed_ms(grading_start)
        return grade.binary_score.strip().lower().startswith("yes")

    def _generate(self, original_query: str, context: list[Document]) -> str:
        context_str = "\n\n".join(
            [
                f"[Page {doc.metadata.get('page', '?')}]: {doc.page_content}"
                for doc in context
            ]
        )
        prompt = (
            "Answer using ONLY the provided context. If unknown, say so.\n\n"
            f"Question: {original_query}\n\n"
            f"Context:\n{context_str}\n\nAnswer:"
        )
        return self.llm.invoke(prompt).content

    def ask(self, query: str) -> RAGResult:
        metrics = QueryMetrics()
        total_start = time.perf_counter()
        current_query = query
        rewritten_query: str | None = None
        retries = 0
        final_context: list[Document] = []

        for attempt in range(self.params.max_retries + 1):
            final_context = self._retrieve_and_rerank(current_query, metrics)
            metrics.final_docs = len(final_context)
            relevant = self._context_is_relevant(current_query, final_context, metrics)
            metrics.accepted_context = relevant
            if relevant or attempt >= self.params.max_retries:
                break

            if not self.params.enable_query_rewrite:
                break

            rewrite_start = time.perf_counter()
            current_query = self._rewrite_query(query)
            metrics.rewrite_ms += _elapsed_ms(rewrite_start)
            metrics.rewrite_triggered = True
            rewritten_query = current_query
            retries += 1

        generation_start = time.perf_counter()
        answer = self._generate(query, final_context)
        metrics.generation_ms = _elapsed_ms(generation_start)
        metrics.total_ms = _elapsed_ms(total_start)
        metrics.no_answer_detected = _looks_like_no_answer(answer)

        return RAGResult(
            answer=answer,
            sources=serialize_sources(final_context),
            mode="advanced",
            provider=self.provider,
            model=self.model,
            metrics=asdict(metrics),
            params=asdict(self.params),
            rewritten_query=rewritten_query or current_query,
            retries=retries,
        )


class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = Lock()
        self._embeddings: Any | None = None
        self._llms: dict[tuple[str, str, float], Any] = {}
        self._reranker: Any | None = None
        self._index_ready = False

    def _prepare(self) -> None:
        _require_config(self.settings)
        _install_env(self.settings)
        if self._index_ready:
            return
        with self._lock:
            if not self._index_ready:
                ensure_pinecone_index(self.settings)
                self._index_ready = True

    @property
    def embeddings(self) -> Any:
        with self._lock:
            if self._embeddings is None:
                from langchain_huggingface import HuggingFaceEmbeddings

                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.settings.embedding_model,
                    encode_kwargs={"normalize_embeddings": True},
                )
            return self._embeddings

    def _llm_for(self, provider: str, temperature: float) -> Any:
        provider = provider.strip().lower()
        model = self.settings.groq_model if provider == "groq" else self.settings.gemini_model
        key = (provider, model, round(float(temperature), 3))
        with self._lock:
            if key not in self._llms:
                if provider == "groq":
                    self._llms[key] = GroqChatModel(
                        api_key=self.settings.groq_api_key or "",
                        model=model,
                        base_url=self.settings.groq_base_url,
                        temperature=temperature,
                    )
                else:
                    from langchain_google_genai import ChatGoogleGenerativeAI

                    self._llms[key] = ChatGoogleGenerativeAI(
                        model=model,
                        temperature=temperature,
                        google_api_key=self.settings.google_api_key,
                    )
            return self._llms[key]

    def _provider_model(self, provider: str) -> tuple[str, str]:
        provider = provider.strip().lower()
        if provider == "groq":
            return provider, self.settings.groq_model
        return "gemini", self.settings.gemini_model

    @property
    def reranker(self) -> Any:
        with self._lock:
            if self._reranker is None:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(self.settings.reranker_model)
            return self._reranker

    def _vector_store(self, namespace: str | None = None) -> PineconeVectorStore:
        return PineconeVectorStore(
            index_name=self.settings.pinecone_index_name,
            embedding=self.embeddings,
            namespace=namespace or None,
        )

    def ingest_pdf(
        self,
        pdf_path: Path,
        namespace: str | None = None,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        source_name: str | None = None,
    ) -> IngestResult:
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self._prepare()
        documents, page_count = extract_pdf_documents(pdf_path, source_name)
        if not documents:
            raise ValueError("No extractable text was found in the PDF.")

        chunks = split_documents(documents, chunk_size, chunk_overlap)
        PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            index_name=self.settings.pinecone_index_name,
            namespace=namespace or None,
        )
        return IngestResult(
            pages=page_count,
            chunks=len(chunks),
            namespace=namespace or None,
            index_name=self.settings.pinecone_index_name,
        )

    def ask(
        self,
        query: str,
        mode: Literal["advanced", "standard"] = "advanced",
        namespace: str | None = None,
        params: RAGParams | None = None,
    ) -> RAGResult:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        params = _validate_params(params or RAGParams(max_retries=self.settings.max_retries))
        self._prepare()
        vector_store = self._vector_store(namespace)
        return self._ask_with_fallback(query.strip(), mode, vector_store, params)

    def _ask_with_fallback(
        self,
        query: str,
        mode: Literal["advanced", "standard"],
        vector_store: PineconeVectorStore,
        params: RAGParams,
    ) -> RAGResult:
        provider = self.settings.llm_provider
        try:
            return self._ask_with_provider(query, mode, vector_store, params, provider)
        except Exception as exc:
            fallback_provider = self._fallback_provider(provider)
            if not params.use_fallback or not fallback_provider or not _is_rate_limit_error(exc):
                raise

            result = self._ask_with_provider(
                query, mode, vector_store, params, fallback_provider
            )
            result.metrics["fallback_used"] = True
            result.metrics["fallback_reason"] = str(exc)
            return result

    def _ask_with_provider(
        self,
        query: str,
        mode: Literal["advanced", "standard"],
        vector_store: PineconeVectorStore,
        params: RAGParams,
        provider: str,
    ) -> RAGResult:
        provider, model = self._provider_model(provider)
        llm = self._llm_for(provider, params.temperature)
        if mode == "standard":
            return StandardRAG(llm, vector_store, params, provider, model).ask(query)
        return AdvancedRAG(
            llm,
            self.reranker,
            vector_store,
            params,
            provider,
            model,
        ).ask(query)

    def _fallback_provider(self, provider: str) -> str | None:
        if provider == "groq" and self.settings.google_api_key:
            return "gemini"
        if provider == "gemini" and self.settings.groq_api_key:
            return "groq"
        return None

    def list_namespaces(self) -> list[dict[str, Any]]:
        self._prepare()
        index = Pinecone(api_key=self.settings.pinecone_api_key).Index(
            self.settings.pinecone_index_name
        )
        stats = index.describe_index_stats()
        namespaces = _index_value(stats, "namespaces") or {}
        items = []
        for name, detail in namespaces.items():
            vector_count = None
            if isinstance(detail, dict):
                vector_count = detail.get("vector_count")
            else:
                vector_count = getattr(detail, "vector_count", None)
            items.append(
                {
                    "namespace": name or "default",
                    "raw_namespace": name,
                    "vector_count": vector_count or 0,
                }
            )
        return sorted(items, key=lambda item: item["namespace"])

    def delete_namespace(self, namespace: str | None) -> None:
        self._prepare()
        index = Pinecone(api_key=self.settings.pinecone_api_key).Index(
            self.settings.pinecone_index_name
        )
        index.delete(delete_all=True, namespace=namespace or None)


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "quota" in text
