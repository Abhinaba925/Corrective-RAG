from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel, Field
from pypdf import PdfReader
from sentence_transformers import CrossEncoder
from typing_extensions import TypedDict

from app.config import Settings


class RAGState(TypedDict):
    query: str
    context: list[Document]
    response: str


class AdvancedRAGState(TypedDict):
    original_query: str
    current_query: str
    context: list[Document]
    response: str
    retry_count: int


class GradeDocuments(BaseModel):
    """Binary relevance decision used by the CRAG router."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


@dataclass
class RAGResult:
    answer: str
    sources: list[dict[str, Any]]
    mode: str
    rewritten_query: str | None = None
    retries: int = 0


@dataclass
class IngestResult:
    pages: int
    chunks: int
    namespace: str | None
    index_name: str


def _require_config(settings: Settings) -> None:
    missing = settings.missing_env
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variables: {joined}")


def _install_env(settings: Settings) -> None:
    # LangChain integrations also read these environment variables internally.
    if settings.google_api_key:
        os.environ["GOOGLE_API_KEY"] = settings.google_api_key
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


def extract_pdf_documents(pdf_path: Path) -> tuple[list[Document], int]:
    reader = PdfReader(str(pdf_path))
    documents: list[Document] = []

    for index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"page": index + 1, "source": pdf_path.name},
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
            }
        )
    return sources


class StandardRAG:
    def __init__(self, llm: ChatGoogleGenerativeAI, vector_store: PineconeVectorStore):
        self.llm = llm
        self.vector_store = vector_store
        self.workflow = self._build_graph()

    def _build_graph(self):
        def retrieve(state: RAGState) -> dict[str, Any]:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 15})
            return {"context": retriever.invoke(state["query"])}

        def generate(state: RAGState) -> dict[str, Any]:
            context_str = "\n\n".join(
                [
                    f"[Page {doc.metadata.get('page', '?')}]: {doc.page_content}"
                    for doc in state["context"]
                ]
            )
            prompt = (
                "Answer the user's question using only the context below. "
                "If the context lacks the answer, state that it is missing.\n\n"
                f"Context:\n{context_str}\n\n"
                f"Question: {state['query']}\nAnswer:"
            )
            return {"response": self.llm.invoke(prompt).content}

        graph = StateGraph(RAGState)
        graph.add_node("retrieve", retrieve)
        graph.add_node("generate", generate)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)
        return graph.compile()

    def ask(self, query: str) -> RAGResult:
        output = self.workflow.invoke({"query": query, "context": [], "response": ""})
        return RAGResult(
            answer=output["response"],
            sources=serialize_sources(output["context"]),
            mode="standard",
        )


class AdvancedRAG:
    """Corrective RAG with query rewriting, reranking, and relevance grading."""

    def __init__(
        self,
        llm: ChatGoogleGenerativeAI,
        reranker: CrossEncoder,
        vector_store: PineconeVectorStore,
        max_retries: int,
    ):
        self.llm = llm
        self.grader_llm = llm.with_structured_output(GradeDocuments)
        self.reranker = reranker
        self.vector_store = vector_store
        self.max_retries = max_retries
        self.workflow = self._build_graph()

    def _build_graph(self):
        def rewrite_query(state: AdvancedRAGState) -> dict[str, Any]:
            prompt = (
                "Extract keywords and concepts from this question to optimize it "
                "for vector search. Do not answer it.\n"
                f"Question: {state['original_query']}\nOptimized:"
            )
            rewritten = self.llm.invoke(prompt).content.strip()
            return {
                "current_query": rewritten or state["original_query"],
                "retry_count": state["retry_count"] + 1,
            }

        def retrieve_and_rerank(state: AdvancedRAGState) -> dict[str, Any]:
            query = state["current_query"] or state["original_query"]
            broad_docs = self.vector_store.as_retriever(search_kwargs={"k": 40}).invoke(
                query
            )
            if not broad_docs:
                return {"context": []}

            pairs = [[query, doc.page_content] for doc in broad_docs]
            scores = self.reranker.predict(pairs)
            for doc, score in zip(broad_docs, scores):
                doc.metadata["relevance_score"] = float(score)

            broad_docs.sort(
                key=lambda item: item.metadata["relevance_score"], reverse=True
            )
            return {"context": broad_docs[:5]}

        def grade_context(state: AdvancedRAGState) -> dict[str, Any]:
            return {"context": state["context"]}

        def router(state: AdvancedRAGState) -> Literal["generate", "rewrite_query"]:
            if state["retry_count"] >= self.max_retries:
                return "generate"
            if not state["context"]:
                return "rewrite_query"

            context_str = "\n\n".join([doc.page_content for doc in state["context"]])
            prompt = (
                "Does this context relate to the question? "
                "Return only the structured binary score.\n"
                f"Q: {state['current_query']}\nContext: {context_str}"
            )
            grade = self.grader_llm.invoke(prompt)
            if grade.binary_score.strip().lower().startswith("yes"):
                return "generate"
            return "rewrite_query"

        def generate(state: AdvancedRAGState) -> dict[str, Any]:
            context_str = "\n\n".join(
                [
                    f"[Page {doc.metadata.get('page', '?')}]: {doc.page_content}"
                    for doc in state["context"]
                ]
            )
            prompt = (
                "Answer using ONLY the provided context. If unknown, say so.\n\n"
                f"Question: {state['original_query']}\n\n"
                f"Context:\n{context_str}\n\nAnswer:"
            )
            return {"response": self.llm.invoke(prompt).content}

        graph = StateGraph(AdvancedRAGState)
        graph.add_node("rewrite_query", rewrite_query)
        graph.add_node("retrieve_and_rerank", retrieve_and_rerank)
        graph.add_node("grade_context", grade_context)
        graph.add_node("generate", generate)
        graph.add_edge(START, "rewrite_query")
        graph.add_edge("rewrite_query", "retrieve_and_rerank")
        graph.add_edge("retrieve_and_rerank", "grade_context")
        graph.add_conditional_edges("grade_context", router)
        graph.add_edge("generate", END)
        return graph.compile()

    def ask(self, query: str) -> RAGResult:
        output = self.workflow.invoke(
            {
                "original_query": query,
                "current_query": "",
                "context": [],
                "response": "",
                "retry_count": 0,
            },
            config={"recursion_limit": self.max_retries * 4 + 8},
        )
        return RAGResult(
            answer=output["response"],
            sources=serialize_sources(output["context"]),
            mode="advanced",
            rewritten_query=output["current_query"],
            retries=output["retry_count"],
        )


class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = Lock()
        self._embeddings: HuggingFaceEmbeddings | None = None
        self._llm: ChatGoogleGenerativeAI | None = None
        self._reranker: CrossEncoder | None = None
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
    def embeddings(self) -> HuggingFaceEmbeddings:
        with self._lock:
            if self._embeddings is None:
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.settings.embedding_model,
                    encode_kwargs={"normalize_embeddings": True},
                )
            return self._embeddings

    @property
    def llm(self) -> ChatGoogleGenerativeAI:
        with self._lock:
            if self._llm is None:
                self._llm = ChatGoogleGenerativeAI(
                    model=self.settings.gemini_model,
                    temperature=0,
                    google_api_key=self.settings.google_api_key,
                )
            return self._llm

    @property
    def reranker(self) -> CrossEncoder:
        with self._lock:
            if self._reranker is None:
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
    ) -> IngestResult:
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self._prepare()
        documents, page_count = extract_pdf_documents(pdf_path)
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
    ) -> RAGResult:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        self._prepare()
        vector_store = self._vector_store(namespace)
        if mode == "standard":
            return StandardRAG(self.llm, vector_store).ask(query.strip())
        return AdvancedRAG(
            self.llm,
            self.reranker,
            vector_store,
            self.settings.max_retries,
        ).ask(query.strip())
