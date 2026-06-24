from __future__ import annotations

import os
import tempfile
from html import escape
from pathlib import Path
from typing import Iterable

import streamlit as st
from dotenv import load_dotenv

from app.config import Settings
from app.rag import RAGService


st.set_page_config(
    page_title="Corrective RAG",
    layout="wide",
    initial_sidebar_state="expanded",
)

SECRET_KEYS = (
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "PINECONE_API_KEY",
    "PINECONE_INDEX_NAME",
    "PINECONE_CLOUD",
    "PINECONE_REGION",
    "LLM_PROVIDER",
    "GEMINI_MODEL",
    "GROQ_MODEL",
    "GROQ_BASE_URL",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
    "RERANKER_MODEL",
    "MAX_RETRIES",
)


def hydrate_environment_from_streamlit(keys: Iterable[str]) -> None:
    try:
        secrets = st.secrets
        for key in keys:
            if key in secrets and str(secrets[key]).strip():
                os.environ[key] = str(secrets[key])
    except FileNotFoundError:
        return


@st.cache_resource(show_spinner=False)
def get_service() -> RAGService:
    return RAGService(Settings.from_env())


def save_uploaded_pdf(uploaded_file) -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)


def normalize_namespace(value: str) -> str | None:
    stripped = value.strip()
    return stripped or None


load_dotenv()
hydrate_environment_from_streamlit(SECRET_KEYS)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; max-width: 1180px; }
      [data-testid="stMetricValue"] { font-size: 1.35rem; }
      .source-box {
        border: 1px solid rgba(49, 51, 63, 0.16);
        border-radius: 8px;
        padding: 0.8rem 0.95rem;
        margin-bottom: 0.75rem;
      }
      .source-title {
        font-size: 0.86rem;
        font-weight: 700;
        margin-bottom: 0.45rem;
      }
      .source-preview {
        color: rgba(49, 51, 63, 0.76);
        font-size: 0.9rem;
        line-height: 1.45;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

settings = Settings.from_env()
service = get_service()

st.title("Corrective RAG")

with st.sidebar:
    st.header("Index")

    if settings.configured:
        st.success("Configured")
    else:
        st.error("Missing secrets")
        messages = settings.configuration_errors + settings.missing_env
        st.code("\n".join(messages), language="text")

    st.caption(f"Pinecone index: {settings.pinecone_index_name}")
    st.caption(f"LLM provider: {settings.llm_provider}")
    namespace = st.text_input("Namespace", value="", placeholder="default")
    chunk_size = st.number_input(
        "Chunk size", min_value=200, max_value=2000, value=800, step=100
    )
    chunk_overlap = st.number_input(
        "Chunk overlap", min_value=0, max_value=500, value=100, step=25
    )
    uploaded_pdf = st.file_uploader("PDF", type=["pdf"])

    ingest_clicked = st.button(
        "Ingest PDF",
        type="primary",
        use_container_width=True,
        disabled=not settings.configured or uploaded_pdf is None,
    )

    if ingest_clicked and uploaded_pdf is not None:
        temp_path = save_uploaded_pdf(uploaded_pdf)
        try:
            with st.spinner("Indexing PDF in Pinecone"):
                result = service.ingest_pdf(
                    temp_path,
                    namespace=normalize_namespace(namespace),
                    chunk_size=int(chunk_size),
                    chunk_overlap=int(chunk_overlap),
                )
            st.success(
                f"Indexed {result.chunks} chunks from {result.pages} pages."
            )
        except Exception as exc:
            st.error(str(exc))
        finally:
            temp_path.unlink(missing_ok=True)

left, right = st.columns([0.68, 0.32], gap="large")

with left:
    mode_label = st.segmented_control(
        "Mode",
        ["Advanced CRAG", "Standard RAG"],
        default="Advanced CRAG",
        label_visibility="collapsed",
    )
    question = st.text_area(
        "Question",
        height=150,
        placeholder="Ask a question about the indexed PDF...",
    )
    ask_clicked = st.button(
        "Ask",
        type="primary",
        disabled=not settings.configured or not question.strip(),
    )

with right:
    st.metric("LLM", settings.active_llm_model)
    st.metric("Provider", settings.llm_provider)
    st.metric("Embedding dimension", settings.embedding_dimension)
    st.metric("Max retries", settings.max_retries)

if ask_clicked:
    mode = "advanced" if mode_label == "Advanced CRAG" else "standard"
    try:
        with st.spinner("Running retrieval and generation"):
            response = service.ask(
                question,
                mode=mode,
                namespace=normalize_namespace(namespace),
            )

        st.subheader("Answer")
        st.write(response.answer)

        if response.mode == "advanced":
            with st.expander("CRAG route", expanded=False):
                st.write(f"Rewritten query: {response.rewritten_query or question}")
                st.write(f"Retries: {response.retries}")

        if response.sources:
            st.subheader("Sources")
            for index, source in enumerate(response.sources, start=1):
                page = source.get("page") or "?"
                score = source.get("score")
                score_text = f" - score {score}" if score is not None else ""
                preview = escape(source.get("preview", ""))
                st.markdown(
                    f"""
                    <div class="source-box">
                      <div class="source-title">Source {index} - Page {page}{score_text}</div>
                      <div class="source-preview">{preview}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    except Exception as exc:
        st.error(str(exc))
elif not settings.configured:
    st.info("Add Streamlit secrets before running ingestion or questions.")
