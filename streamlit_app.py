from __future__ import annotations

import csv
import os
import tempfile
from io import StringIO
from pathlib import Path
from typing import Iterable

import streamlit as st
from dotenv import load_dotenv

from app.config import Settings
from app.rag import (
    RAGParams,
    RAGResult,
    RAGService,
    answer_word_count,
    source_page_overlap,
)


SERVICE_CACHE_VERSION = "rag-service-judge-v2"


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
    "APP_PASSWORD",
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
def get_service(cache_version: str) -> RAGService:
    del cache_version
    return RAGService(Settings.from_env())


def save_uploaded_pdf(uploaded_file) -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)


def normalize_namespace(value: str) -> str | None:
    stripped = value.strip()
    return stripped or None


def build_params(
    standard_top_k: int,
    advanced_broad_k: int,
    advanced_final_k: int,
    max_retries: int,
    temperature: float,
    relevance_threshold_enabled: bool,
    relevance_threshold: float,
    enable_reranking: bool,
    enable_query_rewrite: bool,
    enable_llm_grading: bool,
    use_fallback: bool,
) -> RAGParams:
    return RAGParams(
        standard_top_k=standard_top_k,
        advanced_broad_k=advanced_broad_k,
        advanced_final_k=advanced_final_k,
        max_retries=max_retries,
        temperature=temperature,
        relevance_threshold=relevance_threshold if relevance_threshold_enabled else None,
        enable_reranking=enable_reranking,
        enable_query_rewrite=enable_query_rewrite,
        enable_llm_grading=enable_llm_grading,
        use_fallback=use_fallback,
    )


def metric_value(result: RAGResult, key: str, default="-"):
    value = result.metrics.get(key)
    return default if value is None else value


def render_metric_cards(result: RAGResult) -> None:
    metrics = result.metrics
    row_one = st.columns(5)
    row_one[0].metric("Total", f"{metrics.get('total_ms', 0)} ms")
    row_one[1].metric("Retrieval", f"{metrics.get('retrieval_ms', 0)} ms")
    row_one[2].metric("Rerank", f"{metrics.get('rerank_ms', 0)} ms")
    row_one[3].metric("Grading", f"{metrics.get('grading_ms', 0)} ms")
    row_one[4].metric("Generation", f"{metrics.get('generation_ms', 0)} ms")

    row_two = st.columns(5)
    row_two[0].metric("Retrieved", metrics.get("retrieved_docs", 0))
    row_two[1].metric("Final docs", metrics.get("final_docs", 0))
    row_two[2].metric("Top score", metric_value(result, "top_score"))
    row_two[3].metric("Avg score", metric_value(result, "avg_score"))
    row_two[4].metric("Retries", result.retries)


def render_sources(result: RAGResult) -> None:
    if not result.sources:
        st.info("No sources returned.")
        return

    for index, source in enumerate(result.sources, start=1):
        page = source.get("page") or "?"
        score = source.get("score")
        score_text = f" - score {score}" if score is not None else ""
        title = f"Source {index} - Page {page}{score_text}"
        with st.expander(title, expanded=index == 1):
            if source.get("source"):
                st.caption(source["source"])
            st.write(source.get("content") or source.get("preview") or "")


def render_result(result: RAGResult, label: str | None = None) -> None:
    if label:
        st.subheader(label)

    st.caption(f"{result.mode.upper()} - {result.provider} - {result.model}")
    st.write(result.answer)
    st.download_button(
        "Download answer",
        result.answer,
        file_name=f"{result.mode}_answer.md",
        mime="text/markdown",
        key=f"download_{label or result.mode}_{id(result)}",
    )

    with st.expander("Performance metrics", expanded=True):
        render_metric_cards(result)
        st.json(result.metrics)

    if result.mode == "advanced":
        with st.expander("CRAG route", expanded=False):
            st.write(f"Rewritten query: {result.rewritten_query or '-'}")
            st.write(f"Retries: {result.retries}")
            st.write(f"Query rewrite triggered: {result.metrics.get('rewrite_triggered')}")
            st.write(f"Accepted context: {result.metrics.get('accepted_context')}")
            st.write(f"LLM grading used: {result.metrics.get('llm_grading_used')}")
            st.write(f"No-answer detected: {result.metrics.get('no_answer_detected')}")

    with st.expander("Tuning used", expanded=False):
        st.json(result.params)

    st.subheader("Sources")
    render_sources(result)


def comparison_rows(standard: RAGResult, advanced: RAGResult) -> list[dict[str, object]]:
    pairs = (
        ("Total ms", "total_ms"),
        ("Retrieval ms", "retrieval_ms"),
        ("Rerank ms", "rerank_ms"),
        ("Grading ms", "grading_ms"),
        ("Generation ms", "generation_ms"),
        ("Retrieved docs", "retrieved_docs"),
        ("Final docs", "final_docs"),
        ("Top score", "top_score"),
        ("Avg score", "avg_score"),
        ("No-answer detected", "no_answer_detected"),
        ("Fallback used", "fallback_used"),
        ("LLM grading used", "llm_grading_used"),
    )
    rows = [
        {
            "metric": label,
            "standard_rag": standard.metrics.get(key),
            "advanced_crag": advanced.metrics.get(key),
        }
        for label, key in pairs
    ]
    rows.append(
        {
            "metric": "Retries",
            "standard_rag": standard.retries,
            "advanced_crag": advanced.retries,
        }
    )
    rows.extend(
        [
            {
                "metric": "Answer words",
                "standard_rag": answer_word_count(standard.answer),
                "advanced_crag": answer_word_count(advanced.answer),
            },
            {
                "metric": "Source page overlap",
                "standard_rag": "-",
                "advanced_crag": source_page_overlap(standard, advanced),
            },
        ]
    )
    return rows


def rows_to_csv(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def handle_error(exc: Exception) -> None:
    message = str(exc)
    lowered = message.lower()
    if "rate limit" in lowered or "quota" in lowered or "429" in lowered:
        st.error("The model provider is rate-limiting this request.")
        st.caption(message)
    else:
        st.error(message)


def run_llm_judge(
    question: str,
    standard: RAGResult,
    advanced: RAGResult,
) -> dict[str, object]:
    if not hasattr(service, "judge_answers"):
        get_service.clear()
        raise RuntimeError(
            "The app is using an old cached RAG service. Reboot the Streamlit app "
            "once, then run the judge again."
        )
    return service.judge_answers(
        question,
        standard,
        advanced,
        temperature=0,
    )


def add_history(question: str, result: RAGResult) -> None:
    st.session_state.history.insert(
        0,
        {
            "question": question,
            "mode": result.mode,
            "provider": result.provider,
            "model": result.model,
            "answer": result.answer,
            "metrics": result.metrics,
            "sources": result.sources,
        },
    )
    st.session_state.history = st.session_state.history[:20]


load_dotenv()
hydrate_environment_from_streamlit(SECRET_KEYS)
settings = Settings.from_env()
service = get_service(SERVICE_CACHE_VERSION)

if "history" not in st.session_state:
    st.session_state.history = []

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.7rem; max-width: 1220px; }
      [data-testid="stMetricValue"] { font-size: 1.15rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Corrective RAG")

app_password = os.getenv("APP_PASSWORD")
if app_password and not st.session_state.get("authenticated"):
    with st.form("auth_form"):
        supplied_password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Unlock", type="primary")
        if submitted:
            if supplied_password == app_password:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")
    st.stop()

with st.sidebar:
    st.header("Status")
    if settings.configured:
        st.success("Configured")
    else:
        st.error("Missing configuration")
        messages = settings.configuration_errors + settings.missing_env
        st.code("\n".join(messages), language="text")

    st.caption(f"Index: {settings.pinecone_index_name}")
    st.caption(f"Provider: {settings.llm_provider}")
    st.caption(f"Model: {settings.active_llm_model}")

    st.header("Namespace")
    namespace = st.text_input("Namespace", value="", placeholder="default")

    st.header("Index PDF")
    uploaded_pdf = st.file_uploader("PDF", type=["pdf"])
    if uploaded_pdf and uploaded_pdf.size > 20 * 1024 * 1024:
        st.warning("Large PDFs can be slow on Streamlit Community Cloud.")

    with st.expander("Tuning", expanded=True):
        chunk_size = st.number_input(
            "Chunk size", min_value=200, max_value=2000, value=800, step=100
        )
        chunk_overlap = st.number_input(
            "Chunk overlap", min_value=0, max_value=500, value=100, step=25
        )
        standard_top_k = st.slider("Standard top k", 1, 30, 15)
        advanced_broad_k = st.slider("CRAG broad k", 5, 80, 40)
        advanced_final_k = st.slider(
            "CRAG final docs", 1, min(15, advanced_broad_k), min(5, advanced_broad_k)
        )
        max_retries = st.slider("CRAG max retries", 0, 5, settings.max_retries)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05)
        enable_reranking = st.checkbox("Rerank CRAG results", value=True)
        enable_query_rewrite = st.checkbox("Rewrite weak queries", value=True)
        enable_llm_grading = st.checkbox("LLM relevance grading", value=False)
        relevance_threshold_enabled = st.checkbox("Use score threshold", value=False)
        relevance_threshold = st.slider("Score threshold", -10.0, 10.0, 0.0, 0.25)
        use_fallback = st.checkbox("Fallback provider on rate limit", value=False)

    params = build_params(
        standard_top_k=int(standard_top_k),
        advanced_broad_k=int(advanced_broad_k),
        advanced_final_k=int(advanced_final_k),
        max_retries=int(max_retries),
        temperature=float(temperature),
        relevance_threshold_enabled=bool(relevance_threshold_enabled),
        relevance_threshold=float(relevance_threshold),
        enable_reranking=bool(enable_reranking),
        enable_query_rewrite=bool(enable_query_rewrite),
        enable_llm_grading=bool(enable_llm_grading),
        use_fallback=bool(use_fallback),
    )

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
                    source_name=uploaded_pdf.name,
                )
            st.success(f"Indexed {result.chunks} chunks from {result.pages} pages.")
        except Exception as exc:
            handle_error(exc)
        finally:
            temp_path.unlink(missing_ok=True)

ask_tab, compare_tab, evaluate_tab, manage_tab, history_tab = st.tabs(
    ["Ask", "Compare", "Evaluate", "Manage", "History"]
)

with ask_tab:
    mode_label = st.radio(
        "Mode",
        ["Advanced CRAG", "Standard RAG"],
        horizontal=True,
        label_visibility="collapsed",
    )
    question = st.text_area(
        "Question",
        height=140,
        placeholder="Ask a question about the indexed PDF...",
        key="ask_question",
    )
    ask_clicked = st.button(
        "Ask",
        type="primary",
        disabled=not settings.configured or not question.strip(),
        key="ask_button",
    )
    if ask_clicked:
        mode = "advanced" if mode_label == "Advanced CRAG" else "standard"
        try:
            with st.spinner("Running retrieval and generation"):
                result = service.ask(
                    question,
                    mode=mode,
                    namespace=normalize_namespace(namespace),
                    params=params,
                )
            add_history(question, result)
            render_result(result)
        except Exception as exc:
            handle_error(exc)

with compare_tab:
    compare_question = st.text_area(
        "Question",
        height=120,
        placeholder="Run the same question through Standard RAG and CRAG...",
        key="compare_question",
    )
    run_compare_judge = st.checkbox(
        "Run LLM judge",
        value=False,
        key="compare_judge",
    )
    compare_clicked = st.button(
        "Compare RAG vs CRAG",
        type="primary",
        disabled=not settings.configured or not compare_question.strip(),
    )
    if compare_clicked:
        try:
            with st.spinner("Running Standard RAG"):
                standard = service.ask(
                    compare_question,
                    mode="standard",
                    namespace=normalize_namespace(namespace),
                    params=params,
                )
            with st.spinner("Running Advanced CRAG"):
                advanced = service.ask(
                    compare_question,
                    mode="advanced",
                    namespace=normalize_namespace(namespace),
                    params=params,
                )
            add_history(compare_question, standard)
            add_history(compare_question, advanced)
            st.subheader("Metric comparison")
            rows = comparison_rows(standard, advanced)
            st.dataframe(rows, hide_index=True, use_container_width=True)
            st.download_button(
                "Download metrics CSV",
                rows_to_csv(rows),
                file_name="rag_crag_comparison.csv",
                mime="text/csv",
            )

            if run_compare_judge:
                try:
                    with st.spinner("Running LLM judge"):
                        judgement = run_llm_judge(
                            compare_question,
                            standard,
                            advanced,
                        )
                    st.subheader("LLM judge")
                    judge_cols = st.columns(4)
                    judge_cols[0].metric("Winner", judgement.get("winner", "-"))
                    judge_cols[1].metric(
                        "Standard", judgement.get("standard_score", "-")
                    )
                    judge_cols[2].metric(
                        "CRAG", judgement.get("advanced_score", "-")
                    )
                    judge_cols[3].metric("Judge", judgement.get("model", "-"))
                    st.write(judgement.get("rationale", ""))
                except Exception as judge_exc:
                    st.warning(f"LLM judge failed: {judge_exc}")

            left, right = st.columns(2, gap="large")
            with left:
                render_result(standard, "Standard RAG")
            with right:
                render_result(advanced, "Advanced CRAG")
        except Exception as exc:
            handle_error(exc)

with evaluate_tab:
    eval_questions = st.text_area(
        "Questions",
        height=180,
        placeholder="One question per line",
        key="eval_questions",
    )
    eval_limit = st.number_input("Limit", min_value=1, max_value=20, value=5)
    run_eval_judge = st.checkbox(
        "Run LLM judge for each question",
        value=False,
        key="eval_judge",
    )
    eval_clicked = st.button(
        "Run evaluation",
        type="primary",
        disabled=not settings.configured or not eval_questions.strip(),
    )
    if eval_clicked:
        questions = [
            item.strip()
            for item in eval_questions.splitlines()
            if item.strip()
        ][: int(eval_limit)]
        rows: list[dict[str, object]] = []
        progress = st.progress(0)
        try:
            for index, item in enumerate(questions, start=1):
                with st.spinner(f"Evaluating {index} of {len(questions)}"):
                    standard = service.ask(
                        item,
                        mode="standard",
                        namespace=normalize_namespace(namespace),
                        params=params,
                    )
                    advanced = service.ask(
                        item,
                        mode="advanced",
                        namespace=normalize_namespace(namespace),
                        params=params,
                    )
                judgement = None
                if run_eval_judge:
                    try:
                        with st.spinner(f"Judging {index} of {len(questions)}"):
                            judgement = run_llm_judge(
                                item,
                                standard,
                                advanced,
                            )
                    except Exception as judge_exc:
                        judgement = {"error": str(judge_exc)}
                rows.append(
                    {
                        "question": item,
                        "standard_total_ms": standard.metrics.get("total_ms"),
                        "crag_total_ms": advanced.metrics.get("total_ms"),
                        "crag_retries": advanced.retries,
                        "standard_sources": len(standard.sources),
                        "crag_sources": len(advanced.sources),
                        "crag_top_score": advanced.metrics.get("top_score"),
                        "standard_no_answer": standard.metrics.get("no_answer_detected"),
                        "crag_no_answer": advanced.metrics.get("no_answer_detected"),
                        "source_page_overlap": source_page_overlap(standard, advanced),
                        "standard_answer_words": answer_word_count(standard.answer),
                        "crag_answer_words": answer_word_count(advanced.answer),
                        "judge_winner": (
                            judgement.get("winner") if judgement else None
                        ),
                        "judge_standard_score": (
                            judgement.get("standard_score") if judgement else None
                        ),
                        "judge_crag_score": (
                            judgement.get("advanced_score") if judgement else None
                        ),
                        "judge_rationale": (
                            judgement.get("rationale")
                            if judgement
                            else None
                        ),
                        "judge_error": judgement.get("error") if judgement else None,
                        "standard_answer": standard.answer[:240],
                        "crag_answer": advanced.answer[:240],
                    }
                )
                progress.progress(index / len(questions))
            st.dataframe(rows, hide_index=True, use_container_width=True)
            st.download_button(
                "Download evaluation CSV",
                rows_to_csv(rows),
                file_name="rag_evaluation.csv",
                mime="text/csv",
            )
        except Exception as exc:
            handle_error(exc)

with manage_tab:
    col_a, col_b = st.columns([0.5, 0.5], gap="large")
    with col_a:
        if st.button("Refresh namespaces", disabled=not settings.configured):
            try:
                st.session_state.namespaces = service.list_namespaces()
            except Exception as exc:
                handle_error(exc)

        namespaces = st.session_state.get("namespaces", [])
        if namespaces:
            st.dataframe(namespaces, hide_index=True, use_container_width=True)
        else:
            st.info("No namespace data loaded.")

    with col_b:
        namespaces = st.session_state.get("namespaces", [])
        choices = [item["namespace"] for item in namespaces]
        if choices:
            selected = st.selectbox("Namespace to delete", choices)
            confirm = st.text_input("Type namespace name to confirm")
            delete_clicked = st.button(
                "Delete namespace",
                disabled=confirm != selected,
            )
            if delete_clicked:
                try:
                    raw = next(
                        item["raw_namespace"]
                        for item in namespaces
                        if item["namespace"] == selected
                    )
                    service.delete_namespace(raw)
                    st.success(f"Deleted namespace {selected}.")
                    st.session_state.namespaces = service.list_namespaces()
                except Exception as exc:
                    handle_error(exc)
        else:
            st.info("Refresh namespaces before deleting.")

with history_tab:
    if not st.session_state.history:
        st.info("No history yet.")
    else:
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()
        for index, item in enumerate(st.session_state.history, start=1):
            title = f"{index}. {item['mode']} - {item['question'][:80]}"
            with st.expander(title, expanded=index == 1):
                st.caption(f"{item['provider']} - {item['model']}")
                st.write(item["answer"])
                st.json(item["metrics"])
