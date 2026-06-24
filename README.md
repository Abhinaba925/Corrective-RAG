# Corrective RAG Streamlit App

This project turns the original Corrective RAG notebook into a deployable
Streamlit app. It indexes PDFs into Pinecone and answers questions with either
Standard RAG or Advanced Corrective RAG.

## What It Does

- Upload a PDF and index its chunks in Pinecone.
- Ask questions against the indexed document set.
- Compare Standard RAG and Advanced CRAG side by side.
- Run a small batch evaluation set and download CSV metrics.
- Inspect performance timings for retrieval, reranking, grading, rewriting, and
  generation.
- Tune retrieval depth, chunking, retry count, reranking, query rewriting,
  temperature, and relevance thresholds from the sidebar.
- List and delete Pinecone namespaces from the app.
- Choose between:
  - **Advanced CRAG**: rewrite the query, retrieve broadly, rerank context,
    grade relevance, retry when needed, then answer.
  - **Standard RAG**: retrieve context and generate an answer.
- Fall back to the alternate provider when the active provider rate-limits and
  the alternate API key is configured.
- Deploy directly on Streamlit Community Cloud from GitHub.

## Stack

- **UI**: Streamlit
- **Pipeline**: Standard RAG and corrective RAG service logic
- **LLM**: Groq-hosted Qwen by default, or Google Gemini
- **Vector DB**: Pinecone Serverless
- **Embeddings**: `BAAI/bge-base-en-v1.5`
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **PDF parsing**: PyPDF

## Project Structure

```text
streamlit_app.py     Streamlit Cloud entry point
app/
  config.py          Environment configuration
  rag.py             RAG and CRAG pipeline logic
  main.py            Optional FastAPI entry point
  static/            Optional FastAPI browser UI
Corrective_RAG.ipynb Original notebook
requirements.txt     Python dependencies
.env.example         Local environment template
.streamlit/
  config.toml        Streamlit theme
  secrets.toml.example
```

## Secrets

For Streamlit Community Cloud, add these in **App settings -> Secrets**:

```toml
GOOGLE_API_KEY = "your-google-ai-studio-key"
GROQ_API_KEY = "your-groq-key"
PINECONE_API_KEY = "your-pinecone-key"
PINECONE_INDEX_NAME = "rag-index"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
LLM_PROVIDER = "groq"
GEMINI_MODEL = "gemini-2.5-flash"
GROQ_MODEL = "qwen/qwen3-32b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIMENSION = "768"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_RETRIES = "2"
APP_PASSWORD = ""
```

The default embedding model uses 768-dimensional vectors, so keep
`EMBEDDING_DIMENSION=768` unless you change the embedding model too. The app
creates the Pinecone index automatically if it does not exist.

Set `LLM_PROVIDER = "groq"` to use Groq. Set `LLM_PROVIDER = "gemini"` to use
Gemini again.

Set `APP_PASSWORD` only if you want the public Streamlit app to require a
password before use.

## Run Locally With Streamlit

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`, then run:

```bash
streamlit run streamlit_app.py
```

You can also use a local Streamlit secrets file:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
streamlit run streamlit_app.py
```

Do not commit `.env` or `.streamlit/secrets.toml`.

## Streamlit Tabs

- **Ask**: Run one question through Standard RAG or CRAG.
- **Compare**: Run the same question through both pipelines and compare latency,
  source counts, reranker scores, retry count, and no-answer detection.
- **Evaluate**: Paste one question per line, run both pipelines, and download a
  CSV summary.
- **Manage**: Refresh Pinecone namespace stats and delete a namespace after
  typing its exact name.
- **History**: Review the latest answers and metrics from the current session.

## Deploy To Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select **Create app**.
4. Choose this GitHub repository.
5. Set the main file path to `streamlit_app.py`.
6. Open **Advanced settings** and select Python `3.11`.
7. Paste the TOML secrets from the **Secrets**
   section above.
8. Deploy the app.

The first ingestion or Advanced CRAG query can take longer because Hugging Face
models are downloaded on demand.

## Optional FastAPI/Docker Run

The repo still includes the FastAPI entry point for container-based hosting:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

Docker:

```bash
docker build -t corrective-rag .
docker run --env-file .env -p 8080:8080 corrective-rag
```

## Notes

- Pinecone namespaces let you keep separate document collections in the same
  index. Leave namespace blank to use the default namespace.
- Uploaded PDFs are parsed, embedded, sent to Pinecone, and then removed from
  the app runtime.
- The notebook is still included for reference; deploy `streamlit_app.py`.
