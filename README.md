# Corrective RAG Cloud App

This project turns the original Corrective RAG notebook into a deployable web app.
It provides a browser UI and FastAPI backend for indexing PDFs into Pinecone and
asking questions with either Standard RAG or Advanced Corrective RAG.

## What It Does

- Upload a PDF from the browser and index its chunks in Pinecone.
- Ask questions against the indexed document set.
- Choose between:
  - **Standard RAG**: retrieve context and generate an answer.
  - **Advanced CRAG**: rewrite the query, retrieve broadly, rerank with a
    cross-encoder, grade context relevance, retry when needed, then answer.
- Deploy as a single Docker web service.

## Stack

- **API/UI**: FastAPI, vanilla HTML/CSS/JS
- **Workflow**: LangGraph
- **LLM**: Google Gemini via `langchain-google-genai`
- **Vector DB**: Pinecone Serverless
- **Embeddings**: `BAAI/bge-base-en-v1.5`
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **PDF parsing**: PyPDF

## Project Structure

```text
app/
  config.py          Environment configuration
  rag.py             RAG and CRAG pipeline logic
  main.py            FastAPI routes
  static/            Browser UI
Corrective_RAG.ipynb Original notebook
Dockerfile           Container build
render.yaml          Render blueprint
requirements.txt     Python dependencies
.env.example         Required environment variables
```

## Required Environment Variables

Create a `.env` locally or configure these in your cloud provider:

```bash
GOOGLE_API_KEY=your-google-ai-studio-key
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=rag-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
GEMINI_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DIMENSION=768
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
MAX_RETRIES=2
```

The app creates the Pinecone index automatically if it does not exist. The
default embedding model uses 768-dimensional vectors, so keep
`EMBEDDING_DIMENSION=768` unless you change the embedding model too.

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`, then run:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

Open `http://localhost:8080`.

## Run With Docker

```bash
docker build -t corrective-rag .
docker run --env-file .env -p 8080:8080 corrective-rag
```

Open `http://localhost:8080`.

## Deploy To Render

1. Push this repository to GitHub.
2. In Render, create a new **Blueprint** from the repository, or create a new
   Docker web service manually.
3. Use the included `render.yaml` if deploying as a Blueprint.
4. Add `GOOGLE_API_KEY` and `PINECONE_API_KEY` as secret environment variables.
5. Deploy the service.

The first query or ingestion can take longer because Hugging Face models are
downloaded on demand. Use a plan with enough memory for `sentence-transformers`
and the cross-encoder reranker.

## API

Health:

```bash
curl http://localhost:8080/api/health
```

Ingest a PDF:

```bash
curl -X POST http://localhost:8080/api/ingest \
  -F "file=@/path/to/document.pdf" \
  -F "namespace=default"
```

Ask a question:

```bash
curl -X POST http://localhost:8080/api/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What are the main observations?","mode":"advanced","namespace":"default"}'
```

## Notes

- Pinecone namespaces let you keep separate document collections in the same
  index. Leave namespace blank to use the default namespace.
- Uploaded PDFs are parsed, embedded, sent to Pinecone, and then removed from
  the app container.
- The notebook is still included for reference, but the deployable entry point
  is `app.main:app`.
