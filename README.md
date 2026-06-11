# Corrective RAG (CRAG) Pipeline

This repository contains a Retrieval-Augmented Generation (RAG) system with an advanced corrective architecture (CRAG). It leverages LangGraph to create a cyclic workflow that evaluates retrieved documents for relevance, reranks them using a cross-encoder, and rewrites search queries if the initial context is insufficient.

## Architecture

The system provides two pipelines:
1. **Standard RAG**: A baseline linear pipeline that retrieves chunks via cosine similarity and generates an answer.
2. **Advanced CRAG**: A stateful workflow utilizing LangGraph. 
   - **Retrieval & Reranking**: Fetches a broad set of documents from Pinecone and reranks them using `ms-marco-MiniLM-L-6-v2`.
   - **Grading**: Uses a structured LLM output to perform a binary relevance check on the reranked documents.
   - **Routing & Rewriting**: If documents are deemed irrelevant, the query is optimized and rewritten, triggering a new search cycle (capped at a maximum retry limit).
   - **Generation**: Constructs the final response based strictly on validated context.

## Technology Stack

- **Orchestration**: LangChain, LangGraph
- **Vector Database**: Pinecone (Serverless)
- **LLM**: Google Gemini (gemini-2.5-flash)
- **Embeddings**: HuggingFace (`BAAI/bge-base-en-v1.5`)
- **Reranker**: Sentence Transformers Cross-Encoder
- **Document Processing**: PyPDF

## Prerequisites

You need active accounts and API keys for:
- Google AI Studio (Gemini)
- Pinecone

## Installation
Create and activate a virtual environment:

Bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the required dependencies:

Bash
pip install -r requirements.txt
Configure environment variables. Copy the example environment file and add your keys:

Bash
cp .env.example .env
Usage
Place the PDF you wish to analyze (e.g., RBI_Annual_Report.pdf) into the data/ directory.

Open the Jupyter Notebook notebooks/Corrective_RAG.ipynb.

If running for the first time, uncomment the ingestion lines in the execution block to parse, chunk, and embed your PDF into Pinecone.

Execute the notebook to initialize the LangGraph pipeline and pass your query.

State Management
The Advanced CRAG pipeline maintains state via the AdvancedRAGState TypedDict, which tracks:

original_query: The user's initial input.

current_query: The active search string (can be mutated by the rewrite node).

context: The filtered and reranked LangChain documents.

retry_count: Prevents infinite routing loops.

1. Clone the repository:
```bash
git clone [https://github.com/yourusername/corrective-rag-pipeline.git](https://github.com/yourusername/corrective-rag-pipeline.git)
cd corrective-rag-pipeline
