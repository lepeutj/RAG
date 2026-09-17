# RAG System

A compact, production-oriented Retrieval-Augmented Generation (RAG) service designed to demonstrate the complete path from source documents to cited, grounded answers. It exposes a FastAPI API, stores embeddings in persistent ChromaDB, and supports local or OpenAI embeddings with Anthropic, OpenAI, or a local llama.cpp server for generation.

## What this project demonstrates

- A recursive text chunker implemented in the repository, with semantic separators and overlap.
- Clear boundaries between document loading, chunking, embeddings, vector storage, retrieval, and generation.
- Swappable embedding and LLM providers configured through environment variables rather than application code.
- A persistent local vector store suitable for a small single-node deployment.
- An HTTP API for ingestion, querying, service health, and index statistics.
- A small, realistic support-policy corpus for a repeatable end-to-end demo.

## Architecture

```text
Source documents
      |
      v
Document loader --> Recursive chunker --> Embedding provider --> ChromaDB
                                                                  |
User question --> Query embedding --> Retriever ------------------+
                                                                  |
                                                                  v
                                                        LLM with retrieved context
                                                                  |
                                                                  v
                                                   Answer, source files, and chunks
```

## Quick start

### 1. Configure the project

```bash
git clone https://github.com/lepeutj/RAG.git
cd RAG
cp .env.example .env
```

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in `.env`. Local embeddings are the default, so no embedding API key is required unless you set `EMBEDDING_PROVIDER=openai`.

### 2. Start the API with Docker

```bash
docker compose up -d --build
curl http://localhost:8000/api/v1/health
```

Expected response:

```json
{"status":"ok"}
```

The first startup downloads the sentence-transformers model and can take a few minutes. The model cache and ChromaDB data are kept in Docker volumes.

### 3. Ingest the demonstration corpus

```bash
docker compose exec rag-api python scripts/ingest.py --path /app/data/documents
curl http://localhost:8000/api/v1/stats
```

The corpus contains policies for refunds, delivery, and subscriptions. The `/stats` response should show a non-zero `chunks_indexed` value.

### 4. Run a grounded query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question":"How long do I have to request a refund?"}'
```

The answer should state that the refund request window is 30 days and list `refund-policy.md` in `sources`. The wording may vary because an LLM generates the response.

Try these additional queries to verify retrieval across the corpus:

| Question | Expected source |
| --- | --- |
| `When is express delivery available?` | `delivery-policy.md` |
| `How do I cancel my subscription?` | `subscription-policy.md` |
| `Are return shipping costs refunded for defective items?` | `refund-policy.md` |

### Use llama.cpp for fully local generation

The default embedding provider is already local. To keep answer generation local as well, run a GGUF model with llama.cpp on the host machine:

```bash
llama-server -m /absolute/path/to/model.gguf --host 127.0.0.1 --port 8080
```

Then set the following values in `.env`:

```dotenv
LLM_PROVIDER=llama_cpp
LLAMA_CPP_MODEL=local-model
```

For local development, the default `LLAMA_CPP_BASE_URL=http://127.0.0.1:8080/v1` is correct. If the RAG API runs in Docker, a llama.cpp server bound to the host's `127.0.0.1` is not reachable from the container. Run both services on a private Docker network, or run the RAG API directly on the host instead. Do not expose an unauthenticated llama.cpp port to the public internet.

llama.cpp's server provides an OpenAI-compatible chat-completions endpoint, which is why the project can use the existing OpenAI Python client without sending requests to OpenAI. See the [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

## API overview

| Endpoint | Purpose |
| --- | --- |
| `GET /api/v1/health` | Liveness endpoint; always public for deployment checks. |
| `GET /api/v1/stats` | Index size and selected providers. |
| `GET /api/v1/documents` | List indexed documents and their stable IDs. |
| `POST /api/v1/ingest` | Upload one `.txt`, `.md`, or `.pdf` document. |
| `POST /api/v1/documents/{document_id}/reindex` | Rebuild one document's chunks and embeddings. |
| `DELETE /api/v1/documents/{document_id}` | Remove one document from the index and managed upload store. |
| `POST /api/v1/query` | Retrieve context and generate an answer. |

Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

If `API_KEY` is set in `.env`, provide it for protected endpoints:

```bash
curl -H "x-api-key: your-secret" http://localhost:8000/api/v1/stats
```

### Document lifecycle

Files uploaded through the API are copied to `storage/documents/` and assigned
a deterministic ID derived from their logical filename. Uploading the same
filename again replaces its indexed chunks instead of accumulating duplicates.

Use the document list to obtain an ID, then reindex or delete it:

```bash
curl http://localhost:8000/api/v1/documents
curl -X POST http://localhost:8000/api/v1/documents/<document_id>/reindex
curl -X DELETE http://localhost:8000/api/v1/documents/<document_id>
```

Deleting a document uploaded through the API also removes its managed source
file. Documents indexed directly from a local directory are removed from the
vector index only; their original local files are left untouched.

If you created the ChromaDB index with an earlier version of this project,
rebuild it once to add document-lifecycle metadata:

```bash
python scripts/ingest.py --path data/documents --reset
```

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/ingest.py --path data/documents
uvicorn src.main:app --reload
```

Run the test suite with:

```bash
pytest -v
```

## Retrieval evaluation

The checked-in evaluation dataset contains representative questions, their expected source documents, and reference answers. The evaluation command measures retrieval only, so it does not need an Anthropic, OpenAI, or llama.cpp server.

First ingest the demonstration corpus, then run:

```bash
python scripts/evaluate_retrieval.py --top-k 3 --json-output evaluation-results.json
```

The report includes two metrics:

- **Source recall@k**: the share of expected source documents returned among the top `k` chunks.
- **Mean reciprocal rank (MRR)**: how high the first expected source appears; `1.0` means first place.

Use the JSON report as a baseline before changing chunking, metadata, hybrid search, or reranking. A future answer-quality evaluator can compare generated responses with the `expected_answer` field in the same dataset.

## Deployment notes

Docker Compose is the recommended route for a small VPS. It mounts `storage/` for persistent ChromaDB data and `data/` for source documents. The repository also includes example systemd and nginx configurations under `deploy/`.

This is a technical demonstration, not yet a multi-tenant production system. The next planned improvements are stable document lifecycle management, retrieval evaluation, hybrid retrieval/reranking, CI, and operational metrics.

## Project structure

```text
src/
  api/           # FastAPI routes and schemas
  ingestion/     # loaders and recursive chunking
  embeddings/    # local and OpenAI embedding providers
  vectorstore/   # persistent ChromaDB wrapper
  retrieval/     # similarity retrieval
  generation/    # Anthropic, OpenAI, and llama.cpp LLM providers
scripts/         # command-line ingestion
data/documents/  # reproducible demo corpus
tests/           # unit and API contract tests
deploy/          # nginx and systemd examples
```
