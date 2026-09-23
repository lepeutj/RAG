# RAG System

A compact, production-oriented Retrieval-Augmented Generation (RAG) service designed to demonstrate the complete path from source documents to cited, grounded answers. It exposes a FastAPI API, stores embeddings in ChromaDB, and supports local or OpenAI embeddings with Anthropic, OpenAI, or a local llama.cpp server for generation.

## What this project demonstrates

- A recursive text chunker implemented in the repository, with semantic separators and overlap.
- Clear boundaries between document loading, chunking, embeddings, vector storage, retrieval, and generation.
- Swappable embedding and LLM providers configured through environment variables rather than application code.
- Embedded Chroma for local development and an HTTP Chroma sidecar for containers.
- An HTTP API for ingestion, querying, service health, and index statistics.
- A small, realistic support-policy corpus for a repeatable end-to-end demo.
- A labeled RAG evaluation set with retrieval scoring, answer review, and paired run comparison.

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

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` and `PUBLIC_DEMO_QUERY=true` in `.env` to use the browser and query endpoint locally. Local embeddings are the default, so no embedding API key is required unless you set `EMBEDDING_PROVIDER=openai`.

### 2. Start the API with Docker

```bash
docker compose up -d --build
curl http://localhost:8000/api/v1/health
```

Expected response:

```json
{"status":"ok"}
```

The first startup downloads the sentence-transformers model and can take a few minutes. Docker Compose starts Chroma as an internal service and rebuilds the demo index when the API starts.

The bundled corpus contains policies for refunds, delivery, and subscriptions and is ingested automatically when the API container starts.

### 3. Run a grounded query

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

## Public demo security

The target deployment is Google Cloud Run. The public service exposes the browser page, health check, static files, and query endpoint. Administrative document operations are disabled by default and return `404`; enabling them requires both `ADMIN_API_ENABLED=true` and an API key. Production also disables the interactive API documentation.

Public queries are limited to five retrieved chunks and 1,000 non-whitespace characters. Outbound LLM requests have a timeout and bounded retries. The container runs as a non-root user and accepts Cloud Run's `$PORT`. See the [Cloud Run security profile](deploy/cloud-run-security.md) for service limits, IAM, Secret Manager, and release checks.

The API writes structured logs to stdout. A failed query records whether retrieval or generation failed, and every response includes an `x-request-id` header for finding its logs. Cloud Run alerts must be configured during deployment; see the operations section of the [Cloud Run profile](deploy/cloud-run-security.md#finding-failures).

The public corpus is packaged with the API image. Each Cloud Run instance rebuilds its ephemeral Chroma index during startup.
The image includes only the three policy files named in the [Dockerfile](Dockerfile); evaluation material and other local notes are excluded from the Docker build context. Git tracking is separate from image contents.

## RAG evaluation

The [labeled set](data/evaluation/cases.json) has 24 questions: 8 for development and 16 for a held-out test run. It covers direct facts, paraphrases, conditions, a two-document question, and questions the corpus cannot answer. Every answerable case has a reference answer and one or more exact evidence spans linked to source files. Gold spans are checked against the corpus before a run.

```bash
python scripts/ingest.py --path data/documents --reset
python scripts/evaluate.py --split dev --output storage/evaluation/dev.json
# Tune retrieval only on dev, then run test once with the selected settings:
python scripts/evaluate.py --split test --output storage/evaluation/test-retrieval.json
python scripts/evaluate.py --split test --generate --output storage/evaluation/test-answers.json
```

Retrieval scoring is deterministic: **evidence recall@k** is the fraction of required gold spans found in the top k passages, **hit@k** indicates whether any required span was found, and **MRR** rewards finding the first relevant passage early. The report keeps every passage, score, evidence rank, configuration, dataset and corpus hashes, and warm retrieval latency. It gives 95% bootstrap intervals across questions. These intervals describe this small question set; correlated questions and a tiny corpus limit wider generalization. Unanswerable questions have no gold retrieval passage, so they are excluded from retrieval success metrics.

Generated answers require a separate [human review protocol](data/evaluation/REVIEW.md). The report records the exact answer and context. Reviewers score correctness against the reference, grounding in the supplied context, completeness, citation support, and abstention on unanswerable questions. Keyword overlap is deliberately absent from the quality score. Create and score reviews with:

```bash
python scripts/review_answers.py init storage/evaluation/test-answers.json storage/evaluation/reviewer-a.json --reviewer reviewer-a
# Fill all applicable 0/1 judgments in reviewer-a.json, then:
python scripts/review_answers.py score storage/evaluation/test-answers.json storage/evaluation/reviewer-a.json
```

For diagnosis, generate answers with gold source documents as context, then review those answers using the same rubric. If oracle-context answers succeed while normal RAG answers fail, retrieval is a likely bottleneck. If both fail, inspect the prompt or generator. Oracle runs on unanswerable cases are trivial because they receive no context, so compare answerable cases separately.

```bash
python scripts/evaluate.py --split test --generate --context oracle --output storage/evaluation/test-oracle.json
python scripts/compare_runs.py storage/evaluation/test-retrieval.json storage/evaluation/candidate.json
```

The comparison script uses matched questions and paired bootstrap differences and lists gains and regressions. Keep the test set frozen while tuning; add a new held-out set if you revise questions or gold labels. This starter set is useful for regression checks, but a convincing public claim needs more independently authored questions, more documents, and ideally two reviewers with disagreements adjudicated.

This is a single-node technical demonstration, not a multi-tenant service. Scanned PDFs need OCR, and document index updates are not transactional with file replacement. These are the main remaining ingestion limits. Hybrid retrieval and reranking should follow measured retrieval failures rather than being added by default.

## Project structure

```text
src/
  api/           # FastAPI routes and schemas
  ingestion/     # loaders and recursive chunking
  embeddings/    # local and OpenAI embedding providers
  vectorstore/   # embedded or HTTP ChromaDB wrapper
  retrieval/     # similarity retrieval
  generation/    # Anthropic, OpenAI, and llama.cpp LLM providers
scripts/         # command-line ingestion
data/documents/  # reproducible demo corpus
tests/           # unit and API contract tests
deploy/          # Cloud Run security and operations guide
```
