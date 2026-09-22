# Cloud Run security profile

The recruiter demo is a public read-only application. Its versioned corpus is
public, and changing that corpus requires a new image. Runtime API ingestion,
reindexing, deletion, document listing, and statistics are disabled.

## Required service settings

- Allow unauthenticated invocation only because the query page is public.
- Set `ENVIRONMENT=prod`, `PUBLIC_DEMO_QUERY=true`, and
  `ADMIN_API_ENABLED=false`.
- Store the selected LLM API key in Secret Manager and expose it only to the
  dedicated Cloud Run runtime service account.
- Start with 2 vCPU, 2 GiB memory, concurrency 2, maximum instances 1, minimum
  instances 0, and a 60 second request timeout. Revisit these values after load
  testing. Maximum instances is a cost guardrail, not an exact rate limiter.
- Set a project billing budget and provider-side API spending alert.
- Grant the runtime service account no project roles except Secret Manager
  Secret Accessor on the single LLM secret.

The container intentionally listens on `0.0.0.0:$PORT`; Cloud Run requires the
ingress container to listen on all interfaces. External exposure is controlled
by Cloud Run IAM and ingress, rather than by a host port mapping.

Deploy the API as the ingress container and `chromadb/chroma:0.5.5` as a
sidecar. Configure the API to depend on Chroma, give Chroma a TCP startup probe
on port 8000, and set these API environment variables:

```dotenv
CHROMA_HOST=localhost
CHROMA_PORT=8000
DEMO_CORPUS_PATH=/app/data/documents
```

Containers in one Cloud Run instance share a network, so Chroma remains
unexposed while the API reaches it through `localhost`. The API image contains
the public corpus and rebuilds the ephemeral index at startup.

In the Cloud Run console, configure the API container with the Artifact
Registry image, port 8080, the environment above, and the LLM secret. Add the
Chroma image as the second container, then apply the service limits from the
previous section. Cloud Run requires an explicit startup probe when container
startup ordering is used.

## Release checks

After deployment, verify that:

1. `/`, `/static/*`, `/api/v1/health`, and `POST /api/v1/query` work over HTTPS.
2. `/docs` and `/openapi.json` return 404 in production.
3. `/api/v1/ingest`, `/api/v1/documents`, and `/api/v1/stats` return 404,
   including when an `x-api-key` header is supplied.
4. Logs contain no LLM key, document contents, retrieved passages, or complete
   user questions.
5. The Cloud Run revision uses the dedicated service account, concurrency 2,
   maximum instances 1, and the configured timeout.

Cloud Run's writable filesystem is ephemeral and consumes instance memory. The
demo intentionally rebuilds its index after an instance replacement.
