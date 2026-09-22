# Cloud Run security profile

The recruiter demo is a public read-only application. Its versioned corpus is
public, and changing that corpus requires a new image. Runtime ingestion,
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

Example deployment settings (replace placeholders and deploy only after the
fixed Chroma index has been packaged into the image):

```bash
gcloud run deploy rag-demo \
  --image europe-west1-docker.pkg.dev/PROJECT/rag/rag-demo:TAG \
  --region europe-west1 \
  --service-account rag-demo-runtime@PROJECT.iam.gserviceaccount.com \
  --allow-unauthenticated \
  --port 8080 \
  --cpu 2 \
  --memory 2Gi \
  --concurrency 2 \
  --max-instances 1 \
  --min-instances 0 \
  --timeout 60 \
  --set-env-vars ENVIRONMENT=prod,PUBLIC_DEMO_QUERY=true,ADMIN_API_ENABLED=false,MAX_TOKENS=512,LLM_TIMEOUT_SECONDS=45,LLM_MAX_RETRIES=1 \
  --set-secrets ANTHROPIC_API_KEY=rag-anthropic-api-key:latest
```

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

Cloud Run's writable filesystem is ephemeral and consumes instance memory.
Do not rely on runtime uploads or a runtime-built Chroma index for persistence.
The fixed index packaging step is a deployment task and is intentionally kept
separate from this security change.
