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
only the three policy files named in `Dockerfile` and rebuilds the ephemeral
index at startup. `.dockerignore` excludes evaluation files and other notes
from the Docker build context. `.gcloudignore` applies the same file list when
submitting source with `gcloud`. Before a source build, run
`gcloud meta list-files-for-upload` and confirm it lists only the Dockerfile,
requirements, application source, and three policy files.

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

## Finding failures

The application writes no log file under `storage/`. Locally, use
`docker compose logs rag-api` or `docker compose logs chroma`. Cloud Run
collects each container's stdout and HTTP request logs in Cloud Logging; open
the service's **Logs** tab or Logs Explorer. The default Cloud Logging bucket
[retains logs for 30 days](https://docs.cloud.google.com/logging/docs/store-log-entries)
unless its retention is changed.

Application logs are JSON records; failure records have an `event` and
`severity`, and HTTP failure records carry a request ID. The same ID is returned in
the `x-request-id` response header. Search for that ID in Logs Explorer to
connect an error response with the application record. Query
failures include `stage` (`retrieval` or `generation`), exception class, the
failing code location, and the upstream HTTP status when available. Successful
queries report retrieval and generation time and the number of retrieved
chunks. These application records omit
questions, passages, answers, and exception messages.

In Logs Explorer, filter on `resource.type="cloud_run_revision"` and the
service name. Search `jsonPayload.event="query_failed"` for RAG errors or
`jsonPayload.event="startup_failed"` when a revision does not become ready.
The Chroma sidecar's logs are available under the same service and revision.
If a request has no application failure record, inspect Cloud Run system logs
for container startup, memory, or shutdown errors.
Set a Cloud Monitoring alert for HTTP 5xx responses and startup failures with
an email notification channel. Cloud Run does not create these alerts by
itself. [Cloud Run logging](https://docs.cloud.google.com/run/docs/logging)
and [log-based alerts](https://docs.cloud.google.com/logging/docs/alerting/log-based-alerts)
describe the console setup.
