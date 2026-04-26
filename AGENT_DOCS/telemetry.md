# OpenTelemetry & Observability in ReverieCore

ReverieCore uses **OpenTelemetry (OTel)** to provide deep visibility into the performance and logic of the memory pipelines. This allows you to identify latency bottlenecks, monitor LLM token usage, and debug complex graph traversals.

## Architecture

The system is instrumented with manual spans that track:
1.  **Enrichment Pipeline**: Ingestion, importance scoring, and semantic profiling.
2.  **Retrieval Pipeline**: Discovery (Vector/Graph), Ranking, and Budgeting.
3.  **LLM Calls**: Token usage and model latency for external LLM providers.
4.  **Graph Operations**: Performance of knowledge graph traversals.

## GenAI Semantic Conventions

We follow the [OpenTelemetry GenAI Semantic Conventions](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) to ensure compatibility with standard observability tools like Jaeger, Grafana, and Honeycomb.

| Attribute | Description |
| :--- | :--- |
| `gen_ai.system` | The LLM provider/backend (e.g., `http://localhost:11434/v1`). |
| `gen_ai.request.model` | The specific model being called (e.g., `gemma2:2b`). |
| `gen_ai.operation.name` | The type of LLM operation (usually `chat`). |
| `gen_ai.usage.input_tokens` | Tokens sent in the prompt. |
| `gen_ai.usage.output_tokens` | Tokens generated in the response. |

## Viewing Traces Locally

The system is configured to export traces via OTLP (gRPC) to `localhost:4317`.

### 1. Run Jaeger
You can run Jaeger using Docker or the standalone **all-in-one binary** (highly recommended for a zero-dependency setup).

#### Option A: Standalone Binary (Recommended)
1.  Download the latest [Jaeger release](https://www.jaegertracing.io/download/).
2.  Extract the archive and locate the `jaeger-all-in-one` (or `jaeger-all-in-one.exe`) binary.
3.  Run the binary from your terminal. No configuration is needed for default gRPC support.
4.  Open `http://localhost:16686`.

#### Option B: Docker
```bash
docker run --rm -d --name jaeger \
  -e COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
  -p 16686:16686 \
  -p 4317:4317 \
  -p 4318:4318 \
  jaegertracing/all-in-one:latest
```

### 2. Trace-to-Log Correlation
ReverieCore automatically injects `trace_id` and `span_id` into `reveriecore.log`. 
Example log line:
`2026-04-25 16:11:23 - reveriecore.retrieval - INFO - [trace_id=583e7... span_id=a1b2c...] Retrieved 3 memories...`

This allows you to find the exact log lines associated with a specific request in your dashboard.

## Instrumentation Details

### Pipeline Spans
- `reverie.enrichment`: Parent span for the ingestion process.
- `reverie.retrieval`: Parent span for semantic search.
- `reverie.enrichment.handler.<Name>`: Individual stages of enrichment.
- `reverie.retrieval.handler.<Name>`: Individual stages of retrieval.

### Error Handling
If a pipeline step fails, the corresponding span will be marked as **ERROR** and include the exception details and stack trace. This makes it easy to visually identify failed operations in Jaeger.
