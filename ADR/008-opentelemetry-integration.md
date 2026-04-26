# ADR 008: Framework Observability via OpenTelemetry

## Status
Accepted (2026-04-25)

## Context
As ReverieCore evolved from a simple plugin into a modular framework with multi-stage enrichment and retrieval pipelines (see [ADR 006](006-reverie-framework-pipeline-architecture.md)), it became increasingly difficult to diagnose performance bottlenecks and non-deterministic behavior.

Key challenges included:
1.  **Pipeline Visibility**: It was unclear which handler in a pipeline (e.g., `SoulImportance` vs. `HeuristicImportance`) was consuming the most time or causing a failure.
2.  **LLM Cost & Performance**: There was no standardized way to track token usage and latency across different LLM backends.
3.  **Context Precision Debugging**: Visualizing how many candidates were being filtered by the "Precision Gate" or expanded by graph traversal was manual and error-prone.

## Decision
We will implement **OpenTelemetry (OTel)** as the standard observability layer for ReverieCore. This provides a vendor-neutral, industry-standard way to collect and export traces and metrics.

### 1. Manual Pipeline Instrumentation
Every handler in the Enrichment and Retrieval pipelines is wrapped in a "Span."
- **Parent Spans**: `reverie.enrichment` and `reverie.retrieval` provide a high-level view of a turn's performance.
- **Child Spans**: Each `Handler.process` call is instrumented as a child span (e.g., `reverie.enrichment.handler.SoulImportance`). This allows for a "waterfall" visualization of the entire pipeline.

### 2. GenAI Semantic Conventions
To ensure compatibility with modern dashboarding tools, we implement the **OTel GenAI Semantic Conventions** (`gen_ai.*` attributes).
- LLM calls in `InternalLLMClient` capture model versions, system identifiers, and token counts.
- This ensures that standard OTel dashboards (Grafana, Jaeger, Honeycomb) can automatically visualize ReverieCore data without custom configuration.

### 3. Trace-to-Log Correlation
We integrated `opentelemetry-instrumentation-logging` to inject `trace_id` and `span_id` into the standard `reveriecore.log`. This allows developers to jump from a visual trace in Jaeger directly to the raw logs for that specific interaction.

### 4. Robust OTLP Exporting
The telemetry layer is designed to be "fail-safe":
- **Exporter Fallback**: The system first attempts to use the high-performance OTLP/gRPC exporter, but gracefully falls back to OTLP/HTTP if dependencies are missing or connectivity fails.
- **No-Op Mode**: If no OTel dependencies are present, the system defaults to a No-Op mode, ensuring that observability code never crashes the core RAG engine.

## Consequences

### Positive
- **Deep Observability**: Developers can now see exactly where latency is occurring (e.g., a slow embedding model or a high-latency LLM call).
- **Standardization**: By using OTel, ReverieCore can be plugged into any enterprise observability stack.
- **Visual Debugging**: Failed pipeline steps are marked red in traces, making error diagnosis immediate and visual.

### Negative
- **Dependency Overhead**: Adds several OTel-related packages to the project requirements.
- **Instrumentation Surface**: Every new handler must be instrumented to maintain visibility, though the orchestrator handles much of this automatically.
- **Performance Impact**: Minimal, but batching spans and network I/O for exporting does consume some resources (mitigated by asynchronous exporting).
