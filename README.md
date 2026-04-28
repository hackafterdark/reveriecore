# 🌌 Reverie Core

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Hermes-purple.svg)](https://github.com/hackafterdark/reveriecore)

**Reverie Core** is a modular, high-performance RAG framework designed for the Hermes agent ecosystem. It transforms flat text memories into a structured, semantic knowledge graph using a decoupled pipeline architecture and industry-standard observability.

---

## ✨ Key Features

- **🧠 Pipeline Orchestration**: Decoupled **Enrichment** and **Retrieval** pipelines that allow for composable handlers (Classification, Profiling, Ranking).
- **⚡ Hybrid Graph-RAG**: Combines `sqlite-vec` similarity search with **Bidirectional Graph Traversal** to find non-obvious context.
- **🛡️ Hub Protection**: Intelligent per-node limits to prevent popular entities from creating "noise" in your context window.
- **🌐 Augmented Knowledge**: Automatically extracts canonical entities (Files, Tools, API nodes) and maps their relationships.
- **📊 Standardized Observability**: Full **OpenTelemetry** integration with GenAI semantic conventions for tracing and debugging.
- **📦 Validated Configuration**: Type-safe, Pydantic-backed configuration via `reveriecore.yaml` with automatic mathematical validation (weights, thresholds).
- **🧹 Active Maintenance**: Background `MesaService` automatically archives noise and sanitizes your context window.

---

## 🛠️ Technology Stack

- **Intelligence**: `transformers` (BART), `sentence-transformers` (all-MiniLM-L6-v2), `flashrank` (MiniLM reranker)
- **Storage**: `SQLite 3` with `sqlite-vec` extension
- **Observability**: `OpenTelemetry` (OTLP/gRPC)
- **Integration**: Python 3.11+, Hermes Plugin Bridge

---

## 🚀 Getting Started

### 1. Installation
Clone this repository into your Hermes plugins directory and install dependencies into the agent's virtual environment:

```bash
git clone https://github.com/hackafterdark/reveriecore.git
cd reveriecore
VIRTUAL_ENV=~/.hermes/hermes-agent/venv uv pip install -e .
./run_tests.sh  # Verifies environment and initializes local models
```

### 1.5. Download Query Rewriter Model (Optional)
If you plan to use the **Query Rewriter** (highly recommended for complex queries), you must download the GGUF model manually. From the `reveriecore` root (assuming you have the Hugging Face CLI tool installed):

```bash
hf download microsoft/Phi-3-mini-4k-instruct-gguf Phi-3-mini-4k-instruct-q4.gguf --local-dir models
```

### 2. Configuration (`reveriecore.yaml`)
Reverie Core now uses a structured, validated configuration system. The engine discovers configuration in this order:
1.  **Hermes Pointer**: `memory.reveriecore_cfg` in `config.yaml`.
2.  **Env Var**: `REVERIECORE_CONFIG` path override.
3.  **Local Default**: `~/.reveriecore.yaml`.

For a full list of available settings and their descriptions, see [**CONFIGURATION.md**](CONFIGURATION.md).

#### Example `reveriecore.yaml`
```yaml
retrieval:
  discovery:
    vector:
      precision_gate: 0.45
  rewriter:
    enabled: true
    model_path: "models/Phi-3-mini-4k-instruct-q4.gguf"
  pipeline:
    discovery: ["anchoring", "vector"]
    ranking: ["intent", "scoring", "rerank"]

enrichment:
  models:
    embedding: "all-MiniLM-L6-v2"
  pipeline:
    active_stages: ["heuristics", "classifier", "model_importance"]
```

---

## 🔬 Observability (OpenTelemetry)

Reverie Core is fully instrumented with **OpenTelemetry**. Every stage of the memory pipeline generates spans, allowing you to visualize exactly where latency or context-precision bottlenecks occur.

### Viewing Traces
The engine exports traces via OTLP to `localhost:4317`. To view them locally, you can use either Docker or the Jaeger **all-in-one binary** (highly recommended for simplicity):

#### Option A: All-in-One Binary (Recommended)
1. Download the [Jaeger binary](https://www.jaegertracing.io/download/) for your platform.
2. Run the `jaeger-all-in-one` executable.
3. Visit `http://localhost:16686`.

#### Option B: Docker
```bash
docker run --rm -d --name jaeger -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one:latest
```

---

## 📖 Documentation

For deep dives into the architecture, see the [AGENT_DOCS](AGENT_DOCS) and [ADR](ADR) directories:

- [**Full Configuration Guide**](CONFIGURATION.md)
- [**Telemetry & Tracing**](AGENT_DOCS/telemetry.md)
- [ADR 006: Pipeline Architecture](ADR/006-reverie-framework-pipeline-architecture.md)
- [ADR 008: OpenTelemetry Integration](ADR/008-opentelemetry-integration.md)
- [Database Schema](AGENT_DOCS/db_schema.md)
- [Knowledge Graph Mechanics](AGENT_DOCS/knowledge_graph_mechanics.md)
- [Storage & Enrichment](AGENT_DOCS/how_memory_storage_works.md)
- [Active Maintenance (Mesa)](AGENT_DOCS/how_mesa_works.md)
- [**Cross-Encoder Reranking**](AGENT_DOCS/reranker.md)

---

## 📂 Directory Structure

```bash
reveriecore/
├── AGENT_DOCS/          # Detailed design and research documents
├── ADR/                 # Architectural Decision Records
├── tests/               # Unit and integration tests
├── config.py            # Priority-based configuration discovery
├── database.py          # SQLite & Vector initialization
├── enrichment.py        # Enrichment Pipeline & LLM Handlers
├── graph_query.py       # Graph traversal & Hub Protection
├── mirror.py            # Lazy re-vectorization & persistent workers
├── provider.py          # Hermes Plugin Entrypoint
├── pruning.py           # Token budgeting & selection strategies
├── reranking.py         # Cross-Encoder Reranking (Stage D)
├── retrieval.py         # Hybrid-RAG Pipeline Orchestrator
├── retrieval_base.py    # Shared types & circular import prevention
├── schemas.py           # Standardized data models
├── telemetry.py         # OpenTelemetry infrastructure
├── CONFIGURATION.md     # Detailed configuration reference
└── reveriecore.yaml     # Local framework configuration
```

---

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.
