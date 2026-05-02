# 🌌 Reverie Core

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Hermes-purple.svg)](https://github.com/hackafterdark/reveriecore)

**Reverie Core** is an agentic cognition layer designed for the **Hermes agent ecosystem**. Unlike static RAG frameworks that treat memory as a passive document store, Reverie Core manages memory as a **dynamic, self-pruning knowledge graph.** It bridges the gap between raw data and long-term agent intelligence by automating the **Enrichment** (understanding), **Retention** (budgeting), and **Retrieval** (graph-traversal) of your agent's experience.

Designed from the ground up for local-first performance and full observability, it ensures your agent doesn't just "retrieve" data—it "remembers" with purpose.

---

## 🔗 Platform Strategy: Hermes Memory Provider

**Reverie Core** functions as a high-performance **memory plugin** for the [Hermes](https://hermes-agent.nousresearch.com/) ecosystem. Unlike standard utility plugins, it plugs directly into the agent’s core cognitive loop, replacing basic memory storage with a stateful, graph-based knowledge engine.

* **Current Status**: Built specifically for the Hermes `memory_provider` interface, leveraging native hooks for pre-fetch context injection and background write-back synchronization.
* **The Roadmap**: While currently optimized for the Hermes runtime, the core logic is abstracted via `provider.py`. We are actively refining this **Provider Interface** to ensure the cognition engine remains platform-agnostic, with a roadmap to support future integration as an MCP (Model Context Protocol) server for IDE-based agents (like Cursor) and standalone agent runtimes.

If you are a developer looking to bridge Reverie Core into another agent runtime, `provider.py` serves as the primary abstraction layer, mapping internal graph and memory logic to the host agent's lifecycle events.

---

## ✨ Why Reverie Core?

* **From "RAG" to "Memory"**: Move beyond simple text retrieval. Our `MesaService` actively maintains your knowledge graph, archiving transient noise and elevating critical insights so your context window stays performant.
* **Decoupled Intelligence**: A plug-and-play architecture where enrichment and retrieval pipelines are fully composable. See our [**Pipeline Architecture Diagram**](DIAGRAM.md) for a visual breakdown. Swap handlers for classification, profiling, or ranking as your agent's needs evolve.
* **Local-First, Graph-Powered**: Uses `sqlite-vec` for high-speed similarity search combined with **Bidirectional Graph Traversal** to bridge non-obvious relationships in your data.
* **Production-Ready Observability**: Built-in **OpenTelemetry** instrumentation (OTLP) provides granular, trace-based insight into how your agent "thinks" and where its retrieval precision bottlenecks lie.
* **Config-Driven Engineering**: Move away from hardcoded magic numbers. Every threshold, weight, and pipeline stage is managed via a validated `reveriecore.yaml`, allowing for precise, reproducible benchmark tuning.

### 🧠 Data Portability & Resilience
Reverie Core treats your agent's memory as a first-class citizen. Its **Sync Engine** provides:
- **Bi-Directional Portability**: Export your entire semantic knowledge graph to Hive-partitioned Markdown files (e.g., `year=2026/month=04/day=27/`) with relationship-mapped frontmatter, and import them back into any `ReverieCore` instance with full structural integrity.
- **Version-Controlled Cognition**: Exported memories are human-readable and Git-friendly. Commit your agent's "brain" to your repository to track how its knowledge evolves over time.
- **Data Lake Ready**: Hive-style directory partitioning allows you to hook your agent’s long-term memory into standard data analytics tools (like DuckDB, Trino, or Apache Spark) without additional ETL.

---

### Key Differentiators

| Feature | Reverie Core | Standard RAG Frameworks |
| :--- | :--- | :--- |
| **Primary Goal** | Agent State & Cognition | Document Retrieval |
| **Maintenance** | Active (`MesaService`) | Passive (Static Index) |
| **Graph Logic** | Bi-directional Traversal | Vector-only or Fixed-depth |
| **Configurability** | Pydantic-Validated YAML | Hardcoded or Code-heavy |
| **Tracing** | Native OpenTelemetry | Print statements or external wrappers |
| **Portability** | Hive-Partitioned Markdown | Opaque Binary Blobs |
| **Platform** | Native Hermes (Extensible) | Tied to specific RAG libraries |

---

### 📊 Benchmark Results & Interpretation

Reverie Core is benchmarked against a grounded question-answering [dataset](https://huggingface.co/datasets/vibrantlabsai/amnesty_qa) using [RAGAS](https://docs.ragas.io/en/stable/) to measure system reliability. We prioritize **Faithfulness** (the agent's ability to ground answers in context) and **Context Precision** (the relevance of retrieved information).

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **Faithfulness** | **0.925** | **Reliably Grounded.** High factual alignment between retrieval context and agent response, effectively eliminating hallucination. |
| **Context Precision** | **0.70** | **High-Signal Retrieval.** The pipeline consistently surfaces relevant nodes at the top of the context window for synthesis. |

> **Note on Performance:** These benchmarks represent a "Real-World Baseline" achieved with the configuration settings found in `reveriecore.yaml.example`. We intentionally avoid "benchmark hacking" with synthetic datasets, preferring to tune for grounding and low-latency performance that holds up in daily usage.

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
Reverie Core uses a structured, validated configuration system. The engine discovers configuration in this order:
1.  **Hermes Pointer**: `memory.reveriecore_cfg` in `config.yaml`.
2.  **Env Var**: `REVERIECORE_CONFIG` path override.
3.  **Local Default**: `~/.reveriecore.yaml`.

For a full list of settings, see [**CONFIGURATION.md**](CONFIGURATION.md).

---

## 📖 Documentation & Architecture

For deep dives into the mechanics, see the [AGENT_DOCS](AGENT_DOCS) and [ADR](ADR) directories:

- [**Pipeline Architecture Diagram**](DIAGRAM.md)
- [**Full Configuration Guide**](CONFIGURATION.md)
- [ADR 006: Pipeline Architecture](ADR/006-reverie-framework-pipeline-architecture.md)
- [ADR 008: OpenTelemetry Integration](ADR/008-opentelemetry-integration.md)
- [Knowledge Graph Mechanics](AGENT_DOCS/knowledge_graph_mechanics.md)
- [Active Maintenance (Mesa)](AGENT_DOCS/how_mesa_works.md)

---

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.