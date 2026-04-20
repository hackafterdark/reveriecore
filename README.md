# 🌌 Reverie Core

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/hackafterdark/reveriecore)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Hermes-purple.svg)](https://github.com/hackafterdark/reveriecore)

**Reverie Core** is an high-performance, intelligence-driven Retrieval-Augmented Generation (RAG) memory engine for the Hermes agent ecosystem. It transforms flat text memories into a structured, semantic knowledge graph.

---

## ✨ Key Features

- **🧠 Deep Semantic Enrichment**: Uses `BART-Large-MNLI` for zero-shot classification and importance scoring.
- **⚡ Hybrid Graph-RAG Retrieval**: Combines `sqlite-vec` similarity search with **Bidirectional Graph Traversal** to find non-obvious context.
- **🛡️ Hub Protection**: Intelligent per-node limits (10) to prevent popular entities from creating "noise" in your context window.
- **🌐 Augmented Knowledge**: Automatically extracts canonical entities (Files, Tools, API nodes) and maps their relationships.
- **🛡️ Namespace Isolation**: Robust identity model ensuring profile-based sandboxing and privacy.
- **📦 Zero-Config Connectivity**: Dynamically discovers your LLM provider (Llama Swap, Ollama, OpenAI) from the Hermes `config.yaml`.

---

## 🛠️ Technology Stack

- **Core Intelligence**: `transformers` (BART), `sentence-transformers` (all-MiniLM-L6-v2)
- **Database Engine**: `SQLite 3` with `sqlite-vec` extension
- **Integration**: Python 3.x, Hermes Plugin Bridge

---

## 📂 Directory Structure

```bash
reveriecore/
├── AGENT_DOCS/          # Detailed design and research documents
├── PRD/                 # Product Requirements Document
├── tests/               # Unit and integration tests
├── database.py          # SQLite & Vector initialization and management
├── enrichment.py        # LLM-based extraction, scoring, and profiling
├── graph_query.py       # Bidirectional traversal & Hub Protection logic
├── provider.py          # Hermes Memory Provider implementation
├── retrieval.py         # Hybrid search and re-ranking logic
├── schemas.py           # Data models, Enums, and type definitions
├── requirements.txt     # Python dependencies list
```

---

## 🚀 Getting Started

### 1. Requirements
- Python 3.8+
- SQLite 3 (with loadable extension support)

### 2. Installation
Clone this repository into your Hermes plugins directory:

```bash
git clone https://github.com/hackafterdark/reveriecore.git
cd reveriecore
pip install -r requirements.txt
```

### 3. Verification
You can monitor your Knowledge Graph health using the built-in status tool (no Hermes environment required):
```bash
python3 tests/graph_status.py
```

### 4. Configuration
Reverie Core automatically reads your active LLM provider (base_url, model) from `~/.hermes/config.yaml`. No manual configuration is required unless you want to override the `importance_score` thresholds in `enrichment.py`.

---

## 📖 Documentation

For deep dives into the architecture and theory behind Reverie Core, see the [AGENT_DOCS](AGENT_DOCS) directory:

- [Database Schema](AGENT_DOCS/db_schema.md)
- [Retrieval Workflow](AGENT_DOCS/how_memory_retrieval_works.md)
- [Knowledge Graph Mechanics](AGENT_DOCS/knowledge_graph_mechanics.md)
- [Storage & Enrichment](AGENT_DOCS/how_memory_storage_works.md)
- [Importance Score Mechanics](AGENT_DOCS/memory_importance_score.md)
- [Product Requirements (PRD)](PRD/PRD.md)

---

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.
