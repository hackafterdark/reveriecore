# 🌌 ReverieCore

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/hackafterdark/reveriecore)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Hermes-purple.svg)](https://github.com/hackafterdark/reveriecore)

**ReverieCore** is an high-performance, intelligence-driven Retrieval-Augmented Generation (RAG) memory engine for the Hermes agent ecosystem. It transforms flat text memories into a structured, semantic knowledge graph.

---

## ✨ Key Features

- **🧠 Deep Semantic Enrichment**: Uses `BART-Large-MNLI` for zero-shot classification and semantic importance scoring.
- **⚡ Vector-First Retrieval**: High-speed similarity search powered by `sqlite-vec`.
- **📊 Intelligence Re-ranking**: Hybrid ranking logic that combines semantic similarity with calculated importance weights.
- **🌐 Knowledge Graph (Associations)**: Logical linking of memories (e.g., `CAUSES`, `DEPENDS_ON`) to build cross-contextual understanding.
- **🛡️ Namespace Isolation**: Robust identity model ensuring data privacy and profile-based sandboxing.
- **📦 Embedded & Lightweight**: Single-file SQLite persistence with local transformer inference—zero external database dependencies.

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
├── enrichment.py        # LLM-based classification, scoring, and profiling
├── provider.py          # Hermes Memory Provider interface implementation
├── retrieval.py         # Hybrid search and re-ranking logic
├── schemas.py           # Data models, Enums, and type definitions
├── plugin.yaml          # Plugin metadata and configuration
├── pyproject.toml       # Build and dependency configuration
└── requirements.txt     # Python dependencies list
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

### 3. Usage
The plugin is automatically discovered by Hermes. Ensure your `plugin.yaml` points to the correct `provider.py` entry point.

---

## 📖 Documentation

For deep dives into the architecture and theory behind ReverieCore, see the [AGENT_DOCS](file:///home/tom/.hermes/plugins/reveriecore/AGENT_DOCS) directory:

- [Database Schema](file:///home/tom/.hermes/plugins/reveriecore/AGENT_DOCS/db_schema.md)
- [Retrieval Workflow](file:///home/tom/.hermes/plugins/reveriecore/AGENT_DOCS/how_memory_retrieval_works.md)
- [Storage & Enrichment](file:///home/tom/.hermes/plugins/reveriecore/AGENT_DOCS/how_memory_storage_works.md)
- [Importance Score Mechanics](file:///home/tom/.hermes/plugins/reveriecore/AGENT_DOCS/memory_importance_score.md)
- [Product Requirements (PRD)](file:///home/tom/.hermes/plugins/reveriecore/PRD/PRD.md)

---

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.
