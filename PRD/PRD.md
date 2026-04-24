# Product Requirements Document (PRD): Reverie Core Memory Plugin

## 1. Project Overview
**Reverie Core** is a high-performance, embedded Retrieval-Augmented Generation (RAG) memory system designed for the Hermes agent ecosystem. It replaces simple text-based memory with a robust, searchable knowledge base using **SQLite** as the storage engine and **`sqlite-vec`** for vector similarity search.

The project aims to provide agents with a "long-term memory" that isn't just a list of strings, but a structured, categorized, and scored knowledge graph that can be queried with semantic precision.

## 2. Target Audience
- **Hermes Agents**: To provide persistent, high-context retrieval.
- **Developers**: Building advanced agentic workflows that require reliable and isolated memory storage.

## 3. Core Objectives
1.  **Semantic Retrieval**: Enable agents to find memories based on meaning rather than just keyword matches.
2.  **Intelligence-First Storage**: Automatically classify, score, and summarize memories upon ingestion.
3.  **Namespace Isolation**: Ensure strict sandboxing of memories between different agent profiles.
4.  **Desktop Friendly**: Zero-config, single-file persistence using SQLite.
5.  **Multi-Model Analysis**: Utilize local LLM-based models (mDeBERTa/BART) for deep semantic understanding.
6.  **Data Sovereignty**: Provide a human-readable, portable mirror of agent cognitive state.

## 4. Key Features

### 4.1. Hybrid Graph-RAG Search
Reverie Core combines vector similarity with deterministic graph traversal and hierarchical discovery.
- **Base Search**: KNN search on 384-dimension embeddings (vector similarity).
- **Graph Augmentation**: Bidirectional traversal of the association graph to bridge memories via shared entities.
- **Hierarchical Discovery (Tree of Nuance)**: Automatically detects **Observation Anchors** and provides signals for **Agentic Drill-Down** using the `recall_reverie` tool.
- **Re-ranking**: Final result order determined by vector similarity, mDeBERTa importance, temporal decay, and graph proximity.

### 4.2. The Intelligence Layer (Enrichment)
Automated processing of every memory saved:
- **Zero-Shot Classification**: Categorizes memories into types (e.g., `OBSERVATION`, `TASK`, `USER_PREFERENCE`) using mDeBERTa-v3.
- **Semantic Importance Scoring**: Assigns a weight (1.0 - 5.0) to memories based on their criticality.
- **Summarization**: Generates a 1-2 sentence "semantic profile" (gist) for long memories.

### 4.3. Active Cognitive Maintenance (MesaService)
Ensures the "Active" database remains high-signal over time.
- **Tier 1 (Archive)**: Identifies stale, low-importance fragmented memories and moves them to `ARCHIVED` status.
- **Tier 1.5 (Consolidation)**: Crystallizes clusters of related stale memories into high-level **Observation Anchors**, preserving granularity via `CHILD_OF` links.
- **Tier 2 (Purge)**: Physical cleanup of old archives and database `VACUUM`.
- **Recency Protection**: Retrieval hits act as a "Stay Alive" signal, shielding active context from pruning.

### 4.4. Memory-as-Code (Markdown Mirror) [NEW]
A bi-directional synchronization system between SQLite and a filesystem-based Markdown archive.
- **Markdown Export**: Serializes memory nodes into `.md` files with YAML frontmatter containing IDs, importance, and hierarchical associations.
- **Bi-directional Sync**: Ability to "re-birth" a memory database from a folder of Markdown files.
- **Human Readability**: Allows users to interact with agent memory using standard tools like Obsidian, VS Code, or `grep`.

### 4.5. Identity, Provenance & Security
A robust model to track memory origin and ownership:
- **Multi-Tenant Isolation**: Strict `owner_id` (Namespace) enforcement.
- **Provenance Validation**: Security checks ensuring agents can only recall fragments if they are descendants of an authorized Observation Anchor.
- **Audit Trails**: Full tracking of `author_id`, `actor_id`, `session_id`, and `workspace`.

## 5. Technical Stack
- **Languages**: Python 3.x (Plugin logic), SQL (Persistence), Go (Hermes Bridge).
- **Database**: SQLite 3 with `sqlite-vec` extension for local vector storage.
- **Intelligence Layer**: Unified **`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`** for cross-lingual classification and importance scoring.
- **Embeddings**: `all-MiniLM-L6-v2` (384 dimensions).

## 6. Data Schema
- **`memories`**: Content, metadata, importance scores, and hierarchical signaling.
- **`entities`**: Canonical identifiers for conceptually anchored world knowledge.
- **`memory_relations`**: Polymorphic table mapping relationships (`CHILD_OF`, `SUPERSEDES`, `MENTIONS`).
- **`memories_vec`**: Optimized virtual table for KNN vector search.

## 7. Success Metrics
- **Context Density**: Up to 4x higher information density via Hierarchical Observation Anchors.
- **Reasoning Accuracy**: Significant reduction in "Brain Rot" hallucinations via active Mesa pruning.
- **Recovery**: 100% restoration of Memory State from Markdown Mirror in case of DB corruption.
