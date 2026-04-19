# Product Requirements Document (PRD): ReverieCore Memory Plugin

## 1. Project Overview
**ReverieCore** is a high-performance, embedded Retrieval-Augmented Generation (RAG) memory system designed for the Hermes agent ecosystem. It replaces simple text-based memory with a robust, searchable knowledge base using **SQLite** as the storage engine and **`sqlite-vec`** for vector similarity search.

The project aims to provide agents with a "long-term memory" that isn't just a list of strings, but a structured, categorized, and scored knowledge graph that can be queried with semantic precision.

## 2. Target Audience
- **Hermes Agents**: To provide persistent, high-context retrieval.
- **Developers**: Building advanced agentic workflows that require reliable and isolated memory storage.

## 3. Core Objectives
1.  **Semantic Retrieval**: Enable agents to find memories based on meaning rather than just keyword matches.
2.  **Intelligence-First Storage**: Automatically classify, score, and summarize memories upon ingestion.
3.  **Namespace Isolation**: Ensure strict sandboxing of memories between different agent profiles.
4.  **Desktop Friendly**: Zero-config, single-file persistence using SQLite.
5.  **Multi-Model Analysis**: Utilize local LLM-based models (BART) for deep semantic understanding.

## 4. Key Features

### 4.1. Hybrid Vector Search
Combines high-speed vector similarity (using `sqlite-vec`) with intelligence-based re-ranking.
- **Base Search**: K-nearest neighbors (KNN) on 384-dimension embeddings.
- **Re-ranking**: Final result order determined by a combination of similarity score and a calculated "Importance Score".

### 4.2. The Intelligence Layer (Enrichment)
Automated processing of every memory saved:
- **Zero-Shot Classification**: Categorizes memories into types (e.g., `TASK`, `USER_PREFERENCE`, `RUNTIME_ERROR`) using BART-Large-MNLI.
- **Semantic Importance Scoring**: Uses semantic entailment to assign a weight (1.0 - 5.0) to memories based on their criticality.
- **Summarization**: Generates a 1-2 sentence "semantic profile" (gist) for long memories using DistilBART.

### 4.3. Knowledge Graph (Associations)
Links memories together via logical relationships:
- Support for types like `CAUSES`, `DEPENDS_ON`, `CONTRADICTS`, and `PRECEDES`.
- Enables root-cause analysis and complex reasoning across disparate memory entries.

### 4.4. Identity & Provenance
A robust model to track memory origin and ownership:
- `author_id`: The human user.
- `owner_id`: The specific agent profile (Namespace).
- `actor_id`: The service or agent that wrote the data.
- `session_id` & `workspace`: Audit trails for environmental context.

## 5. Technical Stack
- **Languages**: Python 3.x (Plugin logic), SQL (Persistence), Go (Hermes Interface Bridge).
- **Database**: SQLite 3 with `sqlite-vec` extension.
- **Models**:
    - **Embeddings**: `all-MiniLM-L6-v2` (384 dimensions).
    - **Classification/Scoring**: `facebook/bart-large-mnli`.
    - **Summarization**: `sshleifer/distilbart-cnn-12-6`.

## 6. Data Schema

### 6.1. `memories` Table
Stores raw content, metadata, and calculated scores.
- `content_full`: Original text.
- `content_abstract`: Generated summary.
- `memory_type`: Categorization.
- `importance_score`: 1.0 to 5.0.

### 6.2. `memories_vec` Table
Virtual table for vector search indices.

### 6.3. `memory_associations` Table
Maps relationships between memory IDs.

## 7. Performance Requirements
- **Local Inference**: All enrichment and vectorization must run locally on CPU/GPU.
- **Retrieval Speed**: Nearest neighbor search should complete in sub-100ms for thousands of records.
- **Disk Footprint**: Minimal, leveraging the efficiency of SQLite.

## 8. Success Metrics
- **Relevance**: Increased precision in agent context retrieval compared to standard keyword search.
- **Reliability**: Zero data leakage between profiles (`owner_id` isolation).
- **Integrity**: 100% of memories are successfully categorized and vectorized.
