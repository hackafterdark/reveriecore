# Custom Database Memory Plugin Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** To replace or significantly augment Hermes' default markdown file memory with a robust, searchable database system (RAG capable) using **SQLite** as the desktop-friendly embedded database and the **`sqlite-vec` extension** for vector similarity search.

**Architecture:**
The solution will consist of a dedicated Hermes Memory Provider Plugin written in **Python**. This plugin will abstract data storage and retrieval logic. The data layer uses SQLite to store memory entries, metadata, and handles vector embeddings via a virtual table (`vec0`). The **"Intelligence Layer"** within the plugin analyzes raw data to determine its type and importance before storage. Embedding generation is handled locally using `sentence-transformers`.

**Tech Stack:**
- **Core Plugin Logic:** Python 3.x
- **Database:** SQLite 3 with `sqlite-vec` extension.
- **Embeddings:** `sentence-transformers` (Model: `all-MiniLM-L6-v2`, 384 dimensions).
- **Enrichment:** `transformers` (pipeline) for summarization and sentiment analysis.

---

## Phase 1: Foundation & Proof of Concept (POC)
### Task 1: Environment Setup & Vector POC
**Objective:** Establish a Python environment capable of loading `sqlite-vec` and generating embeddings.
**Files:**
- [NEW] `memory_plugin/requirements.txt`
- [NEW] `memory_plugin/poc_vector_db.py`
**Step 1: Dependency Management:** Define `requirements.txt` with `sqlite-vec`, `sentence-transformers`, and `transformers`.
**Step 2: Vector DB POC:** Write a script to initialize a SQLite database, load the `sqlite-vec` extension, and create a `VIRTUAL TABLE` using `vec0`.
**Step 3: Embedding POC:** Integrate `SentenceTransformer('all-MiniLM-L6-v2')` to generate a 384-dimension vector and insert it into the virtual table.
**Step 4: Commit:** Commit POC scripts.

## Phase 2: Full Plugin Development (The Intelligence Layer)
### Task 2: Data Schema & Type Integrity
**Objective:** Implement the dual-table schema (Relational + Virtual) and define Python constants for memory types.
**Files:**
- [NEW] `memory_plugin/schema.py`
- [NEW] `memory_plugin/types.py`
**Step 1: Schema Definitions:** Create the `memories` relational table (content, actor, etc.) and the `memories_vec` virtual table.
**Step 2: Type Enums:** Use Python `Enum` to define strict categories for `MemoryType` and `RelationType`.
**Step 3: Commit:** Commit schema and type definitions.

### Task 3: Memory Enrichment Engine (saveMemory)
**Objective:** Implement logic to transform raw text into structured, scored, and vectorized memories.
**Files:**
- [NEW] `memory_plugin/enrichment_service.py`
**Step 1: Type Heuristics:** Implement logic to detect if a memory is a `RUNTIME_ERROR`, `TASK`, or `CONVERSATION`.
**Step 2: Importance Scorer:** Implement the scoring algorithm from `memory_importance_score.md` (sentiment boost, keyword detection, time decay).
**Step 3: Summarizer:** Use a `transformers` pipeline to generate a `semantic_profile` for long memories.
**Step 4: Commit:** Commit enrichment logic.

### Task 4: Association & Knowledge Graph
**Objective:** Link memories together using the `memory_relations` table.
**Files:**
- [NEW] `memory_plugin/association_manager.py`
**Step 1: Link Logic:** Implement functions to create relationships (e.g., `CAUSES`, `SUPPORTS`) between memory IDs.
**Step 2: Commit:** Commit association management.

## Phase 3: Hermes Integration
### Task 5: RAG Retrieval & Re-ranking
**Objective:** Implement the `retrieveMemories` function with hybrid similarity/importance ranking.
**Files:**
- [NEW] `memory_plugin/retriever.py`
**Step 1: Vector Search:** Query the `vec0` table for the top K semantic matches.
**Step 2: Importance Re-ranking:** Apply weights to the similarity score and importance score to produce the final result set.
**Step 3: Commit:** Commit the retrieval engine.

### Task 6: Hermes API Adaptor
**Objective:** Wrap the plugin logic in the interface expected by the Hermes Agent.
**Files:**
- [NEW] `memory_plugin/hermes_plugin.py`
**Step 1: Implement Interface:** Bridge `save_memory` and `retrieve_memories` to the Hermes runtime.
**Step 2: Integration Test:** Run a full cycle (Save -> Enrich -> Store -> Retrieve) locally.
**Step 3: Commit:** Commit the final adaptor.