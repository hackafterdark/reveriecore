# Technical Specification: Memory Anchoring System

## 1. Overview
To ensure agent context retrieval is efficient, accurate, and deterministic, we are moving away from "flat" memory retrieval. We are implementing **Virtual Anchors**—a multi-dimensional filtering and ranking strategy that mimics the cognitive benefit of spatial memory (contextual grouping) while maintaining the precision of a structured SQL database.

## 2. Anchor Taxonomy
We define three primary categories of anchors used to narrow the search space:

### A. Categorical Anchors (Context)
* **`workspace`**: The local working directory or project root. This acts as the "Top-level Folder" for the memory.
* **`session_id`**: The interaction timeline. This allows the agent to "zoom in" on the current task's thread of reasoning.
* **`owner_id`**: The sandboxed identity/profile (ensures strict namespace isolation).

### B. Sequential Anchors (Temporal)
* **`learned_at`**: The timestamp of memory creation.
* **Role**: Used to implement **Temporal Contiguity** (recency bias) and timeline reconstruction.

### C. Semantic Anchors (Content)
* **`memory_type`**: Hard classification (e.g., `TASK`, `RUNTIME_ERROR`) to allow deterministic filtering.
* **Entities & Triples**: Hard-coded relationships extracted during enrichment to allow "Entity-Linked" retrieval.

## 3. Implementation Plan

### Phase 1: Scoring & Ranking (The `Retriever` Layer)
* **Requirement**: Modify the `Retriever` query to move from pure `cosine_similarity` to a weighted `final_score`.
* **Algorithm**: 
    `final_score = (similarity * w1) + (importance * w2) + (time_decay * w3)`
* **Action**: Implement the `time_decay` function as a logarithmic function based on `learned_at` relative to `CURRENT_TIMESTAMP`.

### Phase 2: Anchor-Aware Queries
* **Requirement**: Add optional filtering arguments to the `search()` method.
* **Signature**: 
    `def search(self, query: str, workspace: str = None, session_ids: list = None, type_filter: str = None):`
* **Action**: If parameters are provided, dynamically inject `WHERE` clauses into the SQL query before performing the `vec` search.

### Phase 3: The "Closing Summary" (Session Solidification)
* **Requirement**: Automate the creation of high-importance anchor points.
## 3. Implemented Architecture

### Phase 1: The Multi-Dimensional Scoring Engine
The `Retriever` implements a weighted ranking system that combines three primary signals:
- **Vector Similarity ($w_1=0.5$)**: Semantic proximity of text.
- **BART Importance ($w_2=0.3$)**: The objective value of the fact.
- **Temporal Decay ($w_3=0.2$)**: Recency bias.
- **Graph Anchor Boost ($+0.2$)**: A hard bonus for memories directly linked to query entities.

### Phase 2: Importance-Aware Decay
To preserve core project knowledge while keeping the conversation "fresh":
- **Low Importance (< 4.0)**: Uses a **48-hour half-life exponential decay**. Memories naturally fade to prioritize the current task.
- **High Importance (>= 4.0)**: Assigned **Permanent Weight**. Critical architectural facts or user preferences never decay.

### Phase 3: Behavioral Controls (Freshness Mode)
To prevent "Behavioral Anchoring" (where the agent is trapped by its history), the system includes an automated **Freshness Detection** trigger.
- **Keywords**: "clean slate", "fresh start", "forget history", "new project".
- **Operation**: If detected, the `Retriever` bypasses the Knowledge Graph and Temporal Decay, performing a pure semantic slice of the vector index.

---

## 4. Future Roadmap: The "Solidification" Phase

### A. Session Solidification (The "Archive" Step)
Automate the creation of high-level anchor points after a session ends:
- **Trigger**: Inactivity > 30 minutes OR manual "Wrap Up" command.
- **Action**: The LLM summarizes the key `TASK` and `LEARNING_EVENT` entries into a single high-importance `OBSERVATION` memory linked to all entities mentioned.

### B. Influence Ranking (Feedback Loops)
- **Concept**: If the agent successfully answers a query using a specific memory, that memory's `importance_score` should experience a small "Relational Boost." 
- **Goal**: Popular/useful memories become stronger anchors over time.

### C. Negative Anchors
- **Concept**: Explicitly mark memories that are "Outdated" or "Superseded."
- **Logic**: During Graph-Led Search, if a node is linked via a `SUPERSEDES` edge, the retriever should prioritize the target and suppress the source.
