# Knowledge Graph Extraction & Mechanics

Reverie Core is a hybrid **Graph-RAG** system. It combines vector similarity (for semantic context) with a deterministic **Knowledge Graph** (for hard, unambiguous rules and relationships). This allows for root-cause analysis, dependency tracking, and deterministic bridging between disparate memories.

---

## 1. The Value: "Ambiguous" to "Addressable"
If you store only raw text (even with vector embeddings), you rely on the model's ability to "guess" that `reveriecore`, `reverie`, and `the project` all refer to the same thing. This is non-deterministic and prone to hallucinations.

By performing entity extraction, we create **Canonical Identifiers**:
* **Raw:** "I fixed the issue in the reverie core repo."
* **Extracted Entity:** `{ "name": "ReverieCore", "type": "REPOSITORY" }`

When you query this, you aren't just searching for semantically similar text; you are querying a **hard link**. If your agent knows `ReverieCore` is the target, it can retrieve *every* memory related to that repository, regardless of whether the user called it "the repo" or "that project."

---

## 2. Implementation Strategy: Two-Pass Extraction

The `EnrichmentService` implements a **two-pass asynchronous pipeline** to ensure high-precision graph data. This only triggers if the memory has an **Importance Score >= 3.0**.

### Pass 1: Entity Identification
The LLM identifies **Technical Nouns**. 
- **Goal**: De-duplicate references (e.g., "The DB" and "database.py" map to the same entity).
- **Entities Supported**: `FILE`, `CLASS`, `FUNCTION`, `REPOSITORY`, `API_ENDPOINT`, `TOOL`.
- **Logic**: 
  1. Does this entity exist in the `entities` table by name?
  2. If Yes: Use existing `id`.
  3. If No: Create a new canonical record.

### Pass 2: Triple Extraction
Once nodes (entities) are resolved to IDs, the LLM is prompted to identify **Relationships** using the canonical names as anchors.
- **The Triple Pattern**: `(Subject) -> [Predicate] -> (Object)`.
- **Allowed Predicates**: `FIXES`, `CAUSES`, `DEPENDS_ON`, `PART_OF`, `IS_A`, `MENTIONS`.

---

## 3. Retrieval & Traversal Mechanics

### 3.1 Hybrid Graph-RAG Search
1.  **Semantic Map**: Perform a standard vector similarity search to find "Seed" memories.
2.  **Graph Augmentation**: From those seeds, perform a **Bidirectional Traversal** (`Memory <-> Entity <-> Memory`) up to 2 hops.
3.  **Context Injection**: The retrieved context includes "Linked Entity" breadcrumbs (e.g., `[FILE: database.py (MENTIONS)]`), giving the agent explicit relational hints.

### 3.2 Hub Protection (The "Noise" Filter)
To prevent "Hub-Explosion" (where a popular entity like `README.md` drags in 500 irrelevant memories), the system enforces:
- **Per-Node Limit**: Strictly follow only the **top 10 associations** per entity in the traversal.
- **Priority Ranking**: Associations are sorted by `confidence` and `association_type` priority.

### 3.3 Idempotency & Cleanup
- **Purge on Re-extract**: Re-running extraction for the same `memory_id` automatically purges old graph links before inserting new ones.
- **Deduplication**: Unique constraints on the `entities` table ensure that `database.py` is always a single, addressable node.

---

## 4. Technical Stack Highlights
- **Model Discovery**: Dynamically resolves the LLM provider (source URL and model name) from the central Hermes `config.yaml`.
- **Inference**: Uses the same LLM used by the Hermes agent, ensuring zero-config compatibility with Llama Swap, Ollama, and OpenAI agents.
