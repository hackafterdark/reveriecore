# Hybrid-RAG: How Memory Retrieval Works

Retrieval in the Hermes Memory Plugin is a multi-step process that combines **Semantic Similarity** (Vector Search) with **Intelligence-Based Re-ranking** (Scoring).

### 🧠 The Workflow: From Query to Context

When the Agent needs context, it doesn't just search for keywords; it asks a high-level question. Our Python plugin handles the heavy lifting of translating that question into the best possible memories.

$$\text{Agent Needs Context} \xrightarrow{1. Query Formulation} \text{Agent} \xrightarrow{2. Call } \text{PYTHON PLUGIN} \xrightarrow[\text{3. Embed/Hybrid-Search}]{\text{Intelligence Layer}} \text{SQLite (sqlite-vec)} \xrightarrow{4. Context Injection} \text{Agent Window}$$

### 🛠️ The Implementation Bridge

Our plugin implements the `retrieve_memories(query: str, limit: int)` function, which executes three primary jobs:

#### 1. Vectorization (The Map)
The raw query is passed to `sentence-transformers`. This converts the agent's intent into a 384-dimension vector ($V_q$).

#### 2. SQLite Similarity & Namespace Search
Using the `sqlite-vec` extension, we perform a nearest-neighbor search on the `memories_vec` virtual table while enforcing **Namespace Isolation**.
- **The SQL:** 
  ```sql
  SELECT m.id, v.distance 
  FROM memories_vec v JOIN memories m ON v.rowid = m.id
  WHERE v.embedding MATCH $V_q$ 
    AND (m.owner_id = $profile OR m.privacy = 'PUBLIC')
    AND m.status = 'ACTIVE'
    AND v.k = $limit * 2;
  ```
- This ensures an agent only retrieves memories belonging to its current Profile or global Public facts, while automatically skipping "Archived" or "Superseded" noise.

#### 3. Graph Augmentation (Graph-RAG)
Similarity isn't everything. A memory that is semantically similar to your query might be linked to a vital "hidden" fact that doesn't share keywords. 
- **Bidirectional Traversal**: After finding the top vector candidates (Seeds), we traverse the `memory_associations` table in **both directions** (`Memory <-> Entity <-> Entity <-> Memory`).
- **Shared Entity Bridging**: This allows us to find Memory B if it shares an entity with Memory A, even if the query only semantically matched Memory A.
- **Hub Protection**: To prevent popular entities (the "Hub Problem") from flooding results, we strictly limit traversal to the **top 10 associations per node**.

#### 4. Intelligence-Based Re-ranking (The Filter)
Final ranking combines three signals:
1. **Vector Similarity**: How well the text matches the query.
2. **BART Importance**: A zero-shot score (1.0 to 5.0) for the fact's objective value.
3. **Graph Proximity**: A boost for memories discovered via graph links.

$$\text{Final Rank} = (W_1 \times \text{Similarity}) + (W_2 \times \text{Importance}) + \text{Graph Boost}$$

#### 5. Agentic Drill-Down (Recall Reverie)
While the system prioritizes high-level **Observation Anchors** in the initial search to save tokens, it never truly "forgets" the nuance.
- **Discovery**: Search results for Observations automatically append a list of `[Nuanced Details available via recall_reverie for IDs: [102, 105, ...]]`.
- **The Trigger**: The agent can explicitly use the `recall_reverie(memory_id)` tool to fetch specific "Gritty Details."
- **Security Check**: Every drill-down call performs a strict provenance check to ensure the agent is authorized to view the fragment through its parent hierarchy.

#### 6. Recency Protection (The Maintenance Signal)
Every time a memory is successfully retrieved (or recalled via `recall_reverie`), the system updates its `last_accessed_at` timestamp. This acts as a "Stay Alive" signal to the **MesaService**, shielding active context from background pruning even if the memory has low objective importance.

### 📊 Context Hub: Adaptive Memory Budgeting

To prevent long-term memory from consuming the agent's active context window, Reverie Core implements a dynamic budgeting system that listens to signals from Hermes.

#### 1. Usage Zones
We monitor `remaining_tokens` (provided via `on_turn_start`) to determine how much memory to inject:

| Zone | Condition | Strategy | Output Type |
| :--- | :--- | :--- | :--- |
| **Comfort** | >50% Remaining | Balanced | Full Content (Preferred) |
| **Tight** | 20-50% Remaining | Conservative | Weighted towards Abstracts |
| **Danger** | <20% Remaining | Critical | Abstracts ONLY |

#### 2. The Fallback Mechanism
During retrieval, the `Retriever` iterates through the most relevant memories and makes a real-time decision:
1. **Check Full**: Does `content_full` fit in the remaining budget? If yes, use it.
2. **Check Abstract**: If full is too large, does the summary/abstract fit? If yes, use the abstract.
3. **Skip**: If even the abstract doesn't fit, skip this memory to prioritize higher-ranking results.

### 🚀 Key Takeaway
Our plugin ensures the Agent doesn't just get *any* memory, but the **most relevant and important** memory for the current situation—while strictly staying within the model's token constraints to avoid performance degradation or forced context compression.