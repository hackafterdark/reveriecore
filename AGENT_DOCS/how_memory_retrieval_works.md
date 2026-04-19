# Hybrid Recall: How Memory Retrieval Works

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
  SELECT rowid, distance 
  FROM memories_vec 
  WHERE embedding MATCH $V_q$ 
    AND (m.owner_id = $profile OR m.privacy = 'PUBLIC')
    AND k = $limit * 2;
  ```
- This ensures an agent only retrieves memories belonging to its current Profile, or global Public facts.

#### 3. Intelligence-Based Re-ranking (The Filter)
Similarity isn't everything. A recent memory that is 80% similar might be more valuable than an old memory that is 90% similar. We use the **BART Large MNLI** model to calculate a zero-shot importance score (1.0 to 5.0) which is combined with the similarity score for final ranking:

$$\text{Final Rank} = (W_1 \times \text{Similarity Score}) + (W_2 \times \text{BART Importance Score})$$

- **Weights ($W_1, W_2$):** These allow us to tune whether the agent favors "exact semantic matches" or "important facts."

### 📊 Context Hub: Adaptive Memory Budgeting

To prevent long-term memory from consuming the agent's active context window, ReverieCore implements a dynamic budgeting system that listens to signals from Hermes.

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