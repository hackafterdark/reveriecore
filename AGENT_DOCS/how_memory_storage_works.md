# The Intelligence Layer: How Memory Storage Works

In the Hermes Memory Plugin, storage is not just about saving strings; it's about **enrichment**. Our plugin acts as the bridge between raw Agent activity and a structured knowledge base.

### 🔑 The Execution Flow: `save_memory()`

When the Hermes Agent triggers a memory save, it sends raw, unstructured text. Our Python plugin performs the "intelligence" logic before anything touches the disk.

$$\text{Agent} \xrightarrow{\text{Raw Text}} \text{PYTHON PLUGIN} \xrightarrow[\text{1. Classify, 2. Score, 3. Profile, 4. Embed}]{\text{Intelligence Layer}} \text{SQLite DB}$$

#### 1. Classification (BART Zero-Shot)
Instead of brittle keywords, we use a **BART-Large-MNLI** model to perform zero-shot classification on the raw content. This accurately maps memories into the `MemoryType` taxonomy (e.g., `TASK`, `RUNTIME_ERROR`, `USER_PREFERENCE`).

#### 2. Importance Scoring (Grated Cascading Pipeline)
The plugin calculates the `importance_score` (0.0 - 10.0) using a tiered **Cascading Pipeline**. This system optimizes for both speed and identity-alignment:
- **Heuristics (Tier 1)**: Instant keyword/structural detection for errors and deadlines.
- **BART (Tier 2)**: Local zero-shot semantic weight assessment.
- **Soul (Tier 3)**: High-fidelity LLM assessment relative to the agent's active personality.
The system uses **Confidence Thresholds** to "early-exit" the pipeline the moment a reliable score is found, ensuring minimal latency.

#### 3. Identity and Audit Tracking
Every memory is tagged with five identity/audit fields to ensure enterprise-grade provenance:
- `author_id`: The human creator.
- `owner_id`: The current agent profile (Namespace).
- `actor_id`: The specific service (e.g., `REVERIE_SYNC_SERVICE`).
- `session_id`: The active session UUID.
- `workspace`: The local working directory path.

#### 4. Semantic Profiling & Vectorization
For long transcripts, the plugin runs a **Lazy-Loaded Summarizer** (DistilBART) to generate a 1-2 sentence "gist". This profile is then vectorized into a 384-dimension embedding for storage in the `memories_vec` table.

---

### 🛡️ Data Integrity via Python Enums

To prevent schema corruption and ensure consistent agent behavior, we enforce strict type definitions using Python's `Enum` class.

```python
# reveriecore/schemas.py
class MemoryType(Enum):
    CONVERSATION = "CONVERSATION"
    TASK = "TASK"
    OBSERVATION = "OBSERVATION"
    USER_PREFERENCE = "USER_PREFERENCE"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    CODE_SNIPPET = "CODE_SNIPPET"
    LEARNING_EVENT = "LEARNING_EVENT"
    EXPIRED_TASK = "EXPIRED_TASK"
```

### Summary of Responsibilities

| Component | Industry Role | Responsibility |
| :--- | :--- | :--- |
| **Hermes Agent** | **Client** | Decides *what* to save and *when*. |
| **Python Plugin** | **Intelligence Layer** | Decides the *type*, calculates *score*, and runs *embeddings*. |
| **SQLite + vec0** | **Persistence Layer** | Stores the structured relational data and vector indices. |
| **MesaService** | **Maintenance Layer** | Background engine that prunes noise and archives stale data. |

### 🧹 Post-Storage: The Mesa Maintenance Cycle

Storing a memory is only the first half of the lifecycle. Once a memory is saved, the **MesaService** begins monitoring it. If a memory remains low-importance and is not accessed or linked to other facts within 14 days, it is automatically transitioned to `ARCHIVED` status to prevent "Brain Rot" and ensure long-term retrieval remains high-signal.
