### 🧠 Deep Dive: Importance Score Mechanics and Behavior

Importance scoring in ReverieCore ensures that the agent prioritizes relevant, high-signal information while effectively managing "noise." The system uses a **Grated Cascading Pipeline** to balance speed, fidelity, and identity-awareness.

#### 1. The Grated Pipeline (ImportanceHandler Interface)

Instead of a fixed logic block, ReverieCore uses a series of independent **Handlers**. Each handler assesses the text and reports both a **Score** and its **Confidence** in that score.

The pipeline iterates through these handlers and "Early Exits" the moment a handler's confidence meets or exceeds its defined **Threshold**.

| Handler | Type | Speed | Threshold | Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **Heuristic** | Local | < 1ms | **0.9** | Regex/Keyword patterns for errors and code. |
| **BART** | Local | ~300ms | **0.8** | Zero-shot classification (mDeBERTa-v3). |
| **Soul** | Remote | ~1.5s | **0.0** | Identity-relative LLM assessment. |

**A. Confidence Math**
- **Heuristics**: Binary. A "hit" returns 1.0 confidence; a "miss" returns 0.0.
- **BART**: Uses the maximum softmax probability among the labels (Critical, Important, Minor, Trivial).
- **Soul**: The LLM itself provides a confidence estimate alongside the score.

#### 2. Intelligence Benefits
*   **Speed-Optimized**: Local models handle 80% of scoring duties, reserving the expensive LLM for ambiguous or identity-critical cases.
*   **Extensible**: New handlers (e.g., a "Legal Doc Scorer") can be added by implementing the `ImportanceHandler` interface.
*   **Identity Alignment**: An agent with a "System Administrator" persona will naturally rank server logs higher than recipe discussions.
*   **Latency Optimization**: Tier 1 handles "obvious" importance instantly, ensuring the system remains responsive even without a remote LLM.
*   **Adaptive Retention**: Low-importance memories (Score < 5.0) are automatically marked for earlier pruning (see [Mesa Maintenance](how_mesa_works.md)).

#### 3. Developer Interface: `ImportanceHandler`

To support future configuration and custom plugins, scoring logic is implemented using a formal interface:

```python
class ImportanceHandler(ABC):
    @abstractmethod
    def process(self, text: str) -> Dict[str, Any]:
        """
        Must return standard result:
        {
            "score": float,       # 0.0 - 10.0
            "confidence": float,  # 0.0 - 1.0
            "handler_name": str
        }
        """
```

New scoring logic (e.g., domain-specific scanners) can be added by implementing this class and registering it in the `EnrichmentService` pipeline.

#### 4. Retrieval Integration (How the Score is Used)

When the agent retrieves memories, the importance score acts as a secondary filter and re-ranker:
1.  **Semantic Search**: Finds the Top K matches based purely on vector similarity.
2.  **Relational Boost**: Boosts nodes based on graph connectivity (centrality).
3.  **Importance Tuning**: Final results are weighted by their importance score to ensure that "Critical" facts surface even if they have slightly lower semantic similarity than "Trivial" conversation.

$$\text{Final Rank} = (\text{Similarity}) + \text{Centrality} + (\text{Importance} \times 0.2)$$