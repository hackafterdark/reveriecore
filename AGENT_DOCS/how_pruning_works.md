# How Pruning Works in ReverieCore

Pruning is a critical quality-control mechanism in the ReverieCore retrieval pipeline. It acts as a "Quality Gate" that filters out low-signal or irrelevant information immediately after re-ranking, ensuring that only the most pertinent memories are passed to the budgeting and context-filling stages.

---

## ⚙️ Configuration Parameters

Pruning is governed by three primary values in your `reveriecore.yaml` under the `retrieval.pruning` section:

### 1. `top_n`: The "Quota" Gate
`top_n` is an **absolute limit** on the number of candidates allowed to pass through the gate. It prioritizes volume control over quality nuance.

*   **Logic:** "Take the top X."
*   **Behavior:** It ensures your context window never overflows by capping the result count. However, it will keep low-scoring items just to fill the quota if `relative_threshold` is not set.

### 2. `relative_threshold`: The "Quality" Gate
`relative_threshold` is a **contextual standard** based on the performance of your best result. It ensures that every candidate is "good enough" compared to the top performer.

*   **Logic:** "Everyone must be at least X% as good as the star player."
*   **The Math:** If your top candidate has a score of `0.9` and your `relative_threshold` is `0.5`, the minimum score for any other candidate is `0.45` ($0.9 \times 0.5$).
*   **Behavior:** It dynamically shrinks the result set when the search quality drops off sharply.

### 3. `min_absolute_score`: The "Hard Floor"
This is a safety net that prevents the pruner from keeping "junk" results even if they are relatively strong compared to an even worse top result.

*   **Logic:** "No matter what, don't keep anything below this score."
*   **Behavior:** If your top result is `0.2` (very poor) and your `min_absolute_score` is `0.3`, the pruner will discard *everything*, protecting the LLM from hallucinating based on irrelevant context.

---

## 🏃 The Talent Scout Analogy

Think of your retrieval pipeline like a talent scout looking for the best team.

| Feature | `top_n` | `relative_threshold` |
| :--- | :--- | :--- |
| **Philosophy** | "We have a fixed budget/space." | "We only want quality performers." |
| **Primary Goal** | Prevents context window overflow. | Prevents low-quality noise. |
| **Flexibility** | Rigid; always returns the same count. | Dynamic; returns fewer items if quality is low. |
| **Logic** | "Take the top X." | "Take everyone who is 'good enough' compared to the best." |

### Better Together
In ReverieCore, these work as a **two-step filter**:
1.  **`top_n` acts as the safety net:** It ensures you never trigger a massive token bill by dumping too many nodes into the LLM.
2.  **`relative_threshold` acts as the quality filter:** It ensures that within your quota, you aren't including "junk" just to fill spots.

---

## 🔄 System Integration

### Retrieval Pipeline
In the retrieval pipeline (`retrieval.py`), the `PruningHandler` is positioned **immediately after the Reranker**. 
- The Reranker calculates high-precision scores.
- The PruningHandler immediately applies the gates above to mutate the candidate list.
- This results in a cleaner, higher-density set of candidates for the `BudgetHandler` to process.

### Mesa Service (Maintenance)
Pruning principles are also applied asynchronously by the `MesaService` for database maintenance.
- **Background Pruning:** Uses a `retention_threshold` to identify memories that consistently fail to meet quality standards over time.
- **Crystallization:** Helps decide which memories are "strong" enough to be crystallized into higher-level nodes or archived to save vector space.

---

## 🛠️ Summary for Agents

- **Stable:** `top_n` keeps the system predictable for token budgeting.
- **Smart:** `relative_threshold` keeps the system relevant by filtering out the "tail" of weak results.
- **Safe:** `min_absolute_score` provides a final line of defense against irrelevant searches.
