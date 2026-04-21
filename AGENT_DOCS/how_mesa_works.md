# MesaService: Active Cognitive Maintenance

> "Less Noise = Smarter Agent."

In most AI memory systems, data is treated as an ever-growing pile. In **Reverie Core**, memory is treated as a biological system requiring active maintenance. **MesaService** is the "janitor service" that performs **Active Cognitive Maintenance**, ensuring that the agent's long-term retrieval always prioritizes signal over noise.

---

## 1. The Philosophy: Signal vs. Noise

Most AI developers believe that more data results in a more capable agent. Our benchmarks prove that **irrelevant data acts as a cognitive weight**, dragging down reasoning accuracy and increasing token costs.

MesaService implements a "Sanitization" strategy that moves beyond simple time-based TTL (Time-To-Live). Instead, it uses a **Two-Tiered Maintenance** approach based on semantic value, graph centrality, and usage history.

### Strategic Benefits:
1.  **Lower Token Costs**: Prevents wasting the context window on "weather reports" or transient chatter.
2.  **Higher Reasoning Accuracy**: By providing 100% relevant "Gold Facts," the LLM is significantly less likely to hallucinate.
3.  **Dynamic smoothing**: Automatically distinguishes between "Reference Material" (Anchors) and "Scratchpad Noise" (Transient).

---

## 2. Tiered Maintenance Strategy

MesaService operates in two distinct tiers to balance database health with safety.

### Tier 1: Soft-Prune (Daily)
This tier identifies **Fragmented Memories** and shifts them to an `ARCHIVED` status. 
A memory is a candidate for archiving if it meets **all** of the following criteria:
- **Low Importance**: `importance_score < 4.0` (Not an Anchor).
- **Stale**: `last_accessed_at` is older than 14 days.
- **Low Centrality**: The memory has `< 2` associations in the Knowledge Graph.
- **Status Independent**: It hasn't already been consolidated.

### Tier 2: Deep Clean (Monthly)
This tier manages physical database health and permanent data removal.
- **Purge**: Permanently `DELETE` records in `ARCHIVED` status for more than 90 days.
- **Optimization**: Triggers a database `VACUUM` outside of any transactions to reorganize the file and reclaim disk space.

---

## 3. The Secret Weapon: Recency Protection

MesaService features a mechanism we call **Dynamic Smoothing** (or Recency Protection). 

> [!IMPORTANT]
> Because every search hit in `retrieval.py` updates the `last_accessed_at` timestamp, MesaService automatically "spares" any memory that has been used in a recent workflow—even if that memory is low-importance or floating alone in the graph.

This ensures that "reference books" you are currently using stay on the desk, while those you haven't touched in weeks are moved to the basement (Archive).

---

## 4. Technical Implementation

### Pruning Logic (SQL)
Mesa identifies candidates using a comprehensive `LEFT JOIN` between the memory and associations tables:

```sql
SELECT m.id FROM memories m
LEFT JOIN (
    SELECT source_id as node_id FROM memory_associations WHERE source_type = 'MEMORY'
    UNION ALL
    SELECT target_id as node_id FROM memory_associations WHERE target_type = 'MEMORY'
) a ON m.id = a.node_id
WHERE m.importance_score < ?
  AND m.last_accessed_at < datetime('now', ?)
  AND m.status = 'ACTIVE'
GROUP BY m.id
HAVING COUNT(a.node_id) < ?
```

### Concurrency Support
MesaService runs in a background thread and uses **WAL (Write-Ahead Logging)** mode. This allows the maintenance service to perform intensive scanning and updates without blocking the agent's primary retrieval or storage operations.

---

## 5. Configuration

Maintenance thresholds can be tuned in the Hermes `config.yaml` under the memory section:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `mesa_centrality_threshold` | `2` | Minimum graph edges required to remain ACTIVE. |
| `mesa_retention_days` | `14` | Days since last access before a memory becomes "stale". |
| `mesa_importance_cutoff` | `4.0` | Importance score threshold for "Anchored" status. |
| `mesa_interval_seconds` | `3600` | Frequency of maintenance checks. |

---

## 6. Value for Cloud Placement (AaaS)

When deployed to an **Agent-as-a-Service** environment, MesaService provides a competitive edge:
- **Clean MCP context**: When plugging this into Cursor or Claude Code, they receive high-fidelity context every time.
- **Long-term Stability**: Prevents the "Brain Rot" often seen in long-running sessions where agents become confused by their own history.
- **Managed Storage**: Keeps DB size predictable for multi-tenant environments.
