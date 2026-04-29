# MesaService: Active Cognitive Maintenance

> "Less Noise = Smarter Agent."

In most AI memory systems, data is treated as an ever-growing pile. In **Reverie Core**, memory is treated as a biological system requiring active maintenance. **MesaService** is the "janitor service" that performs **Active Cognitive Maintenance**, ensuring that the agent's long-term retrieval always prioritizes signal over noise.

---

## 1. The Philosophy: Signal vs. Noise

Most AI developers believe that more data results in a more capable agent. Our benchmarks prove that **irrelevant data acts as a cognitive weight**, dragging down reasoning accuracy and increasing token costs.

MesaService implements a "Sanitization" strategy that moves beyond simple time-based TTL (Time-To-Live). Instead, it uses a **Multi-Stage Pruning** approach based on semantic value, graph centrality, and usage history.

### Strategic Benefits:
1.  **Lower Token Costs**: Prevents wasting the context window on "weather reports" or transient chatter.
2.  **Higher Reasoning Accuracy**: By providing 100% relevant "Gold Facts," the LLM is significantly less likely to hallucinate.
3.  **Dynamic smoothing**: Automatically distinguishes between "Reference Material" (Anchors) and "Scratchpad Noise" (Transient).

---

## 2. Static vs. Dynamic Pruning

ReverieCore utilizes pruning in two distinct ways, both powered by the **PruningEngine**:

1.  **Dynamic Pruning (Retrieval)**: Occurs in real-time during a search. It uses `relative_threshold` to discard "weak" results that aren't as good as the top performer.
2.  **Static Pruning (Mesa)**: Occurs in the background. It evaluates the "baseline" quality of stored memories and archives those that consistently fail to provide value.

---

## 3. Tiered Maintenance Strategy

MesaService operates in three distinct tiers to balance database health with safety.

### Tier 1: Soft-Prune (Daily)
This tier identifies **Fragmented Memories** and shifts them to an `ARCHIVED` status. 
A memory is a candidate for archiving if it meets **all** of the following criteria:
- **Low Importance**: `importance_score < 4.0` (Not an Anchor).
- **Stale**: `last_accessed_at` is older than 14 days.
- **Low Centrality**: The memory has `< 2` associations in the Knowledge Graph.
- **Low Retention Probability**: Fails the `retention_threshold` check (default 0.4).

### Tier 1.5: Hierarchical Consolidation (Daily)
Instead of purely archiving, Mesa identifies clusters of stale/fragmented memories that share a common entity.
- **Crystallization**: If **>= 5** memories share an entity and meet pruning criteria, they are "crystallized" into a high-level **Observation Anchor**.
- **The Hierarchy**: Original memories are preserved as "Child Fragments" (`CHILD_OF`) linked to the anchor.
- **Status Shift**: Fragments move to `ARCHIVED` to clear the primary search, ensuring the agent sees the "Wisdom" (summary) first.

### Tier 2: Deep Clean (Monthly)
This tier manages physical database health and permanent data removal.
- **Purge**: Permanently `DELETE` records in `ARCHIVED` status for more than 90 days.
- **Optimization**: Triggers a database `VACUUM` to reclaim disk space.

---

## 4. Technical Implementation

### PruningEngine Integration
Mesa leverages the same `PruningEngine` logic used in retrieval. By applying `min_absolute_score` filters during maintenance scans, Mesa can identify "junk" memories that were never useful even when first ingested.

### Concurrency Support
MesaService runs in a background thread and uses **WAL (Write-Ahead Logging)** mode. This allows the maintenance service to perform intensive scanning and updates without blocking the agent's primary retrieval or storage operations.

---

## 5. Configuration

Maintenance thresholds are tuned in `.reveriecore.yaml` under `maintenance.mesa`:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `centrality_threshold` | `2` | Min graph edges required to remain ACTIVE. |
| `retention_days` | `14` | Days since last access before a memory becomes "stale". |
| `importance_cutoff` | `4.0` | Importance score threshold for "Anchored" status. |
| `consolidation_threshold` | `5` | Min memories for hierarchical consolidation. |
| `interval_seconds` | `3600` | Frequency of maintenance checks. |
| `pruning.retention_threshold` | `0.4` | The "Quality Floor" for background retention. |

---

## 6. Value for Cloud Placement (AaaS)

When deployed to an **Agent-as-a-Service** environment, MesaService provides a competitive edge:
- **Clean MCP context**: Claude and other LLMs receive high-fidelity context every time.
- **Long-term Stability**: Prevents "Brain Rot" from session clutter.
- **Managed Storage**: Keeps DB size predictable for multi-tenant environments.
