# ADR: Soul-Aware Cascading Importance Pipeline

## Status
Accepted (2026-04-24)

## Context
Importance scoring in ReverieCore was previously reliant on a local BART-based zero-shot classifier. While effective for general categorization, it lacked the ability to prioritize information relative to the agent's specific role, goals, or identity (the "Soul"). Furthermore, executing a full classifier on every turn introduced unnecessary latency for obvious high-signal events (errors, deadlines) or low-signal conversational noise.

## Decision
We will implement a **Grated Cascading Pipeline** using a formal handler-based interface. This ensures that the scoring logic is decoupled, composable, and capable of "early-exiting" when a step provides a high-confidence score.

### The Scoring Interface
All scoring logic must implement the `ImportanceHandler` interface, which guarantees a standard output:
```python
{
    "score": float,         # 0.0 to 10.0
    "confidence": float,    # 0.0 to 1.0
    "handler_name": str     # e.g., "bart", "soul"
}
```

### The Cascading Pipeline
The `EnrichmentService` iterates through a list of registered handlers, each associated with a **Confidence Threshold**:
1.  **HeuristicHandler** (Threshold: 0.9): Instant regex/keyword matches. If a hit occurs (Confidence 1.0), the pipeline exits immediately.
2.  **BARTHandler** (Threshold: 0.8): Local zero-shot classification. If the maximum label probability is high enough, the pipeline exits.
3.  **SoulHandler** (Threshold: 0.0): Identity-relative LLM pass. Acting as the "final word," it is only called if local/heuristic checks are inconclusive.

## Consequences

### Positive
- **Composable Architecture**: New scoring logic (e.g., domain-specific scanners) can be added as new `ImportanceHandler` subclasses without modifying the core pipeline.
- **Latency Optimization**: Cheaper local handlers (Heuristics, BART) are given the first opportunity to score, significantly reducing remote LLM calls.
- **Explicit Confidence**: Each layer is responsible for reporting its own certainty, making the scoring process transparent and easier to debug.
- **Dynamic Identity**: The pipeline honors dynamic personality changes (via `/personality` command), allowing importance to shift as the user's workflow evolves.

### Negative
- **LLM Dependency**: High-fidelity scoring requires a reachable LLM provider.
- **Prompt Sensitivity**: The quality of Soul-aware scoring is dependent on the clarity of the `SOUL.md` or personality prompt.
