# ADR: Signal-Aware Iterative Expansion

## Status
Accepted (2026-04-20)

## Context
Early graph-traversal implementations utilized unconditional 2-hop or 3-hop recursive expansions. In dense knowledge graphs, this led to "The Ambient Noise Problem," where relevant memories were diluted by a large volume of tangentially related nodes (friends-of-friends). This resulted in lower **Faithfulness** scores as the LLM struggled to identify the core context within the noise.

## Decision
We will transition to an **Iterative, 1-Hop Dominant** retrieval strategy. The system will favor immediate structural neighbors and only "deepen" the search if the primary signal is weak.

### Key Logic
1. **Initial Layer Pass:** The retriever performs a `depth=1` traversal to find direct associations.
2. **Signal Strength Calibration:** Expansion stops at 1-hop if the result set meets the "High Confidence" heuristic:
    - Result count >= 3
    - Average discovery score >= 0.6
3. **Adaptive Fallback:** The system triggers a 2-hop search *only* if the initial layer fails to meet these thresholds.

## Consequences

### Positive
- **Increased Faithfulness:** Significant reduction in context dilution, as confirmed by RAGAS benchmarks (Faithfulness increased from 0.85 to 0.87).
- **Reduced Hallucination:** Prevents the agent from forming speculative links between distant, unrelated nodes in the graph.
- **Improved Performance:** Fewer recursive SQL joins for high-confidence queries, reducing overhead.

### Negative
- **Recall Sensitivity:** There is a minor risk of missing deep semantic links if the 1-hop signal is "decoy" high-signal but misleading.
