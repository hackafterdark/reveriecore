# ADR: Multiplicative Semantic Gravity

## Status
Accepted (2026-04-20)

## Context
In early Graph-RAG iterations, semantically relevant "anchors" (nodes identified in the user query) were boosted using additive scores. However, in heterogeneous graphs where `confidence_score` varies widely (0.1 to 1.0), additive boosts often drowned out the "Ground Truth" confidence, causing low-confidence associations to leapfrog high-confidence ones simply because they shared a keyword with the query.

## Decision
We will utilize a **Multiplicative Discovery Score** to ensure that graph traversal remains grounded in association confidence while allowing query anchors to act as powerful force-multipliers.

### The Formula
`discovery_score = confidence_score * (1 + (is_anchor * gravity))`

- **`confidence_score`**: The base truth (source: `memory_relations` table).
- **`is_anchor`**: Binary (0 or 1), indicating if the node directly matches a query entity.
- **`gravity`**: A dynamic multiplier (0.5 to 1.1) derived from Cognitive Intent classification.

## Consequences

### Positive
- **Stable Rankings**: High-confidence associations are preserved even if they aren't direct anchors, preventing "opinionated" retrieval from overshadowing actual data links.
- **Deterministic Influence**: Anchors only gain influence if they *already* have a valid path found by the core algorithm.
- **Normalization Safeguard**: Because the boost is proportional, bad data (low-confidence links) cannot "force" its way into the result set via gravity alone.

### Negative
- **Threshold Sensitivity**: Requires more careful tuning of the `gravity` parameter, as a multiplier of 1.1 has a far more significant impact than an additive +0.1 when applied to a 0.9 confidence score.
