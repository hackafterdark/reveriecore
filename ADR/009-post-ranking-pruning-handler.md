# ADR 009: Post-Ranking Pruning Handler

## Status
Accepted

## Context
As the retrieval pipeline in ReverieCore evolved to include high-precision re-ranking (via FlashRank/LLM), the gap between "Top Matches" and "Relevant Matches" became more apparent. A re-ranker might move a very relevant result to position #1 with a score of 0.9, but still leave "noise" candidates with scores like 0.1 at positions #5-10.

Historically, pruning logic was often bundled into either the Discovery phase (vector thresholds) or the final Budgeting phase (relevance floor). However, after re-ranking, we have the most accurate "quality signal" available.

## Decision
We decided to implement a dedicated `PruningHandler` and a stateless `PruningEngine` as a separate, composable stage. While it is included in the default ranking pipeline, it is independent of other handlers and operates on the final scores provided by whichever previous stages (Discovery, Scoring, or Reranking) are active.

## Rationale

### 1. Composability & Clean Separation
By keeping pruning separate from scoring mechanisms (like the re-ranker), we ensure that ReverieCore remains modular. Users can swap, remove, or reorder handlers without losing the ability to apply quality gates. The pruner focuses purely on **filtering** based on available score data.

### 2. Multi-Context Utility (Retrieval vs. Mesa)
By creating a stateless `PruningEngine`, we can reuse the same quality-gate logic (Top-N, Relative Threshold, Absolute Floor) in both:
- **Real-time Retrieval:** To minimize noise in the LLM context window.
- **Mesa Service (Maintenance):** To identify "junk" memories that should be archived or purged based on long-term quality trends.

### 3. Granular Configuration
A separate handler allows for explicit configuration in `.reveriecore.yaml`. Users can tune the "strictness" of their retrieval quality (e.g., `relative_threshold: 0.5`) independently of their discovery or re-ranking strategies.

### 4. Telemetry Clarity
A dedicated `PruningHandler` allows us to track "Discarded Candidate" metrics independently. We can observe how many nodes are being filtered out for "quality reasons" versus "budget reasons," providing better insight into retrieval health.

## Consequences

- **Pipeline Position:** Typically positioned at the end of the ranking phase to filter based on the most accurate scores available.
- **Optionality:** Can be removed from the pipeline if absolute volume is preferred over quality-based filtering.
- **Configuration Overhead:** Adds a new `pruning` section to the YAML config, but provides significant control over precision.
- **Performance:** Negligible overhead as it performs simple mathematical filtering on a small set of candidates (typically < 20).
