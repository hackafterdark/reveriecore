# ADR 006: Evolution of Reverie Core into a Composable Pipeline Framework

## Status
Accepted (2026-04-25)

## Context
Reverie Core has matured from a basic memory retrieval plugin into a complex system combining vector similarity, graph-based discovery, and multi-stage importance scoring. As we refined our performance metrics (specifically `context_precision` via RAGAS), we discovered that the optimal balance of these signals varies significantly by query intent and user domain. 

The previous "monolithic" implementation of the `Retriever` and `EnrichmentService` made it difficult to:
1.  Experiment with new ranking strategies (e.g., Cross-Encoders or Project-Specific weighting).
2.  Allow end-users to customize the retrieval logic without modifying the core codebase.
3.  Debug "noisy" retrieval results where one signal (e.g., Importance) drowned out others (e.g., Semantic Relevance).

## Decision
We will evolve Reverie Core into a **Composable Pipeline Framework**. This architecture decouples the core memory storage from the logic used to enrich and retrieve it.

### 1. Unified Pipeline Pattern
Both ingestion (Enrichment) and retrieval will operate as a **Chain of Responsibility**. 

- **Enrichment Pipeline**: Already transitioned to a handler-based model (`HeuristicHandler`, `BARTHandler`, `SoulHandler`) where each handler provides a score and a confidence level.
- **Retrieval Pipeline**: The monolithic `search` method will be refactored into an **Orchestrator** that manages two distinct phases of handlers.

### 2. Retrieval Phase Separation
Retrieval handlers will be categorized into two functional groups:

- **Discovery Handlers**: Responsible for finding candidates (High Recall).
  - *Examples*: `VectorDiscovery`, `GraphDiscovery`, `EntityAnchoring`.
  - *Behavior*: Designed for **parallel execution**. Decoupling discovery into independent objects allows the orchestrator to fire all semantic and relational lookups concurrently, significantly reducing the "Time to First Token" for complex RAG tasks.
- **Ranking Handlers**: Responsible for scoring and pruning candidates (High Precision).
  - *Examples*: `ImportanceRanker`, `TemporalDecay`, `CrossEncoderReRanker`.
  - *Behavior*: Executes serially to refine and sort the final result set.

### 4. The `RetrievalContext` Object
To manage state and prevent context window overflow, we utilize a mutable `RetrievalContext` object that evolves as it travels through the pipeline. This puts the retrieval engine in direct control of the output volume, rather than relying on external systems to truncate or compress information.

The object contains:
- **`query`**: The original natural language query and its vector embedding.
- **`token_budget`**: The maximum allowed tokens for the entire retrieval result.
- **`consumed_tokens`**: A running tally of tokens used by selected candidates (calculated by `BudgetHandlers`).
- **`candidate_pool`**: A dictionary of discovered memories, keyed by ID, being refined by the pipeline.
- **`metrics`**: A running telemetry log of what each handler did (e.g., `{"vector_discovery": {"found": 12, "latency_ms": 45}}`).
- **`config`**: A reference to the current pipeline settings (thresholds, weights, etc.).

By exposing `remaining_budget` as a property, downstream handlers can make intelligent decisions—such as opting for a summarized `content_abstract` instead of `content_full` when the budget is tight.

## Consequences

### Positive
- **Experimental Agility**: We can tune `context_precision` by simply re-ordering the pipeline or adding a new `PrecisionGateHandler`.
- **Extensibility**: Third-party developers can create custom handlers (e.g., "Company-Specific Terminology Handler") and register them to the framework.
- **Centralized Observability**: The `RetrievalContext` metrics allow the orchestrator to provide a complete trace of the retrieval process, showing exactly which handler contributed which nodes and how much latency was introduced.
- **Reduced Technical Debt**: Large, conditional blocks of logic are replaced by small, focused classes.

- **Architectural Complexity**: The jump from a single function to a class-based orchestrator introduces more "moving parts" for new contributors to learn.
- **Strict Protocol Adherence**: Handlers must carefully adhere to the `RetrievalContext` update patterns to avoid accidental data loss in the candidate pool.
- **Performance Overhead**: The abstraction layers introduce a negligible amount of execution overhead compared to the latency of the underlying vector or graph queries.
