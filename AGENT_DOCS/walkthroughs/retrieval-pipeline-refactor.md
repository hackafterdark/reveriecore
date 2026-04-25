# Walkthrough - Retrieval Pipeline Refactor

The `Retriever` in `retrieval.py` has been refactored from a monolithic `search` method into a modular, composable pipeline framework. This architecture allows for better experimentation, easier maintenance, and external extensibility.

## Key Components

### 1. `RetrievalContext`
A central state container that carries query information, current candidates, metrics, and token budget throughout the pipeline stages.

### 2. Retrieval Handlers
The logic is now distributed across specialized handlers:
- **`AnchoringDiscovery`**: Performs semantic anchoring via entities.
- **`VectorDiscovery`**: Handles broad vector search with precision gating.
- **`GraphExpansionDiscovery`**: Implements iterative graph traversal for concept bridging.
- **`IntentRanker`**: Dynamically adjusts scoring weights based on query intent.
- **`ScoringRanker`**: Combines similarity, importance, and decay into a final relevance score.
- **`BudgetHandler`**: Manages the token budget and formats the final output (e.g., full content vs. abstracts).

### 3. Orchestration
The `Retriever.search` method now acts as an orchestrator, executing the registered pipelines:
```python
def search(self, ...):
    context = RetrievalContext(...)
    for handler in self.discovery_pipeline: handler.process(context, self)
    for handler in self.ranking_pipeline: handler.process(context, self)
    for handler in self.budget_pipeline: handler.process(context, self)
    return context.results
```

## Benefits
- **Extensibility**: Third-party plugins can now register custom handlers via `retriever.register_handler(my_handler, "ranking")`.
- **Observability**: Detailed metrics are collected in `context.metrics`, showing which stages contributed to the results.
- **Maintainability**: Logic for filtering, scoring, and formatting is isolated and easier to test or modify.

## Verification
- Verified that all core features (anchoring, vector fallback, graph expansion, intent weighting, and budgeting) are preserved in the new structure.
- Ensured that manual weight overrides and archived status filtering are correctly handled.
