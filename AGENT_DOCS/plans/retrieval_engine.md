# Documentation: ReverieCore Retrieval Engine (`retrieval.py`)

## Overview
The `Retriever` is the heart of the ReverieCore memory system. It transforms natural language queries into structured, ranked, and relevant context. It utilizes a **Hybrid-Retrieval** approach, combining vector similarity (intent), graph traversal (relational bridging), and dynamic importance weighting (relevance).

## Core Mechanisms

### 1. Intent-Aware Weighting
The retriever dynamically balances three signals based on query intent:
- **Vector Similarity (Semantic Match)**
- **Importance Score (Objective Value)**
- **Temporal Decay (Freshness)**

By analyzing the query for "fact-seeking" markers (e.g., "what is," "how to"), the system automatically prioritizes **Semantic Similarity**. In "exploratory" queries, it balances similarity with **Importance**, ensuring the agent sees authoritative context.

### 2. The Cascading Retrieval Pipeline
Retrieval executes in three distinct stages to optimize for both speed and recall:
1.  **Semantic Anchoring**: Performs a high-speed graph lookup using entities extracted from the query.
2.  **Precision Gated Vector Fallback**: Executes a vector similarity search (using `sqlite-vec`). A **Precision Gate** (similarity threshold) prunes noise before documents are ranked.
3.  **Graph Augmentation**: If vector results are weak, the engine performs iterative 1-hop and 2-hop graph expansions to bridge related concepts.

### 3. Adaptive Budgeting
To prevent context window overflow, the system dynamically chooses between injecting `content_full` or `content_abstract` based on the agent's available `token_budget` at the start of each turn.

---

# Architectural Implementation Plan: The Composable Pipeline

To evolve the `Retriever` into a framework that supports external plugins, we will refactor the `search` method into a **Chain of Responsibility** pattern.

### Phase 1: Define the `RetrievalContext` & Handler Interface
The `RetrievalContext` is a mutable object that travels through the pipeline, managing discovery state and the token budget.

```python
class RetrievalContext:
    def __init__(self, query: str, budget: int, config: Dict[str, Any] = None):
        self.query = query
        self.budget = budget
        self.consumed = 0
        self.candidates = {} # ID -> Candidate mapping
        self.metrics = {}    # Telemetry: {handler_name: data}
        self.config = config or {}

    @property
    def remaining(self):
        return self.budget - self.consumed

class RetrievalHandler:
    def process(self, context: RetrievalContext) -> None:
        """Handlers modify the context's candidates, consumed count, or metrics."""
        raise NotImplementedError
```

### Phase 2: Decouple the `search` Method
Refactor the monolithic `search` method into an **Orchestrator** that manages a registry of handlers:

1.  **Discovery Handlers**: Create separate handlers for `GraphDiscovery`, `VectorDiscovery`, and `KeywordDiscovery`.
2.  **Ranking Handlers**: Create separate handlers for `ImportanceScorer`, `DecayCalculator`, and `CrossEncoderReRanker`.
3.  **Pipeline Runner**: The `search` method will now simply:
    - Initialize the `RetrievalContext` with the query and token budget.
    - Run all **Discovery Handlers** (Parallel) to populate `context.candidates`.
    - Run all **Ranking & Budget Handlers** (Serial) to score, sort, and select content until `context.remaining` is exhausted.

### Phase 3: Configuration & Composability
Move the pipeline definition to an external JSON configuration. This allows you to reorder the search process without modifying the core code:

```json
{
  "retrieval_pipeline": {
    "discovery": ["vector_search", "graph_traversal"],
    "rankers": ["importance_boost", "temporal_decay", "cross_encoder"]
  }
}
```

### Implementation Checklist
- [ ] **Define Context**: Implement the `RetrievalContext` class.
- [ ] **Define Protocol**: Create the `RetrievalHandler` abstract base class.
- [ ] **Extract Handlers**: Move `graph` logic and `vector` logic into their own classes implementing the protocol.
- [ ] **Implement Orchestrator**: Replace the `search` logic with a loop that executes registered handlers against the context.
- [ ] **Precision Gating**: Ensure the `similarity >= 0.45` logic is encapsulated within the `VectorHandler`.
- [ ] **Registry Pattern**: Add a `register_handler(name, handler)` method to allow dynamic expansion of the pipeline.


### The "Composition" mindset
This solves for **user customization.** If a user wants to prioritize "Location-Based Retrieval" for their specific files, they simply create a `LocationHandler` and register it to the pipeline. They don't have to touch the Graph-RAG or Importance logic, and they don't have to submit a PR to `main`. Reverie Core is a **framework**.