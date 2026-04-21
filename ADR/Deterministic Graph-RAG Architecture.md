# ADR: Deterministic Graph-RAG Architecture

## Status
Accepted (2026-04-19)

## Context
ReverieCore previously relied on a "flat" vector-search architecture. This was efficient but inherently probabilistic, leading to ambiguity in retrieval and "hallucination" of relationships between technical concepts. To achieve deterministic reasoning, we are transitioning to a **Graph-RAG** architecture using the **Triple Pattern** and **Entity Linking**.

## Decision
We will implement an structured Knowledge Graph within our existing SQLite database using standard SQL patterns.

### 1. The Triple Pattern
All structured knowledge is expressed as atomic relations:
`(Subject) -> [Predicate] -> (Object)`

- **Nodes**: Entities (Files, Functions, Repositories) and Event-Memories.
- **Edges**: Predicates defining the relationship between nodes.

### 2. Taxonomy (Fixed Predicates)
To ensure deterministic traversal, the system is restricted to the following association predicates:
- `FIXES`: Memory identifies a solution to a problem.
- `CAUSES`: Root-cause relationship (e.g., Error -> File).
- `DEPENDS_ON`: Structural dependency.
- `PART_OF`: Hierarchical containment.
- `IS_A`: Classification (e.g., File -> Entity).
- `MENTIONS`: General association.

### 3. Implementation Patterns
- **Entity Extraction**: Performed asynchronously by an `internal_llm_client` sidecar service, gated by `importance_score >= 3.0` to minimize latency.
- **Storage**: Entities are stored in a dedicated `entities` table. Relationships are stored in a `memory_associations` edge-list table.
- **Traversal**: We utilize **Recursive Common Table Expressions (CTEs)** for graph traversal. This avoids external library dependencies and keeps the database portable.

## Consequences
### Positive
- **Deterministic Traversal**: Enables precise root-cause analysis (e.g., finding the specific function that caused an error) without vector "fuzziness."
- **Portability**: By avoiding graph-specific extensions (`sqlite-graph`), the database remains a standard, interoperable SQLite file.
- **Maintainability**: The separation of `EnrichmentService` (sidecar) from the Agent's main loop prevents deadlocks and state pollution.

### Negative
- **Complexity**: Adds a background extraction pipeline and requires a two-pass resolution logic (Resolve Entities -> Insert Associations).
- **Maintenance**: Schema changes require careful migration, though the "Triple" format is stable and unlikely to require future breaking changes.

## Graph Traversal Template
All retrieval logic requiring dependency lookup must utilize the standard recursive pattern:

```sql
WITH RECURSIVE graph_traversal(id, level) AS (
    SELECT [start_id], 0
    UNION ALL
    SELECT ma.target_id, gt.level + 1
    FROM memory_associations ma
    JOIN graph_traversal gt ON ma.source_id = gt.id
    WHERE gt.level < 3
)
SELECT * FROM memories WHERE id IN (SELECT id FROM graph_traversal);