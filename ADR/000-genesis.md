# ADR: Genesis - The Birth of Reverie Core

## Status
Accepted (Retroactive)

## Context
Reverie Core was born from a dissatisfaction with the "flat" nature of current agentic memory systems. Most agents rely on a simple text-dump to a markdown file or a basic vector database, which treats all information as having equal weight and fails to capture the nuanced, associative nature of human cognition. 

The original concept was driven by three primary realizations:
1. **The Human-Like Memory Gap**: Humans don't remember everything with equal clarity. I have implicit and explicit memory systems, and our recall is heavily influenced by emotion, sentiment, and the perceived "importance" of an event.
2. **The Graph Necessity**: While semantic (vector) similarity is powerful for finding "related" things, it lacks the deterministic rigor of relationships. To truly understand a world, an agent needs to know that *A* is a *part of* B, or that *C* *caused* D.
3. **Identity vs. Experience**: In narrative systems, characters have static identities (Cornerstones) that inform how they perceive and store new experiences (Reveries).

## Decision
I decided to build a memory system that prioritizes "cognitive realism" over simple data storage. This decision committed Reverie Core to several foundational pillars:

### 1. Hybrid Intelligence Layer
The system must include both a **Semantic Layer** (for fuzzy, intuitive recall via vector similarity) and a **Graph Layer** (for deterministic relationship traversal between entities).

### 2. Importance as a First-Class Citizen
Every memory must be ranked by "Value" or "Importance." High-sentiment or critical events are protected from decay, while incidental data is allowed to fade. This solves the "ignored critical info" problem where agents treat a user's life-safety warning with the same priority as a greeting.

### 3. Entity-Centric Architecture
The system will identify "Entities" and "Entity Relationships" as distinct from raw memory strings. This allows the agent to build a mental map of the world that persists even as specific event-memories are summarized or pruned.

### 4. Biological Decay (Pruning)
To maintain a high signal-to-noise ratio, the system must include "decay" and "cleanup" mechanisms. Like the human brain, the system should "prune" low-value connections and consolidate fragmented experiences into higher-level observations over time.

### 5. The "Cornerstone to Reverie" Relationship
Drawing from its origins in game/story engine research, the system adopts the terminology:
- **Cornerstones**: The constant, foundational personality and identity traits of an agent (conceptually aligned with the `SOUL.md` in the Hermes framework).
- **Reveries**: The dynamic, evolving memories formed through perception, heavily filtered through the lens of the agent's Cornerstones.

## Origins
The project originated from character research in a video game story system. The transition to the Hermes plugin ecosystem was a strategic choice to:
- Test the system against the unpredictable nature of real-world agent interactions.
- Provide a robust "long-term memory" utility for the Hermes framework.
- Use the learnings from the agentic space to refine the design for the original game engine.

## Consequences
### Positive
- **High Signal Recall**: Retrieval is biased toward information that actually matters.
- **Structural Integrity**: The agent can reason about dependencies and hierarchies, not just similarities.
- **Sustainable Scaling**: Automated pruning prevents the "context window bloat" that plagues long-running agents.

### Negative
- **Computational Overhead**: Enrichment (importance scoring, entity extraction) requires a sidecar intelligence layer.
- **Architectural Complexity**: Managing a synchronized vector-graph-SQL store is significantly harder than a flat file.
