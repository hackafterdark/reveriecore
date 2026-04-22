# Technical Specification: Memory-as-Code (Markdown Mirror)

## 1. Overview
To ensure data sovereignty, portability, and "brain" disaster recovery, `ReverieCore` will implement a bi-directional synchronization system between the SQLite runtime and a local filesystem-based Markdown archive.

## 2. The "Mirror" Philosophy
- **Runtime State (SQLite):** Used for performance, high-speed vector retrieval, and `MesaService` operations.
- **Source of Truth (Markdown + Frontmatter):** Used for archival, human-readability, and cross-platform portability.

## 3. Data Structure (The "Memory Passport")
Every memory node will be exported as an independent `.md` file with standardized YAML frontmatter.

```markdown
---
id: "mem_abc123"
learned_at: "YYYY-MM-DDTHH:MM:SSZ"
importance_score: 4.5
memory_type: "OBSERVATION"
status: "ACTIVE"
associations:
  - type: "CHILD_OF"
    target: "mem_xyz789"
  - type: "SUPERSEDES"
    target: "mem_def456"
---

# [Title/Summary]
[Raw text content of the memory]
```

## 3.4. Implementation Requirements
A. Export Utility (export_memory(memory_id))
- Reads the node from SQLite.
- Serializes Association table records into the YAML associations block.
- Generates a file named `{id}.md` in a partitioned directory structure.
- **Hive-Style Partitioning**: To ensure performance and scalability, files are stored in:
  `archive/year=YYYY/month=MM/day=DD/{id}.md`
B. Bulk Mirroring (MesaService Integration)
- The MesaService will include a mirror_to_disk() task.
- Trigger: Every time a node transitions from ACTIVE to ARCHIVED (or during initial creation), the mirror is updated.
- Performance: Use last_modified checks to avoid redundant disk I/O.

C. The "Re-Birth" Tool (sync_from_markdown(path))
- Parser: Uses PyYAML to ingest the frontmatter.
- Upsert Logic:
    1. Check if id exists in SQLite.
    2. If exists: UPDATE existing record.
    3. If missing: INSERT new record.
- Graph Reconstruction: Re-reads the associations block to re-establish edge connections in the association table.
- Re-Embedding: Triggers a background process to re-generate the vector index (sqlite-vec) for the imported text.

5. Benefits
- Vendor Agnostic: You can move your agent memory to any system that supports Markdown.
- Disaster Recovery: A corrupted .db file is no longer a terminal loss of your "Brain."
- Searchable: You can use standard OS search tools (grep, ripgrep, Obsidian, VS Code) to search your memory outside of the agent's constraints.

6. Edge Cases to Handle
- Vector Index Regeneration: Acknowledge that upon import, the vector store (sqlite-vec) will be empty and require a re-index cycle.
- Collision Handling: What happens if an id in Markdown differs from the one in the DB during a sync? (Strategy: Trust the Markdown/Frontmatter as the source of truth).