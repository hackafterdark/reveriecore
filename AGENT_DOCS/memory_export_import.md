# Technical Specification: Memory-as-Code (Markdown Mirror)

## 1. Overview
To ensure data sovereignty, portability, and "brain" disaster recovery, `ReverieCore` implements a bi-directional synchronization system between the SQLite runtime and a local filesystem-based Markdown archive.

## 2. The "Mirror" Philosophy
- **Runtime State (SQLite):** Used for performance, high-speed vector retrieval, and `MesaService` operations.
*   **Source of Truth (Markdown + Frontmatter):** Used for archival, human-readability, and cross-platform portability. This allows for database-less graph traversal and complete reconstruction of the agent's memory.

## 3. Data Structure (The "Memory Passport")
### 3.1. Frontmatter Schema (v1.0)
```yaml
---
version: "1.0"
guid: "36790355-5589-444f-be5f-2f8a12e952b4"
path: "year=2026/month=04/day=24/36790355-5589-444f-be5f-2f8a12e952b4.md"
type: "RUNTIME_ERROR"
importance: 4.02081298828125
status: "ACTIVE"
owner: "default"
learned_at: "2026-04-24T02:28:10Z"
relations:
  - name: "sqlite-vec"
    label: "LIBRARY"
    type: "MENTIONS"
    node_type: "ENTITY"
    confidence: 1.0
    guid: "ce73d3b9-2c36-4840-9a25-7c865f5a296f"
    role: "target"
    description: "High-performance vector similarity search for SQLite"
  - type: "CHILD_OF"
    node_type: "MEMORY"
    confidence: 1.0
    guid: "1b547e1e-937d-496a-8eff-7c628817f9e1"
    role: "target"
metadata: {}
---
```

### 3.2. Directory Structure
Files are stored using Hive-style partitioning for scalability:
`~/.hermes/reverie_archive/year=YYYY/month=MM/day=DD/{guid}.md`

## 4. Implementation Details

### 4.1. Export Logic
- **Identity**: Uses globally unique `guid` for filenames and cross-reference instead of volatile integer IDs.
- **Self-Describing**: The `path` field in frontmatter allows external tools to know exactly where the file lives in the Hive hierarchy.
- **Entity Metadata**: Relations to entities include `name`, `label`, and `description` directly in the memory's frontmatter. This enables full graph restoration even if the `entities` table is lost.

### 4.2. Import & Disaster Recovery
- **Pass 1: Node Restoration**: Reconstructs `memories` records. Matches existing nodes via `guid`.
- **Pass 2: Graph Re-linking**: Re-establishes records in `memory_relations`. 
- **Auto-Restoration**: If an associated `ENTITY` is missing from the database, the system automatically recreates it using the metadata stored in the memory file.
- **Re-Embedding**: Triggers background re-vectorization for imported memories to populate the `memories_vec` table.

### 4.3. Tooling
- **Agent Tool**: `mirror_archive(action="export"|"import")` allows the agent to trigger bulk synchronization on demand.
- **Manual Control**: Exports can be triggered during the `MesaService` pruning cycle or via manual developer intervention.

## 5. Benefits
- **Database-less Traversal**: Using `grep` or Markdown editors (Obsidian, VS Code), you can follow the memory graph via GUIDs and paths without SQL.
- **Total Recovery**: A deleted `.db` file can be 100% reconstructed from the Markdown archive.
- **Human Readable**: Memory "fragments" and "anchors" are stored as clean, readable Markdown.
- **Vendor Agnostic**: You can move your agent memory to any system that supports Markdown.