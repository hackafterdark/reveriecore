# Custom Memory Plugin Database Schema (SQLite + sqlite-vec)

This schema supports a Retrieval-Augmented Generation (RAG) system using **SQLite** as the storage engine and **`sqlite-vec`** for high-performance vector similarity search.

## Architectural Goals
1.  **Vector Search:** Use the `vec0` virtual table for fast similarity search on 384-dimension embeddings.
2.  **Relational Source of Truth:** Core memory metadata and content are stored in standard SQLite tables.
3.  **Graph Relationships:** Map connections between memories (causality, support, chronology).
4. Desktop Friendly: Zero-config persistence using a single .db file.

## Retrieval Interface Contract (Golang)

The primary boundary for memory interaction is defined by the following Go interface, ensuring pluggable memory providers:

```go
type MemoryRetriever interface {
    Search(ctx context.Context, query string) ([]Memory, error)
}


---

## 1. `memories` Table (Source of Truth)

This table stores the rich, structured content of every memory.

```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guid TEXT UNIQUE,                -- Globally Unique ID for archival sync
    
    -- CORE CONTENT
    content_full TEXT NOT NULL,      -- Full raw text or transcript
    content_abstract TEXT,          -- Summarized "gist" (Generated via LLM pipeline)
    
    -- CONTEXT MANAGEMENT
    token_count_full INTEGER,       -- Token count for content_full (Proxy: BART)
    token_count_abstract INTEGER,   -- Token count for content_abstract
    
    -- IDENTITY & AUDIT (Provenance vs Namespace)
    author_id TEXT NOT NULL DEFAULT 'USER',          -- The Human (user_id)
    owner_id TEXT NOT NULL DEFAULT 'PERSONAL_WORKSPACE', -- The Profile (Namespace)
    actor_id TEXT NOT NULL DEFAULT 'REVERIE_SYNC_SERVICE', -- The specific process/agent
    session_id TEXT,                                 -- Current Session UUID
    workspace TEXT,                                  -- Local working directory
    
    -- TYPE & CLASSIFICATION
    memory_type TEXT NOT NULL DEFAULT 'CONVERSATION', -- Categorized via BART Zero-Shot
    
    -- METADATA & SCORES
    importance_score REAL DEFAULT 1.0, -- Calculated via BART Entailment (1.0-5.0)
    privacy TEXT NOT NULL DEFAULT 'PRIVATE',
    metadata TEXT,                  -- JSON string for flexible attributes
    
    -- LIFECYCLE
    learned_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'ACTIVE',    -- ACTIVE, ARCHIVED, PURGED
    expires_at DATETIME
);
```

## 2. `memories_vec` Table (Vector Search)

This is a **Virtual Table** managed by `sqlite-vec`. It acts as a specialized index for the embeddings.

```sql
-- Virtual table for 384-dimension embeddings (all-MiniLM-L6-v2)
CREATE VIRTUAL TABLE memories_vec USING vec0(
    rowid INTEGER PRIMARY KEY, -- Links directly to memories.id
    embedding FLOAT[384]      -- Vector representation
);
```

**Usage Example:**
```sql
-- Finding similar memories
SELECT 
    m.content_full,
    m.importance_score,
    v.distance
FROM memories m
JOIN memories_vec v ON m.id = v.rowid
WHERE v.embedding MATCH ?             -- The query embedding vector
  AND (m.owner_id = ? OR m.privacy = 'PUBLIC') -- Namespace Scoping
  AND v.k = 5                         -- Top 5 results
ORDER BY v.distance;
```

---

## 3. `entities` Table (Canonical Knowledge)

This table stores canonical entities extracted from memories (files, classes, concepts, tools).

```sql
CREATE TABLE entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guid TEXT UNIQUE,                -- Globally Unique ID for archival sync
    name TEXT NOT NULL UNIQUE,       -- Canonical name (e.g. 'database.py')
    label TEXT NOT NULL,             -- Type (e.g. 'FILE', 'CLASS', 'TOOL')
    description TEXT,                -- Unified summary from all mentions
    first_seen_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

---

## 4. `memory_associations` Table (Polymorphic Graph)

Maps the links between memories and entities to create a multi-layer knowledge graph for Reverie Core.

```sql
CREATE TABLE memory_associations (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- SOURCE NODE
    source_id INTEGER NOT NULL,
    source_type TEXT NOT NULL,       -- 'MEMORY' or 'ENTITY'
    
    -- TARGET NODE
    target_id INTEGER NOT NULL,
    target_type TEXT NOT NULL,       -- 'MEMORY' or 'ENTITY'
    
    -- EDGE PROPERTIES
    association_type TEXT NOT NULL,  -- e.g., 'PART_OF', 'DEPENDS_ON', 'MENTIONS'
    evidence_memory_id INTEGER,      -- Links to the memory that proved this link
    confidence REAL DEFAULT 1.0,     -- ML-derived extraction confidence
    
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

**Graph Patterns Supported:**
- **`MEMORY -> ENTITY`**: Memory A mentions Entity X.
- **`ENTITY -> ENTITY`**: Entity X is part of Entity Y (Knowledge Base).
- **`ENTITY -> MEMORY`**: Entity X was discussed in Memory B.
- **`MEMORY -> MEMORY`**: Memory A supports or refutes Memory B.