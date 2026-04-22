# Implementation Plan: Memory-as-Code (Markdown Mirror)

Implement a bi-directional synchronization system between the ReverieCore SQLite database and a local Markdown archive, enabling data sovereignty, portability, and "brain" disaster recovery.

## User Review Required

> [!IMPORTANT]
> **Identity Stability (GUIDs)**: We will add a `guid` (UUID) field to the `memories` and `entities` tables. The integer `id` remains the internal primary key for performance joins, but the `guid` will be the "Business Key" used for external references, API calls, and Markdown filenames.

> [!TIP]
> **Write-Once Archive**: The Markdown archive is a permanent vault. Deleting a record in SQLite will NOT delete the file on disk. Instead, the file will be updated with `status: "PURGED"`.

> [!NOTE]
> **Versioning**: All exported Markdown files will include a `version: 1.0` field in the frontmatter to support future schema evolution.

## Proposed Changes

### [Database & Identity]

#### [MODIFY] [database.py](file:///home/tom/.hermes/plugins/reveriecore/database.py)
- Update `_create_schema` to include a `guid` (UUID) column in `memories` and `entities`.
- Update `_migrate_columns` to backfill `guid` for existing records using `uuid.uuid4()`.

### [Mirroring Logic]

#### [NEW] [mirror.py](file:///home/tom/.hermes/plugins/reveriecore/mirror.py)
- **`MirrorService`**:
    - `export_node(memory_id)`: Fetches a memory and its associated labels/links, serializes to YAML + Markdown, and writes to the Hive path.
    - `import_archive(path)`: Non-blocking ingestion. 
        1. **Fast Path**: Import text and metadata immediately to restore availability.
        2. **Lazy Path**: Identify records with missing embeddings and populate `memories_vec` in the background (or on-demand).
    - **Hive Pathing**: Implements `archive/year=YYYY/month=MM/day=DD/{guid}.md` logic.
    - **Association Management**: Re-links `CHILD_OF`, `SUPERSEDES`, and `MENTIONS` edges using `guid`-to-`id` resolution.

### [Lifecycle & Maintenance]

#### [MODIFY] [pruning.py](file:///home/tom/.hermes/plugins/reveriecore/pruning.py)
- Integrate `MirrorService.export_node` into the `MesaService` heartbeats.
- Ensure any memory transitioning to `ARCHIVED` or being "Synthesized" is immediately mirrored to disk.

#### [MODIFY] [provider.py](file:///home/tom/.hermes/plugins/reveriecore/provider.py)
- Add `export_memory` and `import_memory` as private utilities or tool-ready handlers.

## Open Questions

1. **GUID vs. ID**: Do you approve adding a `guid` column for cross-platform stability?
2. **Permanent Vault?**: Should deleting a memory in SQLite *ever* delete the Markdown file, or should the archive grow monotonically for safety?
3. **Re-Embedding Strategy**: On a bulk import, thousands of memories might need re-vectorization. Should this be a blocking operation or a background "indexing" task?

## Verification Plan

### Automated Tests
- Create a test script that:
    1. Injects a memory with associations.
    2. Exports it to disk (verifying Hive-style partitioning).
    3. Wipes the database.
    4. Imports from the Markdown mirror (Non-blocking check).
    5. Verifies that the memory, its importance score, and its graph associations are perfectly restored using GUID identity.

### Manual Verification
- Inspect the generated directory structure to confirm Hive-style paths.
- Open a generated `.md` file in a Markdown editor to verify readability.
