import sqlite3
import sqlite_vec
import logging
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, List, Dict
import uuid

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Thread-safe Singleton for managing SQLite and vector virtual tables.
    Ensures global locking for write operations across all plugin components.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, db_path: str = "reveries.db"):
        # Initialize only once
        if hasattr(self, 'initialized'):
            return
            
        self.db_path = db_path
        self.conn = None
        self.initialized = False
        self._initialize_db()
        self.initialized = True

    def _initialize_db(self):
        """Connects, loads extensions, and ensures schema exists."""
        try:
            # Busy timeout of 10s to handle contention gracefully
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10)
            
            # 1. Enable WAL mode for better concurrency in AaaS/Background environments
            self.conn.execute("PRAGMA journal_mode=WAL")
            
            # 2. Load sqlite-vec extension
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            
            self._create_schema()
            logger.info(f"Database initialized at {self.db_path} (WAL Mode) with sqlite-vec support and 10s timeout.")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    @contextmanager
    def write_lock(self):
        """Context manager for atomicity and thread-safety during write operations."""
        with self._lock:
            if not self.conn:
                raise RuntimeError("Database connection is not initialized.")
            try:
                yield self.conn.cursor()
                self.conn.commit()
            except Exception as e:
                if self.conn:
                    self.conn.rollback()
                logger.error(f"Database write operation failed (rolled back): {e}")
                raise

    def _create_schema(self):
        """Creates the relational and virtual tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # 1. Main memories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_full TEXT NOT NULL,
                content_abstract TEXT,
                token_count_full INTEGER,
                token_count_abstract INTEGER,
                author_id TEXT NOT NULL DEFAULT 'USER',
                owner_id TEXT NOT NULL DEFAULT 'PERSONAL_WORKSPACE',
                actor_id TEXT NOT NULL DEFAULT 'HERMES_AGENT',
                session_id TEXT,
                workspace TEXT,
                memory_type TEXT NOT NULL DEFAULT 'CONVERSATION',
                importance_score REAL DEFAULT 1.0,
                privacy TEXT NOT NULL DEFAULT 'PRIVATE',
                metadata TEXT,
                learned_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_accessed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                expires_at DATETIME,
                status TEXT NOT NULL DEFAULT 'ACTIVE', -- 'ACTIVE' or 'ARCHIVED'
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                guid TEXT UNIQUE
            )
        """)
        
        # 2. Entities table (Categorical Graph Nodes)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,      -- Canonical Name (e.g., "provider.py")
                label TEXT NOT NULL,            -- Type (e.g., "FILE", "FUNCTION")
                description TEXT,
                metadata TEXT,                   -- JSON
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                guid TEXT UNIQUE
            )
        """)
        
        # Migration: Ensure new columns exist in case table was created with old schema
        self._migrate_columns(cursor)
        
        # 3. Vector virtual table (using vec0)
        # Note: rowid will link to memories.id
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
                rowid INTEGER PRIMARY KEY,
                embedding FLOAT[384]
            )
        """)
        
        # 4. Enhanced Associations table (The Graph Edges)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_associations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                source_type TEXT NOT NULL DEFAULT 'MEMORY', -- 'MEMORY' or 'ENTITY'
                target_id INTEGER NOT NULL,
                target_type TEXT NOT NULL DEFAULT 'MEMORY',
                association_type TEXT NOT NULL,
                confidence_score REAL DEFAULT 1.0,
                metadata TEXT,                               -- JSON
                evidence_memory_id INTEGER,                  -- The memory that asserted this link
                FOREIGN KEY (evidence_memory_id) REFERENCES memories(id) ON DELETE CASCADE
            )
        """)
        
        self.conn.commit()

    def _migrate_columns(self, cursor):
        """Adds missing columns for backward compatibility if needed."""
        cursor.execute("PRAGMA table_info(memories)")
        columns = [row[1] for row in cursor.fetchall()]
        
        # 1. Renames
        if "content" in columns and "content_full" not in columns:
            logger.info("Migrating database: Renaming 'content' to 'content_full'")
            cursor.execute("ALTER TABLE memories RENAME COLUMN content TO content_full")
            columns.append("content_full")
            
        if "semantic_profile" in columns and "content_abstract" not in columns:
            logger.info("Migrating database: Renaming 'semantic_profile' to 'content_abstract'")
            cursor.execute("ALTER TABLE memories RENAME COLUMN semantic_profile TO content_abstract")
            columns.append("content_abstract")

        # 2. Additions
        additions = [
            ("token_count_full", "INTEGER"),
            ("token_count_abstract", "INTEGER"),
            ("author_id", "TEXT NOT NULL DEFAULT 'USER'"),
            ("owner_id", "TEXT NOT NULL DEFAULT 'PERSONAL_WORKSPACE'"),
            ("actor_id", "TEXT NOT NULL DEFAULT 'HERMES_AGENT'"),
            ("session_id", "TEXT"),
            ("workspace", "TEXT"),
            ("status", "TEXT NOT NULL DEFAULT 'ACTIVE'"),
            ("importance_score", "REAL DEFAULT 1.0"),
            ("privacy", "TEXT NOT NULL DEFAULT 'PRIVATE'"),
            ("metadata", "TEXT"),
            ("learned_at", "DATETIME"),
            ("expires_at", "DATETIME"),
            ("updated_at", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
            ("content_abstract", "TEXT")
        ]
        
        for col_name, col_type in additions:
            if col_name not in columns:
                logger.info(f"Migrating database: Adding {col_name} column to memories")
                try:
                    # SQLite limitation: ALTER TABLE ADD COLUMN does not support dynamic (non-constant) defaults like CURRENT_TIMESTAMP.
                    # Constant defaults (like 'ACTIVE' or 1.0) are supported and required for NOT NULL columns.
                    if "CURRENT_TIMESTAMP" in col_type:
                        base_type = col_type.split("DEFAULT")[0].strip()
                        cursor.execute(f"ALTER TABLE memories ADD COLUMN {col_name} {base_type}")
                        cursor.execute(f"UPDATE memories SET {col_name} = CURRENT_TIMESTAMP WHERE {col_name} IS NULL")
                    else:
                        cursor.execute(f"ALTER TABLE memories ADD COLUMN {col_name} {col_type}")
                except sqlite3.OperationalError as e:
                    logger.error(f"Migration failed for {col_name}: {e}")

        if "last_accessed_at" not in columns:
            logger.info("Migrating database: Adding last_accessed_at column to memories")
            cursor.execute("ALTER TABLE memories ADD COLUMN last_accessed_at DATETIME")
            # For existing memories, seed last_accessed_at with learned_at if available, else current time
            # We check if learned_at exists in 'columns' OR if we just added it.
            # Since we just added it in the loop above if it was missing, it's safe to use now.
            cursor.execute("UPDATE memories SET last_accessed_at = COALESCE(learned_at, CURRENT_TIMESTAMP)")
        
        # 3. Enhanced Associations Migration
        cursor.execute("PRAGMA table_info(memory_associations)")
        assoc_cols = [row[1] for row in cursor.fetchall()]
        
        if "source_id" not in assoc_cols:
            # Table is old format (source_memory_id, target_memory_id)
            # Recreate it
            logger.info("Migrating database: Recreating memory_associations with polymorphic support")
            cursor.execute("DROP TABLE IF EXISTS memory_associations")
            cursor.execute("""
                CREATE TABLE memory_associations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id INTEGER NOT NULL,
                    source_type TEXT NOT NULL DEFAULT 'MEMORY',
                    target_id INTEGER NOT NULL,
                    target_type TEXT NOT NULL DEFAULT 'MEMORY',
                    association_type TEXT NOT NULL,
                    confidence_score REAL DEFAULT 1.0,
                    metadata TEXT,
                    evidence_memory_id INTEGER,
                    FOREIGN KEY (evidence_memory_id) REFERENCES memories(id) ON DELETE CASCADE
                )
            """)

        # 4. GUID Migration
        if "guid" not in columns:
            logger.info("Migrating database: Adding guid column to memories")
            cursor.execute("ALTER TABLE memories ADD COLUMN guid TEXT")
            # Backfill existing memories
            cursor.execute("SELECT id FROM memories WHERE guid IS NULL")
            for (mid,) in cursor.fetchall():
                cursor.execute("UPDATE memories SET guid = ? WHERE id = ?", (str(uuid.uuid4()), mid))
            
        cursor.execute("PRAGMA table_info(entities)")
        ent_columns = [row[1] for row in cursor.fetchall()]
        if "guid" not in ent_columns:
            logger.info("Migrating database: Adding guid column to entities")
            cursor.execute("ALTER TABLE entities ADD COLUMN guid TEXT")
            # Backfill existing entities
            cursor.execute("SELECT id FROM entities WHERE guid IS NULL")
            for (eid,) in cursor.fetchall():
                cursor.execute("UPDATE entities SET guid = ? WHERE id = ?", (str(uuid.uuid4()), eid))

        self.conn.commit()

    def purge_associations(self, memory_id: int):
        """Removes all triples derived from a specific memory ID (Idempotency Safeguard)."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM memory_associations WHERE evidence_memory_id = ?", (memory_id,))
        self.conn.commit()
        logger.debug(f"Purged associations for memory {memory_id}")

    def delete_memory(self, memory_id: int):
        """Atomically removes a memory and its vector index."""
        with self.write_lock() as cursor:
            # memory_associations has FOREIGN KEY (evidence_memory_id) REFERENCES memories(id) ON DELETE CASCADE
            # So associations will be cleaned up automatically.
            cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            cursor.execute("DELETE FROM memories_vec WHERE rowid = ?", (memory_id,))
        logger.info(f"Memory {memory_id} deleted successfully.")

    def update_access_timestamp(self, memory_ids: list[int]):
        """Updates the last_accessed_at timestamp for a batch of memories."""
        if not memory_ids:
            return
        try:
            with self.write_lock() as cursor:
                placeholders = ",".join(["?"] * len(memory_ids))
                cursor.execute(f"""
                    UPDATE memories SET last_accessed_at = CURRENT_TIMESTAMP 
                    WHERE id IN ({placeholders})
                """, tuple(memory_ids))
            logger.debug(f"Updated last_accessed_at for {len(memory_ids)} memories.")
        except Exception as e:
            logger.warning(f"Failed to update access timestamps: {e}")

    def update_memory(self, memory_id: int, content_full: str, content_abstract: str, 
                      embedding: list[float], token_count_full: int, token_count_abstract: int,
                      importance_score: float = None):
        """Updates an existing memory and its vector representation."""
        import sqlite_vec
        with self.write_lock() as cursor:
            if importance_score is not None:
                cursor.execute("""
                    UPDATE memories SET 
                        content_full = ?, content_abstract = ?, 
                        token_count_full = ?, token_count_abstract = ?,
                        importance_score = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (content_full, content_abstract, token_count_full, token_count_abstract, importance_score, memory_id))
            else:
                cursor.execute("""
                    UPDATE memories SET 
                        content_full = ?, content_abstract = ?, 
                        token_count_full = ?, token_count_abstract = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (content_full, content_abstract, token_count_full, token_count_abstract, memory_id))
                
            cursor.execute("""
                INSERT OR REPLACE INTO memories_vec (rowid, embedding) VALUES (?, ?)
            """, (memory_id, sqlite_vec.serialize_float32(embedding)))
        logger.info(f"Memory {memory_id} updated successfully.")

    def get_memory(self, memory_id: int) -> Optional[dict]:
        """Fetches a single memory record by ID."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, content_full, content_abstract, importance_score, owner_id, memory_type, guid, status, learned_at, metadata
            FROM memories WHERE id = ?
        """, (memory_id,))
        row = cursor.fetchone()
        if row:
            return {
                "id": row[0],
                "content_full": row[1],
                "content_abstract": row[2],
                "importance_score": row[3],
                "owner_id": row[4],
                "memory_type": row[5],
                "guid": row[6],
                "status": row[7],
                "learned_at": row[8],
                "metadata": row[9]
            }
        return None

    def get_memory_by_guid(self, guid: str) -> Optional[dict]:
        """Fetches a single memory record by GUID."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, content_full, content_abstract, importance_score, owner_id, memory_type, guid, status, learned_at, metadata
            FROM memories WHERE guid = ?
        """, (guid,))
        row = cursor.fetchone()
        if row:
            return {
                "id": row[0],
                "content_full": row[1],
                "content_abstract": row[2],
                "importance_score": row[3],
                "owner_id": row[4],
                "memory_type": row[5],
                "guid": row[6],
                "status": row[7],
                "learned_at": row[8],
                "metadata": row[9]
            }
        return None

    def get_entity(self, entity_id: int) -> Optional[dict]:
        """Fetches a single entity record by ID."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, name, label, description, metadata, guid
            FROM entities WHERE id = ?
        """, (entity_id,))
        row = cursor.fetchone()
        if row:
            return {
                "id": row[0],
                "name": row[1],
                "label": row[2],
                "description": row[3],
                "metadata": row[4],
                "guid": row[5]
            }
        return None

    def get_entity_by_guid(self, guid: str) -> Optional[dict]:
        """Fetches a single entity record by GUID."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, name, label, description, metadata, guid
            FROM entities WHERE guid = ?
        """, (guid,))
        row = cursor.fetchone()
        if row:
            return {
                "id": row[0],
                "name": row[1],
                "label": row[2],
                "description": row[3],
                "metadata": row[4],
                "guid": row[5]
            }
        return None

    def get_associations_for_node(self, node_id: int, node_type: str = 'MEMORY') -> List[Dict]:
        """Fetches all associations where the node is either source or target."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, source_id, source_type, target_id, target_type, association_type, confidence_score, metadata
            FROM memory_associations
            WHERE (source_id = ? AND source_type = ?) OR (target_id = ? AND target_type = ?)
        """, (node_id, node_type, node_id, node_type))
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "source_id": row[1],
                "source_type": row[2],
                "target_id": row[3],
                "target_type": row[4],
                "association_type": row[5],
                "confidence_score": row[6],
                "metadata": row[7]
            })
        return results

    def get_or_create_entity(self, name: str, label: str) -> int:
        """Finds or creates an entity by canonical name with stable GUID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM entities WHERE name = ?", (name,))
        row = cursor.fetchone()
        if row:
            return row[0]
        
        with self.write_lock() as cursor:
            # We generate a GUID for new entities to maintain cross-platform identity
            new_guid = str(uuid.uuid4())
            cursor.execute("INSERT INTO entities (name, label, guid) VALUES (?, ?, ?)", (name, label, new_guid))
            return cursor.lastrowid

    def get_cursor(self):
        return self.conn.cursor()

    def check_provenance_access(self, memory_id: int, owner_id: str) -> bool:
        """
        Verifies if a memory (usually a fragment) is accessible via a parent Observation.
        Strict multi-tenant validation.
        """
        try:
            cursor = self.conn.cursor()
            # Check for CHILD_OF association to an observation owned by owner_id
            query = """
                SELECT COUNT(*) FROM memory_associations ma
                JOIN memories m ON ma.target_id = m.id
                WHERE ma.source_id = ? 
                AND ma.association_type IN ('CHILD_OF', 'SUPERSEDES')
                AND m.owner_id = ?
            """
            cursor.execute(query, (memory_id, owner_id))
            count = cursor.fetchone()[0]
            return count > 0
        except Exception as e:
            logger.error(f"Provenance check failed: {e}")
            return False

    def commit(self):
        self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def __del__(self):
        self.close()
