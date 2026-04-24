import sqlite3
import sqlite_vec
import logging
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any
import uuid

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manages SQLite and vector virtual tables.
    Each instance represents a single database connection.
    """
    def __init__(self, db_path: str = "reveries.db"):
        self.db_path = db_path
        self.conn = None
        self._lock = threading.Lock()
        self._initialize_db()

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
        
        # 1. Memories Table (Rich metadata)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guid TEXT UNIQUE,
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
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_accessed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT NOT NULL DEFAULT 'ACTIVE',
                expires_at DATETIME
            )
        """)
        
        # 2. Entities Table (Canonical technical knowledge)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guid TEXT UNIQUE,
                name TEXT UNIQUE NOT NULL,
                label TEXT NOT NULL DEFAULT 'ENTITY',
                description TEXT,
                metadata TEXT,
                first_seen_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 3. Vector virtual table (using vec0)
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
                rowid INTEGER PRIMARY KEY,
                embedding FLOAT[384]
            )
        """)
        
        # 4. Relations table (The Graph Edges)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                source_type TEXT NOT NULL DEFAULT 'MEMORY',
                target_id INTEGER NOT NULL,
                target_type TEXT NOT NULL DEFAULT 'MEMORY',
                relation_type TEXT NOT NULL,
                confidence_score REAL DEFAULT 1.0,
                metadata TEXT,
                evidence_memory_id INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (evidence_memory_id) REFERENCES memories(id) ON DELETE CASCADE
            )
        """)
        
        self.conn.commit()

    def purge_relations(self, memory_id: int):
        """Removes all triples derived from a specific memory ID (Idempotency Safeguard)."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM memory_relations WHERE evidence_memory_id = ?", (memory_id,))
        self.conn.commit()
        logger.debug(f"Purged relations for memory {memory_id}")

    def delete_memory(self, memory_id: int):
        """Atomically removes a memory and its vector index."""
        with self.write_lock() as cursor:
            # memory_relations has FOREIGN KEY (evidence_memory_id) REFERENCES memories(id) ON DELETE CASCADE
            # So relations will be cleaned up automatically.
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
                      importance_score: float = None, metadata: dict = None):
        """Updates an existing memory and its vector representation."""
        import sqlite_vec
        with self.write_lock() as cursor:
            # Prepare update fields
            updates = {
                "content_full": content_full,
                "content_abstract": content_abstract,
                "token_count_full": token_count_full,
                "token_count_abstract": token_count_abstract
            }
            if importance_score is not None:
                updates["importance_score"] = importance_score
            if metadata is not None:
                updates["metadata"] = json.dumps(metadata)
                
            set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
            set_clause += ", updated_at = CURRENT_TIMESTAMP"
                
            params = [updates[k] for k in updates.keys()]
            params.append(memory_id)
            
            cursor.execute(f"UPDATE memories SET {set_clause} WHERE id = ?", tuple(params))
            
            # Check if the memory actually existed
            if cursor.rowcount == 0:
                raise ValueError(f"Memory with ID {memory_id} not found.")

            # sqlite-vec virtual tables do not support INSERT OR REPLACE.
            # Must DELETE first, then INSERT for existing rows.
            cursor.execute("DELETE FROM memories_vec WHERE rowid = ?", (memory_id,))
            cursor.execute(
                "INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)",
                (memory_id, sqlite_vec.serialize_float32(embedding))
            )
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
            meta = row[9]
            if meta and isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except:
                    pass
                    
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
                "metadata": meta
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
            meta = row[9]
            if meta and isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except:
                    pass
                    
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
                "metadata": meta
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

    def get_relations_for_node(self, node_id: int, node_type: str = 'MEMORY') -> List[Dict]:
        """Fetches all relations where the node is either source or target."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, source_id, source_type, target_id, target_type, relation_type, confidence_score, metadata
            FROM memory_relations
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
                "relation_type": row[5],
                "confidence_score": row[6],
                "metadata": row[7]
            })
        return results

    def get_relations_by_evidence(self, memory_id: int) -> List[Dict]:
        """Fetches all relations where this memory provided the evidence (triples)."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, source_id, source_type, target_id, target_type, relation_type, confidence_score, metadata
            FROM memory_relations
            WHERE evidence_memory_id = ?
        """, (memory_id,))
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "source_id": row[1],
                "source_type": row[2],
                "target_id": row[3],
                "target_type": row[4],
                "relation_type": row[5],
                "confidence_score": row[6],
                "metadata": row[7]
            })
        return results

    def get_or_create_entity(self, name: str, label: str, description: str = None) -> int:
        """Finds or creates an entity by canonical name with stable GUID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM entities WHERE name = ?", (name,))
        row = cursor.fetchone()
        if row:
            return row[0]
        
        with self.write_lock() as cursor:
            # We generate a GUID for new entities to maintain cross-platform identity
            new_guid = str(uuid.uuid4())
            cursor.execute("INSERT INTO entities (name, label, guid, description) VALUES (?, ?, ?, ?)", (name, label, new_guid, description))
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
            # Check for CHILD_OF relation to an observation owned by owner_id
            query = """
                SELECT COUNT(*) FROM memory_relations ma
                JOIN memories m ON ma.target_id = m.id
                WHERE ma.source_id = ? 
                AND ma.relation_type IN ('CHILD_OF', 'SUPERSEDES')
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
