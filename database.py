import sqlite3
import sqlite_vec
import logging
import json
import threading
from contextlib import contextmanager
from pathlib import Path

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
            
            # Load sqlite-vec extension
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            
            self._create_schema()
            logger.info(f"Database initialized at {self.db_path} with sqlite-vec support and 10s timeout.")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    @contextmanager
    def write_lock(self):
        """Context manager for atomicity and thread-safety during write operations."""
        with self._lock:
            try:
                yield self.conn.cursor()
                self.conn.commit()
            except Exception as e:
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
                expires_at DATETIME,
                status TEXT NOT NULL DEFAULT 'ACTIVE' -- 'ACTIVE' or 'ARCHIVED'
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
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
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
        if "token_count_full" not in columns:
            logger.info("Migrating database: Adding token_count_full column")
            cursor.execute("ALTER TABLE memories ADD COLUMN token_count_full INTEGER")
            
        if "token_count_abstract" not in columns:
            logger.info("Migrating database: Adding token_count_abstract column")
            cursor.execute("ALTER TABLE memories ADD COLUMN token_count_abstract INTEGER")

        if "author_id" not in columns:
            logger.info("Migrating database: Adding author_id column")
            cursor.execute("ALTER TABLE memories ADD COLUMN author_id TEXT NOT NULL DEFAULT 'USER'")
        
        if "owner_id" not in columns:
            logger.info("Migrating database: Adding owner_id column")
            cursor.execute("ALTER TABLE memories ADD COLUMN owner_id TEXT NOT NULL DEFAULT 'PERSONAL_WORKSPACE'")
            
        if "actor_id" not in columns:
            logger.info("Migrating database: Adding actor_id column")
            cursor.execute("ALTER TABLE memories ADD COLUMN actor_id TEXT NOT NULL DEFAULT 'HERMES_AGENT'")
            
        if "session_id" not in columns:
            logger.info("Migrating database: Adding session_id column")
            cursor.execute("ALTER TABLE memories ADD COLUMN session_id TEXT")
            
        if "workspace" not in columns:
            logger.info("Migrating database: Adding workspace column")
            cursor.execute("ALTER TABLE memories ADD COLUMN workspace TEXT")
            
        if "status" not in columns:
            logger.info("Migrating database: Adding status column")
            cursor.execute("ALTER TABLE memories ADD COLUMN status TEXT NOT NULL DEFAULT 'ACTIVE'")
        
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
        
        self.conn.commit()

    def purge_associations(self, memory_id: int):
        """Removes all triples derived from a specific memory ID (Idempotency Safeguard)."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM memory_associations WHERE evidence_memory_id = ?", (memory_id,))
        self.conn.commit()
        logger.debug(f"Purged associations for memory {memory_id}")

    def get_cursor(self):
        return self.conn.cursor()

    def commit(self):
        self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def __del__(self):
        self.close()
