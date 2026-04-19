import sqlite3
import sqlite_vec
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages the SQLite connection and vector virtual tables."""
    
    def __init__(self, db_path: str = "reveries.db"):
        self.db_path = db_path
        self.conn = None
        self._initialize_db()

    def _initialize_db(self):
        """Connects, loads extensions, and ensures schema exists."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            
            # Load sqlite-vec extension
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            
            self._create_schema()
            logger.info(f"Database initialized at {self.db_path} with sqlite-vec support.")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
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
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                expires_at DATETIME
            )
        """)
        
        # Migration: Ensure new columns exist in case table was created with old schema
        self._migrate_columns(cursor)
        
        # 2. Vector virtual table (using vec0)
        # Note: rowid will link to memories.id
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
                rowid INTEGER PRIMARY KEY,
                embedding FLOAT[384]
            )
        """)
        
        # 3. Associations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_associations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_memory_id INTEGER NOT NULL,
                target_memory_id INTEGER NOT NULL,
                association_type TEXT NOT NULL,
                context TEXT,
                FOREIGN KEY (source_memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                FOREIGN KEY (target_memory_id) REFERENCES memories(id) ON DELETE CASCADE
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
        
        # 3. Associations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_associations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_memory_id INTEGER NOT NULL,
                target_memory_id INTEGER NOT NULL,
                association_type TEXT NOT NULL,
                context TEXT,
                FOREIGN KEY (source_memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                FOREIGN KEY (target_memory_id) REFERENCES memories(id) ON DELETE CASCADE
            )
        """)
        
        self.conn.commit()

    def get_cursor(self):
        return self.conn.cursor()

    def commit(self):
        self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()
