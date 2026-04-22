import sqlite3
import pytest
import uuid
from pathlib import Path
import sys

# Ensure we can import reveriecore modules
# Note: In the test environment, we might need to adjust sys.path if not running from root
from reveriecore.database import DatabaseManager

@pytest.fixture(autouse=True)
def reset_singleton():
    """Ensures each test gets a fresh DatabaseManager instance."""
    DatabaseManager._instance = None
    yield
    DatabaseManager._instance = None

def test_guid_migration_from_legacy(tmp_path):
    """Verifies that an old DB schema is correctly migrated and backfilled with GUIDs."""
    db_path = tmp_path / "legacy.db"
    
    # 1. Setup a "Legacy" Schema (minimalist)
    conn = sqlite3.connect(db_path)
    # Old memories used 'content' instead of 'content_full'
    conn.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, content TEXT)")
    conn.execute("INSERT INTO memories (content) VALUES (?)", ("Legacy memory content 1",))
    conn.execute("INSERT INTO memories (content) VALUES (?)", ("Legacy memory content 2",))
    
    conn.execute("CREATE TABLE entities (id INTEGER PRIMARY KEY, name TEXT UNIQUE)")
    conn.execute("INSERT INTO entities (name) VALUES (?)", ("Legacy Entity",))
    conn.commit()
    conn.close()
    
    # 2. Initialize DatabaseManager
    # This should trigger _initialize_db -> _create_schema -> _migrate_columns
    db = DatabaseManager(str(db_path))
    
    # 3. Verify Memories Migration
    cursor = db.get_cursor()
    cursor.execute("PRAGMA table_info(memories)")
    cols = {row[1]: row for row in cursor.fetchall()}
    
    assert "guid" in cols, "GUID column was not added to memories"
    assert "content_full" in cols, "content was not renamed to content_full"
    assert "status" in cols, "status column was not added"
    
    # Verify data and GUIDs
    cursor.execute("SELECT guid, content_full FROM memories")
    rows = cursor.fetchall()
    assert len(rows) == 2
    for guid, content in rows:
        assert guid is not None, "GUID was not backfilled"
        # Check if it's a valid UUID
        uuid.UUID(guid)
        assert "Legacy memory content" in content

    # 4. Verify Entities Migration
    cursor.execute("PRAGMA table_info(entities)")
    ent_cols = {row[1]: row for row in cursor.fetchall()}
    assert "guid" in ent_cols, "GUID column was not added to entities"
    
    cursor.execute("SELECT guid, name FROM entities")
    ent_row = cursor.fetchone()
    assert ent_row is not None
    assert ent_row[1] == "Legacy Entity"
    uuid.UUID(ent_row[0])
    
    db.close()

def test_guid_migration_idempotency(tmp_path):
    """Ensures that re-running initialization doesn't re-generate GUIDs."""
    db_path = str(tmp_path / "idempotent.db")
    db = DatabaseManager(db_path)
    
    # Add a memory
    with db.write_lock() as cursor:
        cursor.execute("INSERT INTO memories (content_full, guid) VALUES (?, ?)", 
                       ("Test memory", str(uuid.uuid4())))
        
    db_memory = db.get_memory(1)
    original_guid = db_memory['guid']
    assert original_guid is not None
    db.close()
    
    # Reset Singleton for test purposes to force re-initialization
    DatabaseManager._instance = None
    
    # Re-initialize
    db2 = DatabaseManager(db_path)
    db_memory2 = db2.get_memory(1)
    assert db_memory2['guid'] == original_guid, "GUID changed on re-initialization"
    db2.close()
