import pytest
import sqlite3

def test_database_initialization(db_manager):
    """Verify that all required tables are created."""
    cursor = db_manager.get_cursor()
    
    # Check for standard tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    assert "memories" in tables
    assert "memory_relations" in tables

def test_vector_extension_loaded(db_manager):
    """Verify that sqlite-vec extension is active."""
    cursor = db_manager.get_cursor()
    
    # Check for the virtual table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    assert "memories_vec" in tables
    
    # Try a simple vector query (even if empty) to ensure vec0 works
    try:
        # Zero vector of size 384
        zero_vec = [0.0] * 384
        cursor.execute("SELECT rowid FROM memories_vec WHERE embedding MATCH ? AND k = 1", (str(zero_vec),))
        cursor.fetchall()
    except sqlite3.OperationalError as e:
        pytest.fail(f"sqlite-vec not properly loaded: {e}")
