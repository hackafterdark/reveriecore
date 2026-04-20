import sqlite3
import os
import json
from datetime import datetime
import sys

# Ensure local imports work
sys.path.append(os.getcwd())

from database import DatabaseManager
from enrichment import EnrichmentService
from pruning import MemoryPruningService
from retrieval import Retriever
from schemas import MemoryType

def test_consolidation():
    # Setup isolated DB
    db_path = "test_consolidation.db"
    if os.path.exists(db_path): os.remove(db_path)
    
    db = DatabaseManager(db_path)
    enrichment = EnrichmentService({})
    pruning = MemoryPruningService(db, enrichment)
    retriever = Retriever(db, enrichment)
    
    cursor = db.get_cursor()
    
    # 1. Create 4 memories sharing the same entity "database.py"
    entity_name = "database.py"
    cursor.execute("INSERT INTO entities (name, label) VALUES (?, 'FILE')", (entity_name,))
    ent_id = cursor.lastrowid
    
    memories = [
        "database.py uses sqlite-vec for vector search.",
        "database.py has a table called memory_associations for graph data.",
        "database.py implements migration logic for schema updates.",
        "database.py is located in the reveriecore directory."
    ]
    
    mem_ids = []
    for m in memories:
        cursor.execute("INSERT INTO memories (content_full, status, learned_at) VALUES (?, 'ACTIVE', '2023-01-01T00:00:00Z')", (m,))
        mid = cursor.lastrowid
        mem_ids.append(mid)
        # Link to entity
        cursor.execute("INSERT INTO memory_associations (source_id, source_type, target_id, target_type, association_type) VALUES (?, 'MEMORY', ?, 'ENTITY', 'MENTIONS')", (mid, ent_id))

    db.commit()
    print(f"Created {len(mem_ids)} memories for {entity_name}")

    # 2. Run Consolidation (threshold 4)
    pruning.consolidate_by_entities(threshold=4)
    
    # 3. Verify
    cursor.execute("SELECT id, content_full, status FROM memories WHERE status = 'ACTIVE'")
    active = cursor.fetchall()
    print(f"Active memories after consolidation: {len(active)}")
    for r in active:
        print(f" - ID {r[0]}: {r[1][:50]}...")
    
    cursor.execute("SELECT id FROM memories WHERE status = 'ARCHIVED'")
    archived = cursor.fetchall()
    print(f"Archived memories: {len(archived)}")
    
    assert len(active) == 1, "Should have exactly 1 active summary"
    assert len(archived) == 4, "Should have archived the 4 originals"
    
    # 4. Check SUPERSEDES links
    cursor.execute("SELECT association_type FROM memory_associations WHERE association_type = 'SUPERSEDES'")
    links = cursor.fetchall()
    print(f"Supersedes links created: {len(links)}")
    assert len(links) == 4
    
    # 5. Check Retrieval
    # Create a query vector (dummy)
    results = retriever.search("database info", [0]*384)
    print(f"Search results count: {len(results)}")
    assert len(results) == 1
    
    os.remove(db_path)
    print("CONSOLIDATION TEST PASSED.")

if __name__ == "__main__":
    test_consolidation()
