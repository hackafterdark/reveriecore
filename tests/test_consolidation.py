import sqlite3
import os
import json
from datetime import datetime
import sys

# Ensure local imports work

from reveriecore.database import DatabaseManager
from reveriecore.enrichment import EnrichmentService
from reveriecore.pruning import MesaService
from reveriecore.retrieval import Retriever
from reveriecore.schemas import MemoryType

def test_consolidation(tmp_path):
    # Setup isolated DB
    db_path = str(tmp_path / "test_consolidation.db")
    
    db = DatabaseManager(db_path)
    enrichment = EnrichmentService({})
    pruning = MesaService(db, enrichment)
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
        from datetime import datetime, timedelta
        stale_date = (datetime.now() - timedelta(days=30)).isoformat()
        cursor.execute("INSERT INTO memories (content_full, status, learned_at, last_accessed_at, importance_score) VALUES (?, 'ACTIVE', ?, ?, 1.0)", (m, stale_date, stale_date))
        mid = cursor.lastrowid
        mem_ids.append(mid)
        # Vector entry (Vector-First requirement)
        import sqlite_vec
        cursor.execute("INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)", (mid, sqlite_vec.serialize_float32([0.1]*384)))
        # Link to entity
        cursor.execute("INSERT INTO memory_associations (source_id, source_type, target_id, target_type, association_type) VALUES (?, 'MEMORY', ?, 'ENTITY', 'MENTIONS')", (mid, ent_id))

    db.commit()
    print(f"Created {len(mem_ids)} memories for {entity_name}")

    # 2. Run Consolidation (threshold 4)
    pruning.consolidation_threshold = 4
    pruning.run_hierarchical_consolidation()
    
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
    results = retriever.search([0]*384, "database info")
    print(f"Search results count: {len(results)}")
    assert len(results) == 1
    
    os.remove(db_path)
    print("CONSOLIDATION TEST PASSED.")

if __name__ == "__main__":
    test_consolidation()
