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

def test_resurfacing():
    # Setup isolated DB
    db_path = "test_resurfacing.db"
    if os.path.exists(db_path): os.remove(db_path)
    
    db = DatabaseManager(db_path)
    enrichment = EnrichmentService({})
    retriever = Retriever(db, enrichment)
    
    cursor = db.get_cursor()
    
    # 1. Create an ARCHIVED memory
    cursor.execute("INSERT INTO memories (content_full, status, importance_score) VALUES (?, 'ARCHIVED', 3.0)", 
                   ("Specific detail about a hidden secret.",))
    archived_id = cursor.lastrowid
    
    # Also need a vector entry for search to find it via vector fallback
    import sqlite_vec
    vec = [0.0] * 384
    cursor.execute("INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)", 
                 (archived_id, sqlite_vec.serialize_float32(vec)))
    db.commit()
    print(f"Created archived memory {archived_id} with vector")

    # 2. Test Standard Search (Should NOT find it)
    results = retriever.search([0]*384, "hidden secret", include_archived=False)
    print(f"Standard Search Results: {len(results)}")
    assert len(results) == 0

    # 3. Test Deep Search (Should find it)
    results_deep = retriever.search([0]*384, "hidden secret", include_archived=True)
    print(f"Deep Search Results: {len(results_deep)}")
    assert len(results_deep) == 1
    assert results_deep[0]['id'] == archived_id

    # 4. Test Importance-Based Expiration Signal
    # Mock some technical text
    text = "The server will shut down in 5 minutes for maintenance."
    imp_data = enrichment.calculate_importance(text)
    print(f"Importance Data for transient text: {imp_data}")
    # Since it's mock, result depends on BART or lack thereof. 
    # But signature is verified.

    os.remove(db_path)
    print("RESURFACING TEST PASSED.")

if __name__ == "__main__":
    test_resurfacing()
