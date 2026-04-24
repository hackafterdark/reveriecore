import sys
import os
import sqlite3
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add project root to sys.path

# Note: Removed global sys.modules['sqlite_vec'] mock which caused state pollution.

from reveriecore.database import DatabaseManager
from reveriecore.graph_query import GraphQueryService
from reveriecore.retrieval import Retriever

def setup_mock_db(db_path):
    if os.path.exists(db_path):
        os.remove(db_path)
    db = DatabaseManager(db_path)
    return db

def test_shared_entity_bridging():
    db_path = "test_bridge.db"
    db = setup_mock_db(db_path)
    graph = GraphQueryService(db)
    
    cursor = db.get_cursor()
    
    # 1. Create two memories
    cursor.execute("INSERT INTO memories (id, content_full) VALUES (101, 'Memory A mentions Entity X')")
    cursor.execute("INSERT INTO memories (id, content_full) VALUES (102, 'Memory B also mentions Entity X')")
    
    # 2. Create shared entity
    cursor.execute("INSERT INTO entities (id, name, label) VALUES (500, 'Entity X', 'TOOL')")
    
    # 3. Create bidirectional associations
    # Memory 101 -> Entity 500
    cursor.execute("INSERT INTO memory_relations (source_id, source_type, target_id, target_type, relation_type) VALUES (101, 'MEMORY', 500, 'ENTITY', 'MENTIONS')")
    # Memory 102 -> Entity 500
    cursor.execute("INSERT INTO memory_relations (source_id, source_type, target_id, target_type, relation_type) VALUES (102, 'MEMORY', 500, 'ENTITY', 'MENTIONS')")
    
    db.commit()
    cursor.execute("SELECT * FROM memory_relations")
    print(f"Associations in DB: {cursor.fetchall()}")
    
    # 4. Test Traversal from 101 to find 102

    results = graph.get_related_memories([101], depth=2)
    print(f"Bridge results from 101: {results}")
    assert 102 in results
    
    print("SUCCESS: Shared entity bridging verified.")
    db.close()
    if os.path.exists(db_path):
        os.remove(db_path)

def test_hub_protection():
    db_path = "test_hub.db"
    db = setup_mock_db(db_path)
    graph = GraphQueryService(db)
    
    cursor = db.get_cursor()
    
    # 1. Create seed memory
    cursor.execute("INSERT INTO memories (id, content_full) VALUES (1, 'Seed Memory')")
    # 2. Create hub entity
    cursor.execute("INSERT INTO entities (id, name, label) VALUES (100, 'The Hub', 'CONCEPT')")
    # 3. Link seed to hub
    cursor.execute("INSERT INTO memory_relations (source_id, source_type, target_id, target_type, relation_type) VALUES (1, 'MEMORY', 100, 'ENTITY', 'MENTIONS')")
    
    # 4. Create 20 other memories linked to the hub
    for i in range(2, 22):
        cursor.execute(f"INSERT INTO memories (id, content_full) VALUES ({i}, 'Leaf {i}')")
        # Bidirectional link: Leaf -> Hub
        cursor.execute(f"INSERT INTO memory_relations (source_id, source_type, target_id, target_type, relation_type, confidence_score) VALUES ({i}, 'MEMORY', 100, 'ENTITY', 'MENTIONS', {i/100.0})")
        
    db.commit()
    
    # 5. Run traversal with limit 11
    # Depth 2: Memory 1 -> Entity 100 (Hub) -> Memories 2..21 (Leaves)
    # The Hub (100) has 21 total associations (1 incoming from seed, 20 incoming from leaves).
    # Since Hub is Entity 100, we should get 10 leaf memories back (+ the seed itself which we skip).
    # So we request 11 neighbors from the Hub.
    results = graph.get_related_memories([1], depth=2, per_node_limit=11)
    print(f"Hub protection results count: {len(results)}")
    
    # The set of 10 should be the ones with highest confidence (21, 20, 19...)
    print(f"Hub results IDs: {sorted(results)}")
    assert len(results) == 10
    assert 21 in results
    assert 2 not in results # Leaf 2 has lowest confidence
    
    print("SUCCESS: Hub protection verified.")
    db.close()
    if os.path.exists(db_path):
        os.remove(db_path)

if __name__ == "__main__":
    test_shared_entity_bridging()
    test_hub_protection()
