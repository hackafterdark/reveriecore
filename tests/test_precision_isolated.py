import sqlite3
import re
from typing import List, Dict, Any

# Isolated Class for Testing (Python-Driven Traversal)
class GraphQueryService:
    def __init__(self, db_conn):
        self.db_conn = db_conn

    def get_related_memories(self, start_memory_ids: List[int], depth: int = 2, per_node_limit: int = 10) -> List[int]:
        cursor = self.db_conn.cursor()
        visited = set()
        for mid in start_memory_ids:
            visited.add((mid, 'MEMORY'))
        current_layer = [(mid, 'MEMORY') for mid in start_memory_ids]
        found_memories = set()
        
        for level in range(depth):
            if not current_layer: break
            next_layer = []
            for node_id, node_type in current_layer:
                query = """
                    SELECT 
                        CASE WHEN source_id = ? AND source_type = ? THEN target_id ELSE source_id END as next_id,
                        CASE WHEN source_id = ? AND source_type = ? THEN target_type ELSE source_type END as next_type
                    FROM memory_relations
                    WHERE (source_id = ? AND source_type = ?) OR (target_id = ? AND target_type = ?)
                    ORDER BY confidence DESC, id ASC
                    LIMIT ?
                """
                cursor.execute(query, (node_id, node_type, node_id, node_type, node_id, node_type, node_id, node_type, per_node_limit))
                for next_id, next_type in cursor.fetchall():
                    if (next_id, next_type) not in visited:
                        visited.add((next_id, next_type))
                        next_layer.append((next_id, next_type))
                        if next_type == 'MEMORY': found_memories.add(next_id)
            current_layer = next_layer
            if len(found_memories) >= 50: break
        return list(found_memories)

def run_test():
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, content_full TEXT)")
    cursor.execute("CREATE TABLE entities (id INTEGER PRIMARY KEY, name TEXT, label TEXT)")
    cursor.execute("CREATE TABLE memory_relations (id INTEGER PRIMARY KEY AUTOINCREMENT, source_id INTEGER, source_type TEXT, target_id INTEGER, target_type TEXT, relation_type TEXT, confidence REAL)")
    
    graph = GraphQueryService(conn)
    
    # 1. Bridging Test
    cursor.execute("INSERT INTO memories (id, content_full) VALUES (101, 'M1')")
    cursor.execute("INSERT INTO memories (id, content_full) VALUES (102, 'M2')")
    cursor.execute("INSERT INTO entities (id, name, label) VALUES (500, 'Shared', 'X')")
    cursor.execute("INSERT INTO memory_relations VALUES (101, 'MEMORY', 500, 'ENTITY', 'M', 1.0)")
    cursor.execute("INSERT INTO memory_relations VALUES (102, 'MEMORY', 500, 'ENTITY', 'M', 1.0)")
    
    results = graph.get_related_memories([101])
    print(f"Bridge result (M101 -> X <- M102): {results}")
    assert 102 in results
    
    # 2. Hub Protection Test
    cursor.execute("INSERT INTO memories (id, content_full) VALUES (1, 'Seed')")
    cursor.execute("INSERT INTO entities (id, name) VALUES (1000, 'Hub')")
    cursor.execute("INSERT INTO memory_relations VALUES (1, 'MEMORY', 1000, 'ENTITY', 'M', 1.0)")
    for i in range(2, 22):
        cursor.execute(f"INSERT INTO memories (id, content_full) VALUES ({i}, 'L{i}')")
        # Link: Leaf -> Hub
        cursor.execute(f"INSERT INTO memory_relations VALUES ({i}, 'MEMORY', 1000, 'ENTITY', 'M', {i/100.0})")
    
    # Depth 2: Memory 1 -> Hub (1000) -> Leaves (2..21)
    # Testing per-node limit of 5. Hub should only return 5 leaves.
    results = graph.get_related_memories([1], per_node_limit=5)
    print(f"Hub count (limit 5): {len(results)}")
    assert len(results) == 5
    assert 21 in results
    assert 20 in results
    assert 2 not in results
    
    print("ALL PRECISION TESTS PASSED.")

if __name__ == "__main__":
    run_test()
