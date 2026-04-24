import sqlite3
import os
import sys
from datetime import datetime

# Setup isolated DB
db_path = "test_debug_query.db"
if os.path.exists(db_path): os.remove(db_path)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Schema
cursor.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, content_full TEXT, status TEXT, last_accessed_at DATETIME, importance_score REAL)")
cursor.execute("CREATE TABLE entities (id INTEGER PRIMARY KEY, name TEXT)")
cursor.execute("CREATE TABLE memory_relations (source_id INTEGER, source_type TEXT, target_id INTEGER, target_type TEXT, relation_type TEXT)")

# Data
cursor.execute("INSERT INTO entities (id, name) VALUES (1, 'TestEntity')")
for i in range(6):
    cursor.execute("INSERT INTO memories (id, content_full, status, last_accessed_at, importance_score) VALUES (?, ?, 'ACTIVE', datetime('now', '-20 days'), 2.5)", (i+1, f"text {i}"))
    cursor.execute("INSERT INTO memory_relations (source_id, source_type, target_id, target_type, relation_type) VALUES (?, 'MEMORY', 1, 'ENTITY', 'MENTIONS')", (i+1,))

conn.commit()

# Query
age_filter = "-0 days"
importance_cutoff = 5.0
consolidation_threshold = 5

query = f"""
    SELECT e.id, e.name, COUNT(ma.source_id) as c_count, GROUP_CONCAT(ma.source_id) as member_ids
    FROM memory_relations ma
    JOIN entities e ON ma.target_id = e.id
    JOIN memories m ON ma.source_id = m.id
    WHERE ma.source_type = 'MEMORY' 
    AND ma.target_type = 'ENTITY'
    AND m.status = 'ACTIVE'
    AND (m.last_accessed_at < datetime('now', ?) OR m.importance_score < ?)
    GROUP BY e.id
    HAVING c_count >= ?
"""

cursor.execute(query, (age_filter, importance_cutoff, consolidation_threshold))
results = cursor.fetchall()
print(f"Results: {results}")

os.remove(db_path)
