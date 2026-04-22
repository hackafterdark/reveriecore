import sys
import os
import sqlite3
from pathlib import Path
import logging

# Add project root to sys.path

def check_graph_status():
    # Avoid hermes_constants dependency to allow local python3 runs
    hermes_home = os.environ.get("HERMES_HOME")
    if hermes_home:
        db_path = Path(hermes_home) / "reveriecore.db"
    else:
        db_path = Path.home() / ".hermes" / "reveriecore.db"
    
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        return

    # Using direct sqlite3 to avoid dependency on sqlite_vec in user environment
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    print("\n--- Knowledge Graph Status ---")
    
    # 1. Check Entities
    cursor.execute("SELECT count(*) FROM entities")
    entity_count = cursor.fetchone()[0]
    print(f"Total Entities: {entity_count}")
    
    if entity_count > 0:
        print("\nTop 5 Entities:")
        cursor.execute("SELECT name, label FROM entities ORDER BY id DESC LIMIT 5")
        for name, label in cursor.fetchall():
            print(f" - [{label}] {name}")

    # 2. Check Associations
    cursor.execute("SELECT count(*) FROM memory_associations")
    assoc_count = cursor.fetchone()[0]
    print(f"\nTotal Associations (Triples): {assoc_count}")
    
    if assoc_count > 0:
        print("\nTop 5 Associations:")
        # We join with entities to make it readable
        query = """
            SELECT e1.name, ma.association_type, e2.name 
            FROM memory_associations ma
            JOIN entities e1 ON ma.source_id = e1.id AND ma.source_type = 'ENTITY'
            JOIN entities e2 ON ma.target_id = e2.id AND ma.target_type = 'ENTITY'
            ORDER BY ma.id DESC LIMIT 5
        """
        cursor.execute(query)
        for src, pred, tgt in cursor.fetchall():
            print(f" - {src} --[{pred}]--> {tgt}")

    # 3. Check High-Importance Memories waiting for extraction
    cursor.execute("SELECT count(*) FROM memories WHERE importance_score >= 3.0")
    high_imp_count = cursor.fetchone()[0]
    print(f"\nMemories with Importance >= 3.0: {high_imp_count}")

    print("\n--- End of Status ---\n")

if __name__ == "__main__":
    check_graph_status()
