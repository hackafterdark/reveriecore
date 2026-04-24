import sys
import os
import json
import sqlite3
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add project root to sys.path

from reveriecore.database import DatabaseManager
from reveriecore.enrichment import EnrichmentService
from reveriecore.schemas import RelationType

def test_full_graph_pipeline(tmp_path):
    # 1. Setup Mock DB
    db_path = str(tmp_path / "test_reverie_v2.db")
    
    db = DatabaseManager(db_path)
    cursor = db.get_cursor()
    
    # 2. Setup Enrichment with Mock LLM
    with patch("reveriecore.enrichment.InternalLLMClient.is_connected", return_value=True), \
         patch("reveriecore.enrichment.InternalLLMClient.call") as mock_call:
        
        enrichment = EnrichmentService()
        
        # Pass 1: Entity Response, Pass 2: Triple Response
        mock_call.side_effect = [
            {"entities": [
                {"name": "database.py", "type": "FILE"},
                {"name": "DatabaseManager", "type": "CLASS"}
            ]},
            {"triples": [
                {"source": "DatabaseManager", "predicate": "DEFINED_IN", "target": "database.py"}
            ]}
        ]

        # 3. Simulate sync_turn extraction
        text = "I updated the database.py file."
        mem_id = 1
        with db.write_lock() as cursor_inner:
            cursor_inner.execute("INSERT INTO memories (id, content_full) VALUES (?, ?)", (mem_id, text))
        
        enrichment.extract_graph_data(text, mem_id, db)
    
        # 4. Verify
        cursor.execute("SELECT name, label FROM entities")
        entities = cursor.fetchall()
        assert len(entities) == 2
        
        cursor.execute("SELECT source_id, target_id, relation_type FROM memory_relations WHERE relation_type = 'MENTIONS'")
        mentions = cursor.fetchall()
        assert len(mentions) == 2
        
        cursor.execute("SELECT source_id, target_id, relation_type FROM memory_relations WHERE relation_type = 'DEFINED_IN'")
        triples = cursor.fetchall()
        assert len(triples) == 1
        
    db.close()

if __name__ == "__main__":
    test_full_graph_pipeline()
