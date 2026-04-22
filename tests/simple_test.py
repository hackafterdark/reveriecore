import os
import sys
import json
import logging
from datetime import datetime

# Ensure local imports work
sys.path.append(os.getcwd())

# Mock Hermes
from unittest.mock import MagicMock
import types
sys.modules["agent"] = MagicMock()
sys.modules["agent.memory_provider"] = MagicMock()
sys.modules["hermes_constants"] = MagicMock()
sys.modules["tools"] = MagicMock()
sys.modules["tools.registry"] = MagicMock()

from reveriecore.database import DatabaseManager
from reveriecore.enrichment import EnrichmentService
from reveriecore.pruning import MesaService
from reveriecore.retrieval import Retriever
from reveriecore.schemas import MemoryType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_test():
    db_path = "simple_hierarchy_test.db"
    if os.path.exists(db_path): os.remove(db_path)
    
    db = DatabaseManager(db_path)
    enrichment = EnrichmentService({})
    # Mocks
    enrichment.synthesize_memories = lambda mems, ent: f"Wisdom for {ent}"
    enrichment.generate_semantic_profile = lambda x: "profile"
    enrichment.calculate_importance = lambda x: {"score": 4.5, "expires_at": None}
    enrichment.generate_embedding = lambda x: [0.1] * 384
    
    mesa = MesaService(db, enrichment, age_days=0, importance_cutoff=10.0, interval_seconds=3600)
    mesa.consolidation_threshold = 3 # Lower for speed
    
    with db.write_lock() as cursor:
        cursor.execute("INSERT INTO entities (name, label) VALUES ('TestEnt', 'FILE')")
        ent_id = cursor.lastrowid
        
        for i in range(4):
            cursor.execute("INSERT INTO memories (content_full, status, importance_score) VALUES (?, 'ACTIVE', 1.0)", (f"m {i}",))
            mid = cursor.lastrowid
            cursor.execute("INSERT INTO memory_associations (source_id, source_type, target_id, target_type, association_type) VALUES (?, 'MEMORY', ?, 'ENTITY', 'MENTIONS')", (mid, ent_id))

    logger.info("Running consolidation...")
    mesa.run_hierarchical_consolidation()
    
    cursor = db.get_cursor()
    cursor.execute("SELECT id FROM memories WHERE memory_type = 'OBSERVATION'")
    anchors = cursor.fetchall()
    logger.info(f"Anchors found: {anchors}")
    
    if len(anchors) > 0:
        print("SUCCESS")
    else:
        print("FAILURE")
        
    os.remove(db_path)

if __name__ == "__main__":
    run_test()
