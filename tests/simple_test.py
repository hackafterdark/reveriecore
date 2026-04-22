import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

# Ensure local imports work

def setup_standalone_mocks():
    """Only for running as a standalone script. pytest handles mocks via conftest.py."""
    if "pytest" in sys.modules:
        return

    import types
    
    # Mock agent
    agent = types.ModuleType("agent")
    agent.memory_provider = types.ModuleType("memory_provider")
    class BaseMemoryProvider:
        def __init__(self, *args, **kwargs): pass
    agent.memory_provider.MemoryProvider = BaseMemoryProvider
    sys.modules["agent"] = agent
    sys.modules["agent.memory_provider"] = agent.memory_provider
    
    # Mock hermes_constants safely
    hermes_constants = types.ModuleType("hermes_constants")
    # Use a local temp dir for standalone runs
    temp_dir = Path("/tmp/reverie_standalone_test")
    temp_dir.mkdir(parents=True, exist_ok=True)
    hermes_constants.get_hermes_home = lambda: temp_dir
    sys.modules["hermes_constants"] = hermes_constants
    
    # Mock tools
    tools = types.ModuleType("tools")
    tools.registry = types.ModuleType("registry")
    sys.modules["tools"] = tools
    sys.modules["tools.registry"] = tools.registry

if __name__ == "__main__":
    setup_standalone_mocks()

from reveriecore.database import DatabaseManager
from reveriecore.enrichment import EnrichmentService
from reveriecore.pruning import MesaService
from reveriecore.schemas import MemoryType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_test():
    # Use a test-specific DB name
    db_path = "simple_hierarchy_test.db"
    if os.path.exists(db_path): os.remove(db_path)
    
    db = DatabaseManager(db_path)
    enrichment = EnrichmentService({})
    
    # Mock enrichment logic
    enrichment.synthesize_memories = lambda mems, ent: f"Wisdom for {ent}"
    enrichment.generate_semantic_profile = lambda x: "profile"
    enrichment.calculate_importance = lambda x: {"score": 4.5, "expires_at": None}
    enrichment.generate_embedding = lambda x: [0.1] * 384
    
    mesa = MesaService(db, enrichment, age_days=0, importance_cutoff=10.0, interval_seconds=3600)
    mesa.consolidation_threshold = 3 
    
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
        
    db.close()
    if os.path.exists(db_path): os.remove(db_path)

if __name__ == "__main__":
    run_test()
