import os
import sys
import json
import logging
from datetime import datetime
from typing import List

# Ensure local imports work
sys.path.append(os.getcwd())

# --- Mock Hermes Core ---
import types
from unittest.mock import MagicMock

def setup_mocks():
    agent = types.ModuleType("agent")
    memory_provider = types.ModuleType("memory_provider")
    class BaseMemoryProvider:
        def __init__(self, *args, **kwargs): pass
        def initialize(self, *args, **kwargs): pass
        def shutdown(self, *args, **kwargs): pass
        def sync_turn(self, *args, **kwargs): pass
    memory_provider.MemoryProvider = BaseMemoryProvider
    agent.memory_provider = memory_provider
    sys.modules["agent"] = agent
    sys.modules["agent.memory_provider"] = memory_provider
    sys.modules["hermes_constants"] = MagicMock()
    
    tools = types.ModuleType("tools")
    registry = types.ModuleType("registry")
    registry.tool_error = lambda x: f"ERROR: {x}"
    tools.registry = registry
    sys.modules["tools"] = tools
    sys.modules["tools.registry"] = registry

setup_mocks()

import schemas, database, enrichment, retrieval, pruning, provider

from database import DatabaseManager
from enrichment import EnrichmentService
from pruning import MesaService
from retrieval import Retriever
from provider import ReverieMemoryProvider
from schemas import MemoryType, AssociationType

# Setup logging to console for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_tree_of_nuance():
    """
    Verification script for Hierarchical Cluster Consolidation (Tier 1.5).
    1. Injects 6 related fragments.
    2. Runs Mesa Hierarchical Consolidation.
    3. Verifies Anchor creation and CHILD_OF links.
    4. Verifies Retriever signals child IDs.
    5. Verifies recall_reverie tool security and functionality.
    """
    db_path = "test_hierarchy_nuance.db"
    if os.path.exists(db_path): os.remove(db_path)
    
    try:
        db = DatabaseManager(db_path)
        enrichment = EnrichmentService({})
        
        # Mock LLM Synthesis to produce a clear Anchor
        enrichment.synthesize_memories = lambda mems, ent: f"CONSOLIDATED wisdom for {ent}. Sources: {list(mems.keys())}"
        enrichment.generate_semantic_profile = lambda x: "profile"
        enrichment.calculate_importance = lambda x: {"score": 4.5, "expires_at": None}
        enrichment.generate_embedding = lambda x: [0.1] * 384
        
        # Configure Mesa with low threshold for testing
        mesa = MesaService(db, enrichment, age_days=0, importance_cutoff=5.0, interval_seconds=3600)
        mesa.consolidation_threshold = 5 
        
        retriever = Retriever(db, enrichment)
        
        # Identity for security testing
        provider_inst = ReverieMemoryProvider()
        provider_inst._db = db
        provider_inst._enrichment = enrichment
        provider_inst._retriever = retriever
        provider_inst.owner_id = "TEST_USER_1"

        cursor = db.get_cursor()

        # 1. Setup Entity
        target_entity = "Connection Leak Bug"
        cursor.execute("INSERT INTO entities (name, label) VALUES (?, 'BUG')", (target_entity,))
        ent_id = cursor.lastrowid

        # 2. Inject 6 fragmented memories (dated in the past to trigger Mesa)
        fragments = [
            "Connection leak detected in the database pool during high load.",
            "Leak seems related to unclosed cursors in the pruning.py module.",
            "Issue identified: cursor.close() was missing in the exception handler.",
            "Fix attempted: added try/finally blocks to ensure closure.",
            "Validation: Memory usage stabilized after the fix was deployed.",
            "Note: The leak only occurred when RETENTION_DAYS was set to > 30."
        ]
        
        fragment_ids = []
        for i, text in enumerate(fragments):
            # Create dummy vector
            import sqlite_vec
            vec = [0.1] * 384
            
            cursor.execute("""
                INSERT INTO memories (content_full, content_abstract, importance_score, last_accessed_at, owner_id, status)
                VALUES (?, ?, ?, datetime('now', '-20 days'), 'TEST_USER_1', 'ACTIVE')
            """, (text, f"Sub-point {i}", 2.5))
            mid = cursor.lastrowid
            fragment_ids.append(mid)
            
            cursor.execute("INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)", 
                         (mid, sqlite_vec.serialize_float32(vec)))
            
            # Link to entity
            cursor.execute("""
                INSERT INTO memory_associations (source_id, source_type, target_id, target_type, association_type)
                VALUES (?, 'MEMORY', ?, 'ENTITY', 'MENTIONS')
            """, (mid, ent_id))

        db.commit()
        logger.info(f"Step 1: Injected {len(fragment_ids)} fragments linked to '{target_entity}'.")

        cursor.execute("SELECT id, content_full, status, importance_score FROM memories")
        logger.info(f"Database Memories: {cursor.fetchall()}")
        cursor.execute("SELECT * FROM memory_associations")
        logger.info(f"Database Associations: {cursor.fetchall()}")

        # 3. Run Hierarchical Consolidation
        mesa.run_hierarchical_consolidation()
        
        # 4. Verification: Anchor Creation
        cursor.execute("SELECT id, content_full, memory_type, metadata FROM memories WHERE memory_type = 'OBSERVATION' AND status = 'ACTIVE'")
        anchors = cursor.fetchall()
        
        assert len(anchors) >= 1, "Should have created at least one Observation Anchor."
        anchor_id, anchor_text, m_type, metadata_json = anchors[0]
        metadata = json.loads(metadata_json)
        
        logger.info(f"Step 2: Observation Anchor created (ID: {anchor_id}).")
        logger.info(f"Summary Content: {anchor_text[:100]}...")
        assert "source_ids" in metadata
        assert len(metadata["source_ids"]) >= 5
        
        # 5. Verification: CHILD_OF Links
        cursor.execute("SELECT source_id FROM memory_associations WHERE target_id = ? AND association_type = 'CHILD_OF'", (anchor_id,))
        children = [row[0] for row in cursor.fetchall()]
        logger.info(f"Step 3: Found {len(children)} CHILD_OF links to Anchor.")
        assert len(children) >= 5

        # 6. Verification: Retrieval Discovery
        # Query for the entity topic
        results = retriever.search([0.1]*384, query_text="connection bug", limit=5)
        found_anchor = next((r for r in results if r["id"] == anchor_id), None)
        
        assert found_anchor is not None, "Retriever should find the Anchor."
        logger.info("Step 4: Retriever successfully found Anchor.")
        assert "[Nuanced Details available via recall_reverie" in found_anchor["content"]
        logger.info("Step 5: Retriever correctly signaled Child IDs in context.")

        # 7. Verification: Tool Functionality (recall_reverie)
        child_id = fragment_ids[0]
        # Test authorized recall
        logger.info(f"Step 6: Testing safe recall of Child ID {child_id}...")
        resp = provider_inst.handle_tool_call("recall_reverie", {"memory_id": child_id})
        assert "### RECALLED NUANCE" in resp
        assert fragments[0] in resp
        
        # Test UN-authorized recall (Switch owner)
        provider_inst.owner_id = "ATTACKER_AGENT"
        logger.info("Step 7: Testing unauthorized recall (Expect Security Denial)...")
        resp = provider_inst.handle_tool_call("recall_reverie", {"memory_id": child_id})
        assert "Access Denied" in resp
        logger.info("Step 8: Security validation SUCCESSFUL.")

        print("\n" + "="*40)
        print("THE TREE OF NUANCE: TEST PASSED")
        print("="*40)

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        if os.path.exists(db_path): os.remove(db_path)

if __name__ == "__main__":
    test_tree_of_nuance()
