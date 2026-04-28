import os
import sys
import json
import logging
from datetime import datetime
from typing import List
import pytest
from unittest.mock import MagicMock

# Ensure local imports work

from reveriecore.database import DatabaseManager
from reveriecore.enrichment import EnrichmentService
from reveriecore.pruning import MesaService
from reveriecore.retrieval import Retriever
from reveriecore.provider import ReverieMemoryProvider
from reveriecore.schemas import MemoryType, RelationType

def test_tree_of_nuance(tmp_path):
    """
    Verification script for Hierarchical Cluster Consolidation (Tier 1.5).

    1. Injects 6 related fragments.
    2. Runs Mesa Hierarchical Consolidation.
    3. Verifies Anchor creation and CHILD_OF links.
    4. Verifies Retriever signals child IDs.
    5. Verifies recall_reverie tool security and functionality.
    """
    db_path = str(tmp_path / "test_hierarchy_nuance.db")
    
    try:
        db = DatabaseManager(db_path)
        # Use an EnrichmentService instance configured for testing
        enrichment = EnrichmentService({})
        
        # Mock LLM Synthesis results for deterministic testing
        enrichment.synthesize_memories = MagicMock(side_effect=lambda mems, ent: f"CONSOLIDATED wisdom for {ent}. Sources: {list(mems.keys())}")
        enrichment.generate_semantic_profile = MagicMock(return_value="profile")
        enrichment.calculate_importance = MagicMock(return_value={"score": 4.5, "expires_at": None})
        enrichment.generate_embedding = MagicMock(return_value=[0.1] * 384)
        
        # Configure Mesa with low threshold for testing
        from reveriecore.retrieval import MesaConfig
        config = MesaConfig(retention_days=0, importance_cutoff=5.0, interval_seconds=3600)
        mesa = MesaService(db, enrichment, config=config)
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
                INSERT INTO memory_relations (source_id, source_type, target_id, target_type, relation_type)
                VALUES (?, 'MEMORY', ?, 'ENTITY', 'MENTIONS')
            """, (mid, ent_id))

        db.commit()

        # 3. Run Hierarchical Consolidation
        mesa.run_hierarchical_consolidation()
        
        # 4. Verification: Anchor Creation
        cursor.execute("SELECT id, content_full, memory_type, metadata FROM memories WHERE memory_type = 'OBSERVATION' AND status = 'ACTIVE'")
        anchors = cursor.fetchall()
        
        assert len(anchors) >= 1, "Should have created at least one Observation Anchor."
        anchor_id, anchor_text, m_type, metadata_json = anchors[0]
        metadata = json.loads(metadata_json)
        
        assert "source_ids" in metadata
        assert len(metadata["source_ids"]) >= 5
        
        # 5. Verification: CHILD_OF Links
        cursor.execute("SELECT source_id FROM memory_relations WHERE target_id = ? AND relation_type = 'CHILD_OF'", (anchor_id,))
        children = [row[0] for row in cursor.fetchall()]
        assert len(children) >= 5

        # 6. Verification: Retrieval Discovery
        # Query for the entity topic
        results = retriever.search([0.1]*384, query_text="connection bug", limit=5)
        found_anchor = next((r for r in results if r["id"] == anchor_id), None)
        
        assert found_anchor is not None, "Retriever should find the Anchor."
        assert "[Nuanced Details available via recall_reverie" in found_anchor["content"]

        # 7. Verification: Tool Functionality (recall_reverie)
        child_id = fragment_ids[0]
        # Test authorized recall
        resp = provider_inst.handle_tool_call("recall_reverie", {"memory_id": child_id})
        assert "### RECALLED NUANCE" in resp
        assert fragments[0] in resp
        
        # Test UN-authorized recall (Switch owner)
        provider_inst.owner_id = "ATTACKER_AGENT"
        resp = provider_inst.handle_tool_call("recall_reverie", {"memory_id": child_id})
        assert "Access Denied" in resp

    finally:
        if 'db' in locals():
            db.close()
