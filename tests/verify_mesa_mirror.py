import pytest
import uuid
import json
from pathlib import Path
from unittest.mock import MagicMock
from reveriecore.database import DatabaseManager
from reveriecore.mirror import MirrorService
from reveriecore.pruning import MesaService

@pytest.fixture(autouse=True)
def reset_singleton():
    """Ensures each test gets a fresh DatabaseManager instance."""
    DatabaseManager._instance = None
    yield
    DatabaseManager._instance = None

def test_mesa_integration_mirrors_on_archival(tmp_path):
    """
    Verifies that MESA's archival process automatically triggers a MirrorService export.
    """
    db_path = str(tmp_path / "mesa_test_reveries.db")
    archive_path = tmp_path / "mesa_archive"
    db = DatabaseManager(db_path)
    
    # Mock Enrichment and Mirror
    # We want to see if MesaService calls mirror.export_node
    enrichment = MagicMock()
    enrichment.count_tokens.return_value = 10
    
    # We use a real MirrorService but we can spy on export_node if we want, 
    # or just check the filesystem result.
    mirror = MirrorService(db, enrichment, archive_root=archive_path)
    
    # Initialize MESA with our services
    mesa = MesaService(db, enrichment, mirror)
    
    # 1. Inject fragmented memories (importance < 3.0)
    with db.write_lock() as cursor:
        guids = []
        for i in range(5):
            g = str(uuid.uuid4())
            guids.append(g)
            # Set a past date so they are eligible for pruning
            past_date = "2024-01-01 12:00:00"
            cursor.execute("""
                INSERT INTO memories (content_full, guid, importance_score, status, learned_at, last_accessed_at)
                VALUES (?, ?, ?, 'ACTIVE', ?, ?)
            """, (f"Fragment {i}", g, 1.5, past_date, past_date))
            
    # 2. Run Soft Prune
    mesa.max_age_days = 0
    mesa.run_soft_prune()
    
    # 3. Verify Filesystem
    # There should be 5 markdown files in the archive
    md_files = list(archive_path.glob("**/*.md"))
    assert len(md_files) == 5
    
    # Verify they are marked 'ARCHIVED' in their frontmatter
    for md_file in md_files:
        with open(md_file, "r") as f:
            content = f.read()
            assert "status: ARCHIVED" in content
            
def test_mesa_integration_mirrors_on_synthesis(tmp_path):
    """
    Verifies that MESA's hierarchical consolidation mirrors both the new Observation 
    and the archived children.
    """
    db_path = str(tmp_path / "mesa_synthesis_reveries.db")
    archive_path = tmp_path / "mesa_synthesis_archive"
    db = DatabaseManager(db_path)
    
    enrichment = MagicMock()
    enrichment.count_tokens.return_value = 10
    enrichment.generate_embedding.return_value = [0.1] * 384
    
    # Mock LLM response for consolidation
    # These methods are on the enrichment service, not mesa service
    enrichment.synthesize_memories.return_value = "Synthesized Observation"
    enrichment.generate_semantic_profile.return_value = "Synthetic Profile"
    enrichment.calculate_importance.return_value = {"score": 4.5}
    
    mesa = MesaService(db, enrichment, MirrorService(db, enrichment, archive_root=archive_path))
    
    # 1. Inject memories with strong association to an entity
    child_ids = []
    with db.write_lock() as cursor:
        cursor.execute("INSERT INTO entities (name, label, guid) VALUES ('TargetEntity', 'CONCEPT', 'e-1')")
        ent_id = cursor.lastrowid
        
        child_guids = []
        for i in range(3):
            g = str(uuid.uuid4())
            child_guids.append(g)
            cursor.execute("INSERT INTO memories (content_full, guid, status) VALUES (?, ?, 'ACTIVE')", (f"Child {i}", g))
            mid = cursor.lastrowid
            child_ids.append(mid)
            cursor.execute("INSERT INTO memory_associations (source_id, target_id, target_type, association_type) VALUES (?, ?, 'ENTITY', 'MENTIONS')", (mid, ent_id))

    # 2. Trigger Consolidation
    # We'll call the internal method for testing
    mesa._consolidate_to_hierarchy(child_ids, "TargetEntity", ent_id)
    
    # 3. Verify Mirror
    md_files = list(archive_path.glob("**/*.md"))
    # Should have 3 children (now ARCHIVED) + 1 new Observation (ACTIVE)
    assert len(md_files) == 4
    
    # Find the Observation file
    active_files = []
    for md_file in md_files:
        with open(md_file, "r") as f:
            if "status: ACTIVE" in f.read():
                active_files.append(md_file)
    
    assert len(active_files) == 1
    with open(active_files[0], "r") as f:
        content = f.read()
        assert "Synthesized Observation" in content
        assert "type: OBSERVATION" in content
