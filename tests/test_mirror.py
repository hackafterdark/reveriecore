import os
import pytest
from pathlib import Path
import shutil
import uuid
import json
from unittest.mock import MagicMock
from reveriecore.database import DatabaseManager
from reveriecore.mirror import MirrorService

@pytest.fixture(autouse=True)
def reset_singleton():
    """Ensures each test gets a fresh DatabaseManager instance."""
    DatabaseManager._instance = None
    yield
    DatabaseManager._instance = None

@pytest.fixture
def test_env(tmp_path):
    db_path = tmp_path / "test_reveries.db"
    archive_path = tmp_path / "test_archive"
    # Ensure fresh DB path for each test
    if db_path.exists():
        db_path.unlink()
        
    db = DatabaseManager(str(db_path))
    
    # Mock Enrichment
    enrichment = MagicMock()
    enrichment.generate_embedding.return_value = [0.1] * 384
    enrichment.count_tokens.return_value = 10
    
    mirror = MirrorService(db, enrichment, archive_root=archive_path)
    return db, enrichment, mirror, archive_path

def test_export_import_cycle(test_env):
    db, enrichment, mirror, archive_path = test_env
    
    # 1. Add a memory
    content = "Hello, this is a test memory for mirroring."
    guid = str(uuid.uuid4())
    learned_at = "2024-05-21T12:00:00" # Use a known date for Hive pathing check
    with db.write_lock() as cursor:
        cursor.execute("""
            INSERT INTO memories (content_full, content_abstract, guid, author_id, learned_at, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (content, "Abstract here", guid, "TEST_USER", learned_at, "ACTIVE"))
        mem_id = cursor.lastrowid
    
    # 2. Export
    mirror.export_node(mem_id)
    
    # 3. Verify file exists with Hive pathing
    # year=2024/month=05/day=21/{guid}.md
    hive_path = archive_path / "year=2024" / "month=05" / "day=21" / f"{guid}.md"
    assert hive_path.exists()
    
    # Check content
    with open(hive_path, "r") as f:
        md_content = f.read()
        assert guid in md_content
        assert content in md_content
        assert "Abstract here" in md_content
    
    # 4. Wipe DB
    with db.write_lock() as cursor:
        cursor.execute("DELETE FROM memories")
        cursor.execute("DELETE FROM memories_vec")
    
    assert db.get_memory_by_guid(guid) is None
    
    # 5. Import
    mirror.import_archive()
    
    # 6. Verify restored
    restored = db.get_memory_by_guid(guid)
    assert restored is not None
    assert restored['content_full'] == content
    assert restored['guid'] == guid
    assert restored['status'] == "ACTIVE"
    assert "2024-05-21" in restored['learned_at']

def test_purged_status_handling(test_env):
    db, enrichment, mirror, archive_path = test_env
    
    # 1. Add a memory and export it
    content = "To be purged"
    guid = str(uuid.uuid4())
    with db.write_lock() as cursor:
        cursor.execute("INSERT INTO memories (content_full, guid) VALUES (?, ?)", (content, guid))
        mem_id = cursor.lastrowid
    mirror.export_node(mem_id)
    
    # 2. Find the file and change status to PURGED
    files = list(archive_path.glob("**/*.md"))
    with open(files[0], "r") as f:
        lines = f.readlines()
    
    with open(files[0], "w") as f:
        for line in lines:
            if line.startswith("status:"):
                f.write("status: PURGED\n")
            else:
                f.write(line)
                
    # 3. Import
    mirror.import_archive()
    
    # 4. Memory should be deleted from DB
    assert db.get_memory_by_guid(guid) is None
def test_graph_integrity_across_mirror(test_env):
    db, enrichment, mirror, archive_path = test_env
    
    # 1. Create a structured graph
    with db.write_lock() as cursor:
        # Memory A
        guid_a = str(uuid.uuid4())
        cursor.execute("INSERT INTO memories (content_full, guid) VALUES (?, ?)", ("Memory A", guid_a))
        id_a = cursor.lastrowid
        
        # Memory B
        guid_b = str(uuid.uuid4())
        cursor.execute("INSERT INTO memories (content_full, guid) VALUES (?, ?)", ("Memory B", guid_b))
        id_b = cursor.lastrowid
        
        # Entity E
        guid_e = str(uuid.uuid4())
        cursor.execute("INSERT INTO entities (name, label, guid) VALUES (?, ?, ?)", ("Entity E", "PERSON", guid_e))
        id_e = cursor.lastrowid
        
        # Association: A -> B (PRECEEDS)
        cursor.execute("""
            INSERT INTO memory_relations (source_id, source_type, target_id, target_type, relation_type)
            VALUES (?, 'MEMORY', ?, 'MEMORY', 'PRECEEDS')
        """, (id_a, id_b))
        
        # Association: A -> E (MENTIONS)
        cursor.execute("""
            INSERT INTO memory_relations (source_id, source_type, target_id, target_type, relation_type)
            VALUES (?, 'MEMORY', ?, 'ENTITY', 'MENTIONS')
        """, (id_a, id_e))

    # 2. Export Memory A and Memory B (should include associations in frontmatter)
    mirror.export_node(id_a)
    mirror.export_node(id_b)
    # Note: Entities don't have their own .md files yet, but they are referenced in Memory A's frontmatter.
    
    # 3. Wipe and Import
    with db.write_lock() as cursor:
        cursor.execute("DELETE FROM memories")
        cursor.execute("DELETE FROM memory_relations")
    
    mirror.import_archive()
    
    # 4. Verify Reconstruction
    restored_a = db.get_memory_by_guid(guid_a)
    assert restored_a is not None
    
    # In the second pass, we also imported Memory B and Entity E
    restored_b = db.get_memory_by_guid(guid_b)
    assert restored_b is not None
    
    # Restore associations should have linked them
    assocs = db.get_relations_for_node(restored_a['id'], 'MEMORY')
    assert len(assocs) >= 2
    
    # Check for PRECEEDS link to B
    preceeds = [a for a in assocs if a['relation_type'] == 'PRECEEDS']
    assert len(preceeds) == 1
    assert preceeds[0]['target_id'] == restored_b['id']
    
    # Check for MENTIONS link to E
    mentions = [a for a in assocs if a['relation_type'] == 'MENTIONS']
    assert len(mentions) == 1
    # We need to find Entity E's restored ID
    restored_e = db.get_entity_by_guid(guid_e)
    assert restored_e is not None
    assert mentions[0]['target_id'] == restored_e['id']

def test_lazy_revectorization_worker(test_env):
    db, enrichment, mirror, archive_path = test_env
    
    # 1. Setup an archive file manually
    guid = str(uuid.uuid4())
    archive_path.mkdir(parents=True, exist_ok=True)
    file_path = archive_path / f"{guid}.md"
    with open(file_path, "w") as f:
        f.write("---\n")
        f.write(f"guid: {guid}\n")
        f.write("type: OBSERVATION\n")
        f.write("---\n")
        f.write("Imported memory content")
    
    # 2. Start worker and Import
    mirror.start()
    try:
        mirror.import_archive()
        
        # 3. Wait for worker (it processes the queue)
        import time
        # The worker waits 5s if queue is empty, so we should wait a bit
        timeout = 10
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check if embedding was generated (mock enrichment was called)
            if enrichment.generate_embedding.called:
                break
            time.sleep(0.5)
            
        assert enrichment.generate_embedding.called, "Worker did not process re-vectorization"
        
        # Verify DB updated with vector (mocked as [0.1]*384)
        restored = db.get_memory_by_guid(guid)
        # We need to check if the vector exists in memories_vec
        cursor = db.get_cursor()
        cursor.execute("SELECT rowid FROM memories_vec WHERE rowid = ?", (restored['id'],))
        assert cursor.fetchone() is not None
        
    finally:
        mirror.stop()
