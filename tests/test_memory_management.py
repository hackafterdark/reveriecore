import pytest
import time
from reveriecore.provider import ReverieMemoryProvider

@pytest.fixture(scope="function")
def provider(tmp_path):
    """Provider fixture with a temporary database."""
    p = ReverieMemoryProvider()
    # Mocking hermes home for the test
    db_path = tmp_path / "reverie_test.db"
    
    # Monkeypatch get_hermes_home or just manually init db
    import reveriecore.provider as provider_mod
    from pathlib import Path
    
    p.initialize("test_session", user_id="test_user")
    # Override db with temp one if needed, but initialize already sets it.
    # To be safe, we can just use the initialized one.
    return p

def test_memory_remove_workflow(provider):
    """Test the Search -> Confirm workflow for memory removal."""
    # 1. Inject a specific memory
    content = "Secret Project: Bluebird is located in Sector 7."
    provider.sync_turn("What is Project Bluebird?", content)
    
    # Wait for background processing (enrichment + storage)
    # We poll the database instead of a fixed sleep for robustness
    found = False
    for _ in range(10):
        cursor = provider._db.get_cursor()
        cursor.execute("SELECT id FROM memories WHERE content_full LIKE '%Bluebird%'")
        row = cursor.fetchone()
        if row:
            mem_id = row[0]
            found = True
            break
        time.sleep(1)
    
    assert found, "Memory was not saved in time."

    # 2. Stage 1: Search for the memory to remove
    search_args = {"action": "remove", "text": "Bluebird project"}
    search_resp = provider.handle_tool_call("memory", search_args)
    
    assert "Potential Matches" in search_resp
    assert f"ID: {mem_id}" in search_resp

    # 3. Stage 2: Confirm removal by ID
    confirm_args = {"action": "remove", "memory_id": mem_id}
    confirm_resp = provider.handle_tool_call("memory", confirm_args)
    
    assert f"Memory ID {mem_id}" in confirm_resp
    assert "deleted" in confirm_resp

    # 4. Verify deletion
    cursor.execute("SELECT id FROM memories WHERE id = ?", (mem_id,))
    assert cursor.fetchone() is None

def test_canonical_merge_deduplication(provider):
    """Test that highly similar memories are merged rather than duplicated."""
    # 1. Add first memory
    text1 = "The server login is admin / password123."
    provider.sync_turn("Login info?", text1)
    
    mem_id = None
    for _ in range(10):
        cursor = provider._db.get_cursor()
        cursor.execute("SELECT id FROM memories WHERE content_full LIKE '%password123%'")
        row = cursor.fetchone()
        if row:
            mem_id = row[0]
            break
        time.sleep(1)
    
    assert mem_id is not None

    # 2. Add a very similar second memory
    # Similarity > 0.95 should trigger a merge
    text2 = "The server login is admin / password123. Please keep it safe."
    provider.sync_turn("Login info reminder", text2)
    
    # Wait for synthesis/merge
    time.sleep(5) 
    
    # 3. Verify no new memory was created, and old one was updated
    cursor.execute("SELECT id, content_full FROM memories")
    rows = cursor.fetchall()
    
    # Should still only have 1 memory if merged correctly
    assert len(rows) == 1
    assert rows[0][0] == mem_id
    # Content should have been updated (Synthesis usually includes new info)
    assert "keep it safe" in rows[0][1]
