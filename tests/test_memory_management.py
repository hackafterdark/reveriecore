import pytest
import sys
import os
from unittest.mock import MagicMock
from reveriecore.provider import ReverieMemoryProvider

@pytest.fixture(scope="function")
def provider(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    
    p = ReverieMemoryProvider()
    p._warm_cache = MagicMock()
    p.initialize("test_session", user_id="test_user")
    
    # Ensure fresh DB for each function-scoped test
    cursor = p._db.get_cursor()
    cursor.execute("DELETE FROM memories")
    cursor.execute("DELETE FROM entities")
    cursor.execute("DELETE FROM memory_associations")
    p._db.commit()
    
    # Mock heavy enrichment bits
    p._enrichment.generate_embedding = MagicMock(return_value=[0.1]*384)
    p._enrichment.synthesize_memories = MagicMock(return_value="Merged Memory: keep it safe")
    # Ensure LLM is skipped in these tests
    p._enrichment.llm_client.is_connected = MagicMock(return_value=False)
    
    return p

def test_memory_remove_workflow(provider):
    # Call synchronous helper directly to avoid threading in tests
    provider._save_memory_sync("What is Project Bluebird?", "Secret Project: Bluebird is located in Sector 7.", session_id="test_session")
    
    cursor = provider._db.get_cursor()
    cursor.execute("SELECT id FROM memories WHERE content_full LIKE '%Bluebird%'")
    row = cursor.fetchone()
    assert row is not None, "Memory was not saved synchronously."
    mem_id = row[0]

    # Search for removal
    # We search for 'Bluebird' which should have the same mocked embedding [0.1]*384
    search_args = {"action": "remove", "text": "Bluebird"}
    search_resp = provider.handle_tool_call("memory", search_args)
    
    assert "Potential Matches" in search_resp
    assert f"ID: {mem_id}" in search_resp

    # Confirm removal
    confirm_args = {"action": "remove", "memory_id": mem_id}
    confirm_resp = provider.handle_tool_call("memory", confirm_args)
    assert "deleted" in confirm_resp

    # Verify
    cursor.execute("SELECT id FROM memories WHERE id = ?", (mem_id,))
    assert cursor.fetchone() is None

def test_canonical_merge_deduplication(provider):
    """Test that highly similar memories are merged rather than duplicated."""
    text1 = "The server login is admin / password123."
    provider._save_memory_sync("Login info?", text1, session_id="test_session")
    
    cursor = provider._db.get_cursor()
    cursor.execute("SELECT id FROM memories WHERE content_full LIKE '%password123%'")
    row = cursor.fetchone()
    assert row is not None
    mem_id = row[0]
    provider._db.commit() # Finish the first read transaction to avoid stale views
    
    # Mock retriever to find the first memory as a duplicate

    provider._retriever.find_duplicates = MagicMock(return_value=[{"id": mem_id, "content_full": text1, "similarity": 0.99}])
    
    text2 = "The server login is admin / password123. Please keep it safe."
    provider._save_memory_sync("Login info reminder", text2, session_id="test_session")
    
    # Verify merge happened (no new row)
    # Use get_memory to ensure we're using the provider's standard data access path
    updated_mem = provider._db.get_memory(mem_id)
    
    assert updated_mem is not None
    assert updated_mem["id"] == mem_id
    assert "keep it safe" in updated_mem["content_full"]
