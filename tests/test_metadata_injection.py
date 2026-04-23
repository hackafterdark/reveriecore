import pytest
import json
from reveriecore.retrieval import Retriever
from reveriecore.database import DatabaseManager

def test_importance_scale_and_labeling(db_manager, enrichment_service):
    """Verify that 0-10 scale maps correctly to categorical labels in the context."""
    retriever = Retriever(db_manager)
    cursor = db_manager.get_cursor()
    
    # Clean up
    cursor.execute("DELETE FROM memories")
    cursor.execute("DELETE FROM memories_vec")
    
    # 1. Test Incidental (Score 2.0)
    # We'll insert directly to control the score
    content_inc = "This is a minor detail."
    vec_inc = enrichment_service.generate_embedding(content_inc)
    cursor.execute("""
        INSERT INTO memories (content_full, importance_score, guid, memory_type, learned_at) 
        VALUES (?, ?, ?, ?, ?)
    """, (content_inc, 2.0, "guid-incidental", "CONVERSATION", "2024-04-23 10:00:00"))
    id_inc = cursor.lastrowid
    import sqlite_vec
    cursor.execute("INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)", 
                   (id_inc, sqlite_vec.serialize_float32(vec_inc)))
    
    # 2. Test Relevant (Score 5.5)
    content_rel = "This is a relevant fact."
    vec_rel = enrichment_service.generate_embedding(content_rel)
    cursor.execute("""
        INSERT INTO memories (content_full, importance_score, guid, memory_type, learned_at, metadata) 
        VALUES (?, ?, ?, ?, ?, ?)
    """, (content_rel, 5.5, "guid-relevant", "USER_PREFERENCE", "2024-04-23 11:00:00", json.dumps({"location": "San Francisco"})))
    id_rel = cursor.lastrowid
    cursor.execute("INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)", 
                   (id_rel, sqlite_vec.serialize_float32(vec_rel)))
    
    # 3. Test Critical (Score 9.0)
    content_crit = "This is a critical instruction!"
    vec_crit = enrichment_service.generate_embedding(content_crit)
    cursor.execute("""
        INSERT INTO memories (content_full, importance_score, guid, memory_type, learned_at) 
        VALUES (?, ?, ?, ?, ?)
    """, (content_crit, 9.0, "guid-critical", "TASK", "2024-04-23 12:00:00"))
    id_crit = cursor.lastrowid
    cursor.execute("INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)", 
                   (id_crit, sqlite_vec.serialize_float32(vec_crit)))
    
    db_manager.commit()
    
    # Search
    results = retriever.search(vec_inc, limit=3, similarity_weight=0.1, importance_weight=0.9)
    
    # Verify contents
    res_inc = next(r for r in results if r['id'] == id_inc)
    res_rel = next(r for r in results if r['id'] == id_rel)
    res_crit = next(r for r in results if r['id'] == id_crit)
    
    # Check Incidental
    assert "Importance: Incidental" in res_inc['content']
    assert "MEMORY ID: guid-incidental" in res_inc['content']
    assert "Timestamp: 2024-04-23" in res_inc['content']
    assert "Location: N/A" in res_inc['content']
    
    # Check Relevant
    assert "Importance: Relevant" in res_rel['content']
    assert "Location: San Francisco" in res_rel['content']
    assert "Category: USER_PREFERENCE" in res_rel['content']
    
    # Check Critical
    assert "Importance: Critical" in res_crit['content']
    assert "Category: TASK" in res_crit['content']
    
    # Check Structure
    expected_start = "### MEMORY ID: guid-critical"
    assert res_crit['content'].startswith(expected_start)
    assert "- Context:" in res_crit['content']
    assert "  This is a critical instruction!" in res_crit['content']

def test_importance_generation(enrichment_service):
    """Verify that calculate_importance returns values in the 0-10 range."""
    # Test high importance
    resp_crit = enrichment_service.calculate_importance("This is a critical security vulnerability that must be fixed immediately!")
    assert 7.0 <= resp_crit['score'] <= 10.0
    
    # Test low importance
    resp_trivial = enrichment_service.calculate_importance("The weather is nice today.")
    assert 0.0 <= resp_trivial['score'] <= 5.0
