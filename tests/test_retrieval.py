import pytest
from reveriecore.retrieval import Retriever

def test_hybrid_search_ranking(db_manager, enrichment_service):
    """Verify that importance score influences ranking."""
    retriever = Retriever(db_manager)
    cursor = db_manager.get_cursor()
    
    # Clear tables
    cursor.execute("DELETE FROM memories")
    cursor.execute("DELETE FROM memories_vec")
    
    # Insert two memories about "Python"
    # A: Lower importance
    # B: Higher importance
    
    content_a = "Python is a programming language."
    vec_a = enrichment_service.generate_embedding(content_a)
    cursor.execute("INSERT INTO memories (content_full, importance_score) VALUES (?, ?)", (content_a, 1.0))
    id_a = cursor.lastrowid
    cursor.execute("INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)", (id_a, str(vec_a)))
    
    content_b = "Python is used for data science and it is VERY IMPORTANT."
    vec_b = enrichment_service.generate_embedding(content_b)
    cursor.execute("INSERT INTO memories (content_full, importance_score) VALUES (?, ?)", (content_b, 5.0))
    id_b = cursor.lastrowid
    cursor.execute("INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)", (id_b, str(vec_b)))
    
    db_manager.commit()
    
    # Search for "Python programming"
    query_vec = enrichment_service.generate_embedding("Python programming")
    
    # 1. Test with similarity dominance
    results_sim = retriever.search(query_vec, limit=2, similarity_weight=1.0, importance_weight=0.0)
    # Memory A might be more similar purely on text
    
    # 2. Test with importance influence
    results_hybrid = retriever.search(query_vec, limit=2, similarity_weight=0.5, importance_weight=0.5)
    
    assert len(results_hybrid) == 2
    # Memory B should have a high score due to importance=5.0
    assert any(r['id'] == id_b for r in results_hybrid)
