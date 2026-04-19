import pytest
from reveriecore.schemas import MemoryType

def test_embedding_generation(enrichment_service):
    """Verify that embeddings have the correct dimensions."""
    vec = enrichment_service.generate_embedding("Hello world")
    assert isinstance(vec, list)
    assert len(vec) == 384
    assert isinstance(vec[0], float)

def test_classify_runtime_error(enrichment_service):
    """Verify classification of errors."""
    text = "Traceback (most recent call last):\nFile 'app.py', line 10, in <module>\nRuntimeError: Something went wrong"
    mem_type = enrichment_service.classify_type(text)
    assert mem_type == MemoryType.RUNTIME_ERROR

def test_importance_boost_for_frustrated_user(enrichment_service):
    """Verify sentiment-based importance boost."""
    # Frustrated text should have higher weight
    normal_text = "I am working on the database."
    frustrated_text = "I HATE THIS. NOTHING IS WORKING AND THE DATABASE IS BROKEN AGAIN EXCEPTION CRASH."
    
    score_normal = enrichment_service.calculate_importance(normal_text)
    score_frustrated = enrichment_service.calculate_importance(frustrated_text)
    
    assert score_frustrated > score_normal
    # Frustrated text should get sentiment boost (+1.5) and keyword boost (+0.5)
    assert score_frustrated >= 2.5 

def test_semantic_profiling(enrichment_service):
    """Verify summarization for long text."""
    long_text = "The user spent several hours trying to debug the authentication module. " * 10
    profile = enrichment_service.generate_semantic_profile(long_text)
    
    assert len(profile) < len(long_text)
    assert len(profile.split()) > 0
