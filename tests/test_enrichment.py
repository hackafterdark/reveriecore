import pytest
from unittest.mock import MagicMock
from reveriecore.schemas import MemoryType
from reveriecore.enrichment import EnrichmentService

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
    
    assert score_frustrated["score"] > score_normal["score"]
    # Frustrated text should get sentiment boost (+1.5)
    assert score_frustrated["score"] > 1.5 

def test_semantic_profiling(enrichment_service):
    """Verify summarization for long text."""
    long_text = "The user spent several hours trying to debug the authentication module. " * 10
    profile = enrichment_service.generate_semantic_profile(long_text)
    
    assert len(profile) < len(long_text)
    assert len(profile.split()) > 0

@pytest.fixture
def mock_enrichment():
    service = EnrichmentService()
    service.llm_client = MagicMock()
    return service

def test_heuristic_importance(mock_enrichment):
    """Test the fast path heuristics."""
    text = "CRITICAL ERROR: The server has crashed."
    result = mock_enrichment.calculate_importance(text)
    assert result["score"] == 9.5
    # Heuristics should not trigger LLM call
    mock_enrichment.llm_client.call.assert_not_called()

def test_soul_aware_importance(mock_enrichment):
    """Test importance calculation using the agent's soul."""
    text = "I am writing a new python script."
    soul_prompt = "You are a software engineer."
    mock_enrichment.set_soul(soul_prompt)
    
    # Mock LLM response
    mock_enrichment.llm_client.check_connectivity.return_value = True
    mock_enrichment.llm_client.call.return_value = {"importance": 8.5}
    
    result = mock_enrichment.calculate_importance(text)
    
    assert result["score"] == 8.5
    mock_enrichment.llm_client.call.assert_called_once()
    # Verify prompt contains the soul
    args, kwargs = mock_enrichment.llm_client.call.call_args
    messages = args[0]
    user_content = messages[1]["content"]
    assert soul_prompt in user_content

def test_fallback_to_bart_when_no_soul(mock_enrichment):
    """Verify fallback to BART when soul is missing."""
    text = "Just some casual conversation."
    mock_enrichment.soul_prompt = None
    
    # We can't easily test the BART call without mocking _zero_shot_classify
    # but we can check that LLM call was NOT made
    result = mock_enrichment.calculate_importance(text)
    assert "score" in result
    mock_enrichment.llm_client.call.assert_not_called()

def test_dynamic_soul_update(mock_enrichment):
    """Test that setting a new soul updates the prompt used."""
    mock_enrichment.llm_client.check_connectivity.return_value = True
    mock_enrichment.llm_client.call.return_value = {"importance": 5.0}
    
    mock_enrichment.set_soul("Chef persona")
    mock_enrichment.calculate_importance("Baking bread")
    
    args, _ = mock_enrichment.llm_client.call.call_args
    assert "Chef persona" in args[0][1]["content"]
    
    mock_enrichment.set_soul("Pilot persona")
    mock_enrichment.calculate_importance("Flying a plane")
    
    args, _ = mock_enrichment.llm_client.call.call_args
    assert "Pilot persona" in args[0][1]["content"]
