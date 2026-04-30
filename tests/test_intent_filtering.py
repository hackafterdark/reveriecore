import pytest
from unittest.mock import MagicMock
from reveriecore.retrieval import RetrievalContext, IntentClassifierDiscovery, DiscoveryConfig, IntentClassifierConfig
from reveriecore.retrieval_base import RetrievalHandler

@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.enrichment = MagicMock()
    return retriever

def test_intent_classifier_causal(mock_retriever):
    # Setup
    config = IntentClassifierConfig()
    handler = IntentClassifierDiscovery(config=config)
    context = RetrievalContext(query_text="Why did the build fail?", query_vector=[0.1]*384, limit=5, token_budget=1000, config={})
    
    # Mock mDeBERTa response (Updated labels)
    mock_retriever.enrichment._zero_shot_classify.return_value = {
        "troubleshooting and root cause analysis": 0.8,
        "step-by-step instructions and prerequisites": 0.1,
        "general definition and conceptual mapping": 0.1
    }
    
    # Process
    handler.process(context, mock_retriever)
    
    # Assertions
    assert context.metadata["intent"] == "troubleshooting and root cause analysis"
    assert context.metadata["intent_confidence"] == 0.8
    assert "CAUSES" in context.metadata["allowed_edges"]
    assert "DEPENDS_ON" in context.metadata["allowed_edges"]

def test_intent_classifier_low_confidence(mock_retriever):
    # Setup
    config = IntentClassifierConfig()
    handler = IntentClassifierDiscovery(config=config)
    context = RetrievalContext(query_text="Hello world", query_vector=[0.1]*384, limit=5, token_budget=1000, config={})
    
    # Mock mDeBERTa response (low confidence)
    mock_retriever.enrichment._zero_shot_classify.return_value = {
        "troubleshooting and root cause analysis": 0.2,
        "step-by-step instructions and prerequisites": 0.2,
        "general definition and conceptual mapping": 0.6
    }
    
    # Process
    handler.process(context, mock_retriever)
    
    # Assertions
    assert context.metadata["intent"] == "general definition and conceptual mapping"
    # Should set allowed_edges if confidence > 0.25 (0.6 > 0.25)
    assert "allowed_edges" in context.metadata

def test_intent_classifier_very_low_confidence(mock_retriever):
    # Setup
    config = IntentClassifierConfig()
    handler = IntentClassifierDiscovery(config=config)
    context = RetrievalContext(query_text="Hello world", query_vector=[0.1]*384, limit=5, token_budget=1000, config={})
    
    # Mock mDeBERTa response (very low confidence)
    mock_retriever.enrichment._zero_shot_classify.return_value = {
        "troubleshooting and root cause analysis": 0.21,
        "step-by-step instructions and prerequisites": 0.21,
        "general definition and conceptual mapping": 0.21
    }
    
    # Process
    handler.process(context, mock_retriever)
    
    # Should NOT set allowed_edges if confidence < 0.25
    assert "allowed_edges" not in context.metadata
