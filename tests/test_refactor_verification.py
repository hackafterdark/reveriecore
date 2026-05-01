import pytest
from unittest.mock import MagicMock
from reveriecore.retrieval import Retriever, RetrievalContext, IntentClassifierDiscovery, IntentRanker, IntentClassifierConfig, IntentConfig
from reveriecore.database import DatabaseManager

@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.enrichment = MagicMock()
    # Mock DatabaseManager for Retriever initialization
    retriever.db = MagicMock(spec=DatabaseManager)
    return retriever

def test_activation_by_inclusion():
    """Verify that handlers only run if included in the pipeline list."""
    db_manager = MagicMock(spec=DatabaseManager)
    retriever = Retriever(db_manager)
    
    # 1. Reset pipelines with specific inclusion
    retriever.config.pipeline.discovery = ["vector"]
    retriever.discovery_pipeline = []
    retriever._setup_pipelines()
    
    handler_names = [h.__class__.__name__ for h in retriever.discovery_pipeline]
    assert "VectorDiscovery" in handler_names
    assert "IntentClassifierDiscovery" not in handler_names
    
    # 2. Add intent_classifier
    retriever.config.pipeline.discovery = ["intent_classifier", "vector"]
    retriever.discovery_pipeline = []
    retriever._setup_pipelines()
    
    handler_names = [h.__class__.__name__ for h in retriever.discovery_pipeline]
    assert "IntentClassifierDiscovery" in handler_names
    assert "VectorDiscovery" in handler_names
    
    # 3. Verify 'enabled' key is removed from config models
    intent_handler = [h for h in retriever.discovery_pipeline if h.__class__.__name__ == "IntentClassifierDiscovery"][0]
    assert not hasattr(intent_handler.config, "enabled")

def test_unified_intent_logic(mock_retriever):
    """Verify that IntentRanker uses the intent set by IntentClassifierDiscovery."""
    ic_config = IntentClassifierConfig()
    ic_handler = IntentClassifierDiscovery(config=ic_config)
    
    ir_config = IntentConfig()
    ir_handler = IntentRanker(config=ir_config)
    
    context = RetrievalContext(
        query_text="My app is crashing with a segmentation fault", 
        query_vector=[0.1]*384, 
        limit=5, 
        token_budget=1000, 
        config={}
    )
    
    # Mock model finding descriptive troubleshooting intent
    mock_retriever.enrichment._zero_shot_classify.return_value = {
        "troubleshooting and root cause analysis": 0.8,
        "step-by-step instructions and prerequisites": 0.1,
        "general definition and conceptual mapping": 0.1
    }
    
    # Run Discovery then Ranking
    ic_handler.process(context, mock_retriever)
    ir_handler.process(context, mock_retriever)
    
    assert context.metadata["intent"] == "troubleshooting and root cause analysis"
    assert "CAUSES" in context.metadata["allowed_edges"]
    assert context.intent == "Fact-Seeking"
    assert context.weights["similarity"] == 0.7

def test_intent_threshold_lowered(mock_retriever):
    """Verify edge filter applies at lower confidence (0.25)."""
    ic_config = IntentClassifierConfig()
    handler = IntentClassifierDiscovery(config=ic_config)
    context = RetrievalContext(query_text="help", query_vector=[0.1]*384, limit=5, token_budget=1000, config={})
    
    # Confidence 0.3 should now pass (previous threshold was 0.4)
    mock_retriever.enrichment._zero_shot_classify.return_value = {
        "troubleshooting and root cause analysis": 0.3,
        "general definition and conceptual mapping": 0.65,
        "step-by-step instructions and prerequisites": 0.05
    }
    
    handler.process(context, mock_retriever)
    assert "allowed_edges" in context.metadata
    assert "IS_A" in context.metadata["allowed_edges"]
