import pytest
from unittest.mock import MagicMock
from reveriecore.reranking import RerankerHandler
from reveriecore.retrieval_base import RetrievalContext

def test_reranker_handler_logic(mocker):
    # 1. Setup handler and context
    handler = RerankerHandler()
    
    # Mock context with candidates
    context = RetrievalContext(
        query_text="how to use sqlite-vec",
        query_vector=[0.1]*384,
        limit=5,
        token_budget=1000
    )
    
    context.candidates = {
        1: {"id": 1, "content_full": "sqlite-vec is a vector search extension for SQLite.", "score": 0.5},
        2: {"id": 2, "content_full": "The weather is nice today.", "score": 0.4}
    }
    
    # 2. Mock flashrank.Ranker
    # We patch the property 'ranker' on the handler instance to avoid lazy loading issues
    mock_ranker = MagicMock()
    mock_ranker.rerank.return_value = [
        {"id": 1, "score": 0.99},
        {"id": 2, "score": 0.01}
    ]
    
    mocker.patch.object(RerankerHandler, 'ranker', new_callable=mocker.PropertyMock, return_value=mock_ranker)
    
    # 3. Process
    handler.process(context, MagicMock())
    
    # 4. Verify
    assert context.candidates[1]["score"] == 1.99
    assert context.candidates[1]["source"] == "reranked"
    assert context.candidates[2]["score"] == 1.01
    assert context.candidates[2]["source"] == "reranked"
    
    # Verify rerank was called with correct format
    mock_ranker.rerank.assert_called_once()
    args, _ = mock_ranker.rerank.call_args
    req = args[0]
    assert req.query == "how to use sqlite-vec"
    assert len(req.passages) == 2
    assert req.passages[0]["id"] == 1

def test_reranker_handler_missing_module(mocker, caplog):
    # Test that it handles missing module silently if is_available returns False
    caplog.set_level("DEBUG")
    handler = RerankerHandler()
    
    # Mock is_available to return False
    mocker.patch.object(RerankerHandler, 'is_available', return_value=False)
    
    # Patch the logger's debug method directly
    mock_debug = mocker.patch("reveriecore.reranking.logger.debug")
    
    context = RetrievalContext("test", [0]*384, 5, 1000)
    context.candidates = {
        1: {"id": 1, "content_full": "text 1", "score": 0.5},
        2: {"id": 2, "content_full": "text 2", "score": 0.4}
    }
    
    handler.process(context, MagicMock())
    
    # Verify score didn't change
    assert context.candidates[1]["score"] == 0.5
    # Verify log (silent skip)
    mock_debug.assert_called_with("FlashRank not found. Skipping rerank.")

def test_reranker_handler_initialization_failure(mocker, caplog):
    # Test that it logs an error if is_available is True but ranker fails
    caplog.set_level("ERROR")
    handler = RerankerHandler()
    
    # Mock is_available to return True
    mocker.patch.object(RerankerHandler, 'is_available', return_value=True)
    # Mock the property to return None (simulating an initialization error)
    mocker.patch.object(RerankerHandler, 'ranker', new_callable=mocker.PropertyMock, return_value=None)
    
    context = RetrievalContext("test", [0]*384, 5, 1000)
    context.candidates = {
        1: {"id": 1, "content_full": "text 1", "score": 0.5},
        2: {"id": 2, "content_full": "text 2", "score": 0.4}
    }
    
    handler.process(context, MagicMock())
    
    # Verify score didn't change
    assert context.candidates[1]["score"] == 0.5
    # Should return early without error if ranker is None
    assert "Reranking failed" not in caplog.text
