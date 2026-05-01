import pytest
import sys
import os
from unittest.mock import MagicMock, patch

# Mock llama_cpp module before importing handler
mock_llama = MagicMock()
sys.modules["llama_cpp"] = mock_llama

from reveriecore.retrieval_base import RetrievalContext
from reveriecore.rewriting import QueryRewriterHandler

@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.config = {
        "settings": {
            "rewriter": {
                "enabled": True,
                "model_path": "mock_path.gguf",
                "threads": 2,
                "max_words": 10
            }
        }
    }
    retriever.enrichment = MagicMock()
    retriever.enrichment.generate_embedding.return_value = [0.1, 0.2, 0.3]
    return retriever

def test_rewriter_skips_long_query(mock_retriever):
    handler = QueryRewriterHandler()
    context = RetrievalContext(
        query_text="This is a very long query with more than ten words here",
        query_vector=[0.0, 0.0, 0.0],
        limit=5,
        token_budget=1000
    )
    
    mock_llama.Llama.reset_mock()
    mock_span = MagicMock(name="mock_span")
    
    # Patch the tracer in the rewriting module
    # We patch it directly on the module object to ensure it's hit
    import reveriecore.rewriting
    with patch("opentelemetry.trace.get_tracer") as mock_get_tracer, \
         patch("os.path.exists", return_value=True):
        
        mock_tracer = MagicMock()
        mock_get_tracer.return_value = mock_tracer
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        handler.process(context, mock_retriever)
        
        # Verify skip reason
        mock_span.set_attribute.assert_any_call("retrieval.skip_reason", "query_already_detailed")
        mock_span.set_attribute.assert_any_call("retrieval.word_count", 12)

def test_rewriter_success(mock_retriever):
    handler = QueryRewriterHandler()
    context = RetrievalContext(
        query_text="fix bug",
        query_vector=[0.0, 0.0, 0.0],
        limit=5,
        token_budget=1000
    )
    
    mock_llama.Llama.reset_mock()
    mock_llama_inst = MagicMock(name="mock_llama_inst")
    mock_llama_inst.return_value = {
        "choices": [{"text": " expand bug fix details"}]
    }
    mock_llama.Llama.return_value = mock_llama_inst
    
    mock_span = MagicMock(name="mock_span")
    import reveriecore.rewriting
    with patch("opentelemetry.trace.get_tracer") as mock_get_tracer, \
         patch("os.path.exists", return_value=True):
        
        mock_tracer = MagicMock()
        mock_get_tracer.return_value = mock_tracer
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        handler.process(context, mock_retriever)
        
        assert context.query_text == "expand bug fix details"
        mock_span.set_attribute.assert_any_call("retrieval.word_count", 2)
        mock_span.set_attribute.assert_any_call("retrieval.is_rewritten", True)

def test_rewriter_model_file_missing(mock_retriever):
    handler = QueryRewriterHandler()
    context = RetrievalContext(
        query_text="fix bug",
        query_vector=[0.0, 0.0, 0.0],
        limit=5,
        token_budget=1000
    )
    
    mock_span = MagicMock(name="mock_span")
    import reveriecore.rewriting
    with patch("opentelemetry.trace.get_tracer") as mock_get_tracer, \
         patch("os.path.exists", return_value=False):
        
        mock_tracer = MagicMock()
        mock_get_tracer.return_value = mock_tracer
        mock_tracer = MagicMock()
        mock_get_tracer.return_value = mock_tracer
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        handler.process(context, mock_retriever)
        
        assert handler.generator is None
        mock_span.set_attribute.assert_any_call("retrieval.skip_reason", "model_file_missing")

def test_rewriter_load_failed(mock_retriever):
    handler = QueryRewriterHandler()
    context = RetrievalContext(
        query_text="fix bug",
        query_vector=[0.0, 0.0, 0.0],
        limit=5,
        token_budget=1000
    )
    
    mock_llama.Llama.side_effect = Exception("Load error")
    mock_span = MagicMock(name="mock_span")
    import reveriecore.rewriting
    with patch("opentelemetry.trace.get_tracer") as mock_get_tracer, \
         patch("os.path.exists", return_value=True):
        
        mock_tracer = MagicMock()
        mock_get_tracer.return_value = mock_tracer
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        handler.process(context, mock_retriever)
        
        assert handler.generator is None
        mock_span.set_attribute.assert_any_call("retrieval.skip_reason", "load_failed")
    mock_llama.Llama.side_effect = None

def test_rewriter_disabled_in_config(mock_retriever):
    handler = QueryRewriterHandler()
    mock_retriever.config["settings"]["rewriter"]["enabled"] = False
    context = RetrievalContext(
        query_text="fix bug",
        query_vector=[0.0, 0.0, 0.0],
        limit=5,
        token_budget=1000
    )
    
    mock_span = MagicMock(name="mock_span")
    import reveriecore.rewriting
    with patch("opentelemetry.trace.get_tracer") as mock_get_tracer:
        mock_tracer = MagicMock()
        mock_get_tracer.return_value = mock_tracer
        mock_tracer = MagicMock()
        mock_get_tracer.return_value = mock_tracer
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        handler.process(context, mock_retriever)
        
        assert handler.generator is None
        mock_span.set_attribute.assert_any_call("retrieval.word_count", 2)
