import logging
from reveriecore.provider import ReverieMemoryProvider
from reveriecore.retrieval import RetrievalConfig

logging.basicConfig(level=logging.INFO)

def test_integration_pruning():
    provider = ReverieMemoryProvider()
    provider.initialize(session_id="test_integration")
    
    # 1. Verify pruning is in the pipeline
    ranking_pipeline = [h.__class__.__name__ for h in provider._retriever.ranking_pipeline]
    print(f"Ranking Pipeline: {ranking_pipeline}")
    assert "PruningHandler" in ranking_pipeline
    
    # 2. Run a search (passive mode)
    query = "test query"
    query_vec = [0.1] * 384
    results = provider._retriever.search(query_vec, query_text=query)
    
    print(f"Search results count: {len(results)}")
    # Pruning metrics should be in the logs or we can check the context if we had access to it.
    # But the fact it didn't crash and the pipeline has the handler is good.
    
    provider.shutdown()

if __name__ == "__main__":
    test_integration_pruning()
