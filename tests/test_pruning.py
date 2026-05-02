import pytest
from reveriecore.pruning import PruningEngine, PruningHandler
from reveriecore.schemas import RetrievalContext

def test_pruning_engine_top_n():
    candidates = {
        1: {"score": 0.9},
        2: {"score": 0.8},
        3: {"score": 0.7},
        4: {"score": 0.6}
    }
    # Case: Limit to top 2
    pruned = PruningEngine.prune(candidates, top_n=2, relative_threshold=0.0, min_absolute_score=0.0)
    assert len(pruned) == 2
    assert 1 in pruned
    assert 2 in pruned
    assert 3 not in pruned

def test_pruning_engine_relative_threshold():
    candidates = {
        1: {"score": 1.0},
        2: {"score": 0.4},  # Should be dropped if threshold > 0.4
        3: {"score": 0.9}
    }
    # Case: 50% relative threshold
    pruned = PruningEngine.prune(candidates, top_n=5, relative_threshold=0.5, min_absolute_score=0.0)
    assert len(pruned) == 2
    assert 1 in pruned
    assert 3 in pruned
    assert 2 not in pruned

def test_pruning_engine_min_absolute_score():
    candidates = {
        1: {"score": 0.25}, # Should be dropped if min_abs = 0.3
        2: {"score": 0.35}
    }
    pruned = PruningEngine.prune(candidates, top_n=5, relative_threshold=0.0, min_absolute_score=0.3)
    assert len(pruned) == 1
    assert 2 in pruned
    assert 1 not in pruned

def test_pruning_handler_integration():
    class MockConfig:
        top_n = 2
        relative_threshold = 0.5
        min_absolute_score = 0.3

    handler = PruningHandler(config=MockConfig())
    context = RetrievalContext("test", [0.1]*384, 5, 1000)
    context.candidates = {
        1: {"id": 1, "score": 0.9},
        2: {"id": 2, "score": 0.2}, # Low score
        3: {"id": 3, "score": 0.8}
    }
    
    handler.process(context, None)
    
    assert len(context.candidates) == 2
    assert 1 in context.candidates
    assert 3 in context.candidates
    assert 2 not in context.candidates
    assert context.metrics["pruning"]["pruned"] == 1
    assert context.metrics["pruning"]["remaining"] == 2
