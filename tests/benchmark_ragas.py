import os
import sys
import json
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock

# --- Mock Hermes Core (Must be first for standalone) ---
def setup_standalone_mocks():
    if "pytest" in sys.modules:
        return

    import types
    agent = types.ModuleType("agent")
    agent.memory_provider = types.ModuleType("memory_provider")
    class BaseMemoryProvider:
        def __init__(self, *args, **kwargs): pass
        def initialize(self, *args, **kwargs): pass
        def shutdown(self, *args, **kwargs): pass
    agent.memory_provider.MemoryProvider = BaseMemoryProvider
    sys.modules["agent"] = agent
    sys.modules["agent.memory_provider"] = agent.memory_provider
    
    tools = MagicMock()
    sys.modules["tools"] = tools
    sys.modules["tools.registry"] = tools.registry
    
    # We allow real hermes_constants to provide real home for config access
    try:
        import hermes_constants
    except ImportError:
        hc = MagicMock()
        test_dir = Path(__file__).parent.resolve() / "test_data"
        # test_dir.mkdir(parents=True, exist_ok=True)
        hc.get_hermes_home = lambda: test_dir

        # Fallback to standard location if hermes_constants not in path
        # from pathlib import Path
        # hc.get_hermes_home = lambda: Path.home() / ".hermes"
        sys.modules["hermes_constants"] = hc

if __name__ == "__main__":
    setup_standalone_mocks()
    
    # Configure logging for console visibility
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Ensure reveriecore logs propagate to our console handler
    logging.getLogger("reveriecore").propagate = True
# ---------------------------------------

# Add parent of reveriecore to path
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent.resolve()
parent_dir = project_root.parent.resolve()

if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from reveriecore.provider import ReverieMemoryProvider
from reveriecore.enrichment import ConfigLoader

def check_ragas():
    try:
        from ragas import evaluate
        return True
    except ImportError:
        return False
def run_benchmark():
    if not check_ragas():
        print("Ragas not installed, skipping full benchmark.")
        return

    from ragas import evaluate
    from ragas.metrics import Faithfulness, ContextPrecision
    from ragas.run_config import RunConfig
    from langchain_openai import ChatOpenAI
    from datasets import Dataset

    # 1. Load Data
    data_path = project_root / "tests" / "benchmark_data.json"
    if not data_path.exists():
        print(f"Benchmark data not found at {data_path}")
        return

    with open(data_path, "r") as f:
        samples = json.load(f)
    
    print(f"Loaded {len(samples)} benchmark samples.")

    # 2. Setup Provider & LLM
    provider = ReverieMemoryProvider()
    provider.initialize(session_id="benchmark_run")
    
    # Use your base_url/model logic (hardcoded for stability)
    base_url = "http://172.22.0.1:8080/v1"
    model_name = "gemma4-e4b"
    api_key = "sk-reverie-internal"

    llm = ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
        temperature=0
    )

    results = []
    # Loop through samples
    for i, item in enumerate(samples):
        question = item["question"]
        print(f"[{i+1}/{len(samples)}] Querying: {question[:50]}...")
        
        query_vec = provider._enrichment.generate_embedding(question)
        # retrieved = provider._retriever.search(query_vec, query_text=question, limit=3)
        retrieved = provider._retriever.search(
            query_vec, 
            query_text=question, 
            limit=3,
            strategy="balanced"
        )
        # Inside run_benchmark, after retreiver.search: see if it is actually finding the right documents or just pulling random "active" nodes
        print(f"DEBUG: Retrieved {len(retrieved)} nodes for '{question[:30]}'")
        for r in retrieved:
            print(f"DEBUG: Node ID {r.get('id')} Score: {r.get('score', 0):.4f} {r.get('content', '')[:200]}")
        
        contexts = [r["content"] for r in retrieved if r.get("content")]
        if not contexts:
            continue
            
        context_str = "\n".join(contexts)
        prompt = f"Answer validly based on context.\nContext:\n{context_str}\nQuestion: {question}"
        
        response = llm.invoke(prompt)
        results.append({
            "question": question,
            "answer": response.content,
            "contexts": contexts,
            "ground_truth": item["ground_truth"]
        })

    # 3. Evaluate with throttled concurrency
    df = pd.DataFrame(results)
    dataset = Dataset.from_pandas(df)
    
    # CRITICAL: RunConfig with max_workers=1 forces sequential evaluation.
    # This prevents 'Too many requests' and 'Timeout' on your local Gemma instance.
    run_config = RunConfig(max_workers=1)
    
    result = evaluate(
        dataset,
        metrics=[Faithfulness(), ContextPrecision()],
        llm=llm,
        run_config=run_config
    )
    
    print("\n--- Benchmark Results ---")
    print(result)
    provider.shutdown()

if __name__ == "__main__":
    run_benchmark()
