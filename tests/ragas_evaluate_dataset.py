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
    
    try:
        import hermes_constants
    except ImportError:
        hc = MagicMock()
        test_dir = Path(__file__).parent.resolve() / "test_data"
        hc.get_hermes_home = lambda: test_dir
        sys.modules["hermes_constants"] = hc

if __name__ == "__main__":
    setup_standalone_mocks()
    
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger("reveriecore").propagate = True
# ---------------------------------------

current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent.resolve()
parent_dir = project_root.parent.resolve()

if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from reveriecore.provider import ReverieMemoryProvider
from reveriecore.enrichment import ConfigLoader
from datasets import load_dataset

def check_ragas():
    try:
        from ragas import evaluate
        return True
    except ImportError:
        return False

def run_evaluation():
    if not check_ragas():
        print("Ragas not installed, skipping evaluation.")
        return

    from ragas import evaluate
    from ragas.metrics import Faithfulness, ContextPrecision
    from ragas.run_config import RunConfig
    from langchain_openai import ChatOpenAI
    from datasets import Dataset

    # 1. Load Data
    dataset_path = os.getenv("REVERIE_DATASET_PATH", "vibrantlabsai/amnesty_qa")
    dataset_name = os.getenv("REVERIE_DATASET_NAME", "english_v3")
    dataset_split = os.getenv("REVERIE_DATASET_SPLIT", "eval")
    session_id = os.getenv("REVERIE_SESSION_ID", "amnesty_benchmark")

    print(f"Loading dataset '{dataset_path}' ({dataset_name}) split '{dataset_split}' from Hugging Face...")
    raw_ds = load_dataset(dataset_path, dataset_name)
    eval_split = raw_ds[dataset_split]
    samples = list(eval_split)
    
    print(f"Loaded {len(samples)} benchmark samples.")

    # 2. Setup Provider & LLM
    provider = ReverieMemoryProvider()
    provider.initialize(session_id=session_id)
    
    base_url = os.getenv("REVERIE_LLM_BASE_URL", "http://172.22.0.1:8080/v1")
    model_name = os.getenv("REVERIE_LLM_MODEL", "gemma4-e4b")
    api_key = os.getenv("REVERIE_LLM_API_KEY", "sk-reverie-internal")

    llm = ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
        temperature=0
    )

    results = []
    # Loop through samples
    for i, item in enumerate(samples):
        question = item.get("user_input", "")
        print(f"[{i+1}/{len(samples)}] Querying: {question[:50]}...")
        
        query_vec = provider._enrichment.generate_embedding(question)
        retrieved = provider._retriever.search(
            query_vec, 
            query_text=question, 
            limit=5,
            strategy="balanced",
            include_ids=False
        )

        for r in retrieved:
            print(f"DEBUG: Node ID {r.get('id')} Score: {r.get('score', 0):.4f} {r.get('content', '')[:200]}")
        
        contexts = [r["content"] for r in retrieved if r.get("content")]
        if not contexts:
            # Ragas faithfulness needs at least one empty context if none found
            contexts = [""]
            
        context_str = "\n".join(contexts)
        prompt = f"Answer validly based on context.\nContext:\n{context_str}\nQuestion: {question}"
        
        response = llm.invoke(prompt)
        results.append({
            "question": question,
            "answer": response.content,
            "contexts": contexts,
            "ground_truth": item.get("reference", item.get("ground_truth", item.get("answer", ""))) 
        })

    # 3. Evaluate with throttled concurrency
    df = pd.DataFrame(results)
    dataset = Dataset.from_pandas(df)
    
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
    run_evaluation()
