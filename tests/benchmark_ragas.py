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
    
    hc = MagicMock()
    # SAFE TEMP DIR
    temp_dir = Path("/tmp/reverie_benchmark_run")
    temp_dir.mkdir(parents=True, exist_ok=True)
    hc.get_hermes_home = lambda: temp_dir
    sys.modules["hermes_constants"] = hc

if __name__ == "__main__":
    setup_standalone_mocks()
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
    
    config = ConfigLoader.load_config()
    model_cfg = config.get("model", {})
    providers = config.get("providers", [])
    
    base_url = model_cfg.get("base_url")
    if not base_url and providers:
        base_url = providers[0].get("base_url")
    if not base_url:
        base_url = "http://localhost:11434/v1"
        
    model_name = model_cfg.get("default") or model_cfg.get("model") or "gemma2:2b"
    api_key = model_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY") or "sk-reverie-internal"

    print(f"Using LLM: {model_name} at {base_url}")
    
    llm = ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
        temperature=0
    )

    results = []
    for i, item in enumerate(samples[:5]): # Reduced for quick verification
        question = item["question"]
        print(f"[{i+1}/5] Querying: {question[:50]}...")
        
        query_vec = provider._enrichment.generate_embedding(question)
        retrieved = provider._retriever.search(
            query_vec, 
            query_text=question,
            limit=3
        )
        
        contexts = [r["content"] for r in retrieved]
        context_str = "\n".join(contexts)
        
        prompt = f"Answer validly based on context.\nContext:\n{context_str}\nQuestion: {question}"
        response = llm.invoke(prompt)
        answer = response.content
        
        results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"]
        })

    df = pd.DataFrame(results)
    dataset = Dataset.from_pandas(df)
    
    run_config = RunConfig(max_workers=2)
    result = evaluate(
        dataset,
        metrics=[Faithfulness(), ContextPrecision()],
        llm=llm,
        run_config=run_config
    )
    
    print(result)
    provider.shutdown()

if __name__ == "__main__":
    run_benchmark()
