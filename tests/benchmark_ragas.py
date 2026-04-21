# --- Mock Hermes Core (Must be first) ---
import sys
from pathlib import Path
from unittest.mock import MagicMock
agent = MagicMock()
class BaseMemoryProvider:
    def __init__(self, *args, **kwargs): pass
    def initialize(self, *args, **kwargs): pass
    def shutdown(self, *args, **kwargs): pass
agent.memory_provider.MemoryProvider = BaseMemoryProvider
sys.modules["agent"] = agent
sys.modules["agent.memory_provider"] = agent.memory_provider
tools = MagicMock(); sys.modules["tools"] = tools; sys.modules["tools.registry"] = tools.registry
hc = MagicMock(); hc.get_hermes_home = lambda: Path.home() / ".hermes"
sys.modules["hermes_constants"] = hc
# ---------------------------------------

import os
import json
import pandas as pd
# Add parent of reveriecore to path so it can be imported as a package
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent.resolve()
parent_dir = project_root.parent.resolve()

if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from reveriecore.provider import ReverieMemoryProvider
from reveriecore.enrichment import ConfigLoader

# RAGAS Imports
from ragas import evaluate
from ragas.metrics import Faithfulness, ContextPrecision
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI
from datasets import Dataset

def run_benchmark():
    # 1. Load Data
    data_path = project_root / "tests" / "benchmark_data.json"
    with open(data_path, "r") as f:
        samples = json.load(f)
    
    print(f"Loaded {len(samples)} benchmark samples.")

    # 2. Setup Provider & LLM
    provider = ReverieMemoryProvider()
    provider.initialize(session_id="benchmark_run")
    
    # Load Hermes Config for LLM
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

    # 3. Execute RAG Cycle
    results = []
    for i, item in enumerate(samples[:20]): # 20 questions
        question = item["question"]
        print(f"[{i+1}/20] Querying: {question[:50]}...")
        
        # Retrieval
        # We need to generate an embedding for the query first
        query_vec = provider._enrichment.generate_embedding(question)
        retrieved = provider._retriever.search(
            query_vec, 
            query_text=question,
            limit=3,
            token_budget=2000
        )
        
        contexts = [r["content"] for r in retrieved]
        context_str = "\n".join(contexts)
        
        # Generation
        prompt = f"Answer the following question based ONLY on the provided context.\n\nContext:\n{context_str}\n\nQuestion: {question}\n\nAnswer:"
        response = llm.invoke(prompt)
        answer = response.content
        
        results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"]
        })

    # 4. Convert to RAGAS Dataset
    df = pd.DataFrame(results)
    dataset = Dataset.from_pandas(df)
    
    # 5. Evaluate
    print("\nRunning RAGAS Evaluation... (This may take a few minutes)")
    # Throttle concurrency to prevent overwhelming the local LLM server
    run_config = RunConfig(max_workers=2)
    
    result = evaluate(
        dataset,
        metrics=[Faithfulness(), ContextPrecision()],
        llm=llm,
        run_config=run_config
    )
    
    # 6. Report
    print("\n" + "="*50)
    print("REVERIE CORE RETRIEVAL BENCHMARK RESULTS")
    print("="*50)
    print(result)
    print("="*50)
    
    # Save results
    output_path = project_root / "tests" / "benchmark_results.json"
    # Access the scores dictionary from the EvaluationResult object
    result_dict = result.scores
    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"Results saved to {output_path}")

    provider.shutdown()

if __name__ == "__main__":
    run_benchmark()
