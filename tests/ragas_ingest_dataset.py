import os
import sys
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

from datasets import load_dataset
from reveriecore.provider import ReverieMemoryProvider

def ingest_huggingface_dataset():
    # 1. Load from Hugging Face
    dataset_path = os.getenv("REVERIE_DATASET_PATH", "vibrantlabsai/amnesty_qa")
    dataset_name = os.getenv("REVERIE_DATASET_NAME", "english_v3")
    dataset_split = os.getenv("REVERIE_DATASET_SPLIT", "eval")
    session_id = os.getenv("REVERIE_SESSION_ID", "amnesty_benchmark")

    print(f"Loading dataset '{dataset_path}' ({dataset_name}) split '{dataset_split}' from Hugging Face...")
    raw_ds = load_dataset(dataset_path, dataset_name)
    eval_split = raw_ds[dataset_split] # Usually the split you want
    
    # 2. Initialize your provider
    provider = ReverieMemoryProvider()
    provider.initialize(session_id=session_id)
    
    # 3. Ingest into your ReverieCore memory
    print(f"Ingesting {len(eval_split)} samples...")
    for i, sample in enumerate(eval_split):
        if i % 10 == 0:
            print(f"[{i}/{len(eval_split)}] Ingesting...")
            
        context_data = sample.get('retrieved_contexts', sample.get('context', sample.get('contexts', '')))
        if isinstance(context_data, list):
            context_data = "\n".join(context_data)
            
        # We treat each sample's question/context pair as a document to be learned
        provider.sync_turn(
            user_content=f"External Context: {sample.get('user_input', '')}",
            assistant_content=f"{context_data}",
            session_id=session_id
        )
    
    provider.shutdown()
    print("Amnesty dataset ingested into ReverieCore.")

if __name__ == "__main__":
    ingest_huggingface_dataset()
