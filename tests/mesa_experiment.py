import sys
import os
import time
import json
import logging
import threading
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

# --- Mock Hermes Core (Must be first) ---
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
    # USE A SAFE TEMP DIR, NOT ~/.hermes
    temp_dir = Path("/tmp/reverie_mesa_experiment")
    temp_dir.mkdir(parents=True, exist_ok=True)
    hc.get_hermes_home = lambda: temp_dir
    sys.modules["hermes_constants"] = hc

if __name__ == "__main__":
    setup_standalone_mocks()
# ---------------------------------------

# Add parent of reveriecore to path so it can be imported as a package
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent.resolve()
parent_dir = project_root.parent.resolve()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from reveriecore.provider import ReverieMemoryProvider
from reveriecore.enrichment import ConfigLoader
from reveriecore.database import DatabaseManager
from reveriecore.pruning import MesaService

# RAGAS Imports
def check_ragas():
    try:
        from ragas import evaluate
        return True
    except ImportError:
        return False

# --- Experiment Constants ---
EXP_DB = "tests/mesa_experiment.db"
# ... (rest of the constants remain unchanged)
TARGET_QUESTIONS = [
    {
        "question": "How does ReverieCore translate a user's natural language query into a vector representation?",
        "ground_truth": "The raw query is passed to the SentenceTransformers model (specifically all-MiniLM-L6-v2), which converts it into a 384-dimension vector."
    },
    {
        "question": "What database extension is used for performing nearest-neighbor semantic searches?",
        "ground_truth": "ReverieCore uses the sqlite-vec extension for performing KNN (K-Nearest Neighbor) similarity search within SQLite."
    },
    {
        "question": "Explain the concept of 'Namespace Isolation' in the retrieval process.",
        "ground_truth": "Namespace Isolation ensure that an agent only retrieves memories belonging to its current profile (owner_id) or global public facts (privacy='PUBLIC'), preventing data leakage between agent profiles."
    },
    {
        "question": "What happens when the graph retrieval results are fewer than 3?",
        "ground_truth": "If fewer than 3 results are found via graph-based retrieval (anchoring), the system triggers a Broad Vector Fallback to search the entire vector space."
    },
    {
        "question": "What is 'Bidirectional Traversal' in the context of the knowledge graph?",
        "ground_truth": "After finding top vector candidates, the system traverses the memory_associations table in both directions (Memory <-> Entity and Entity <-> Entity) to bridge discoveries via shared entities."
    }
]

NOISE_MEMORIES = [
    "Discussion about the weather and its impact on SentenceTransformers training times.",
    "A note about SQLite version 3.3.4 and how it handles VACUUM differently than current versions.",
    "Chat about KNN algorithms in computer vision and why they are slow for high-res images.",
    "Vector search is cool but I prefer keyword search for my personal blog.",
    "Graph theory was my favorite subject in college, especially Dijkstra's algorithm.",
    "I'm feeling a bit tired today, maybe we should talk about vectors later.",
    "Does the context window affect how the LLM remembers my grocery list?",
    "Reverie is a nice name for a plugin, reminds me of Debussy.",
    "I had a dream about a database that could predict the future using semantic search.",
    "My current profile is set to 'Loves Coffee', but I'm actually a tea person.",
    "Data leakage is like a leaky faucet, annoying but manageable if you have the right tools.",
    "Namespace sounds like a place where names live and play together.",
    "Isolation is important for focus, but bad for social health.",
    "Fallback strategies are like backup parachutes, you hope you never need them.",
    "Traversal of the knowledge graph is like a choose-your-own-adventure book.",
    "Entities are the building blocks of existence, or at least of this memory system.",
    "Associations are what make us human, or at least what make the graph connected."
] * 2

def setup_experiment_env():
    if os.path.exists(EXP_DB):
        os.remove(EXP_DB)
    
    provider = ReverieMemoryProvider()
    provider.initialize(session_id="mesa_experiment")
    
    # Manually override db_path if needed for this specific experiment
    # But initialize should now use the safe temp dir from mock
    
    return provider

def inject_data(provider, gold=True):
    memories = TARGET_QUESTIONS if gold else [{"ground_truth": m} for m in NOISE_MEMORIES]
    for item in memories:
        content = item.get("ground_truth", item.get("content_full"))
        importance = 5.0 if gold else 2.0
        date = datetime.now().isoformat() if gold else (datetime.now() - timedelta(days=20)).isoformat()
        
        with provider._db.write_lock() as cursor:
            cursor.execute("""
                INSERT INTO memories (content_full, importance_score, learned_at, last_accessed_at, status)
                VALUES (?, ?, ?, ?, ?)
            """, (content, importance, date, date, "ACTIVE"))
            mem_id = cursor.lastrowid
            vec = provider._enrichment.generate_embedding(content)
            import sqlite_vec
            cursor.execute(
                "INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)",
                (mem_id, sqlite_vec.serialize_float32(vec))
            )

def main():
    if not check_ragas():
        print("Ragas not installed, skipping full experiment. Run in environment with ragas/openai.")
        return

    provider = setup_experiment_env()
    try:
        inject_data(provider, gold=True)
        inject_data(provider, gold=False)
        print("Mesa experiment data injected safely.")
    finally:
        provider.shutdown()
        if os.path.exists(EXP_DB):
            os.remove(EXP_DB)

if __name__ == "__main__":
    main()
