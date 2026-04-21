# --- Mock Hermes Core (Must be first) ---
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock Hermes environment
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
from ragas import evaluate
from ragas.metrics import Faithfulness, ContextPrecision
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI
from datasets import Dataset

# --- Experiment Constants ---
EXP_DB = "tests/mesa_experiment.db"
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
] * 2 # 34 noise records

def setup_experiment_env():
    if os.path.exists(EXP_DB):
        os.remove(EXP_DB)
    
    provider = ReverieMemoryProvider()
    # Mocking storage path to our experiment db
    provider.initialize(session_id="mesa_experiment", db_path=EXP_DB)
    
    # Setup LLM - Specifically targeting Ollama 11434 which was seen in config as first choice
    config = ConfigLoader.load_config()
    providers = config.get("providers", [])
    
    # Try to find Ollama specifically or fall back to first provider
    ollama_provider = next((p for p in providers if p.get("name") == "Ollama"), None)
    target = ollama_provider or (providers[0] if providers else {})
    
    base_url = target.get("base_url") or "http://172.22.0.1:11434/v1"
    model_name = target.get("model") or "gemma2:2b"
    
    logger.info(f"Using LLM: {model_name} at {base_url} with 180s timeout")
    
    llm = ChatOpenAI(
        base_url=base_url,
        api_key="sk-reverie-exp",
        model=model_name,
        temperature=0,
        timeout=180.0,
        max_retries=3
    )
    
    return provider, llm

def inject_data(provider, gold=True):
    logger.info(f"Injecting {'GOLD' if gold else 'NOISE'} data...")
    memories = TARGET_QUESTIONS if gold else [{"ground_truth": m} for m in NOISE_MEMORIES]
    
    for item in memories:
        content = item["ground_truth"]
        importance = 5.0 if gold else 2.0
        # Stale date for noise, recent for gold
        date = datetime.now().isoformat() if gold else (datetime.now() - timedelta(days=20)).isoformat()
        
        with provider._db.write_lock() as cursor:
            cursor.execute("""
                INSERT INTO memories (content_full, importance_score, learned_at, last_accessed_at, status)
                VALUES (?, ?, ?, ?, ?)
            """, (content, importance, date, date, "ACTIVE"))
            mem_id = cursor.lastrowid
            
            # Manually embed so it shows up in KNN searches
            vec = provider._enrichment.generate_embedding(content)
            import sqlite_vec
            cursor.execute(
                "INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)",
                (mem_id, sqlite_vec.serialize_float32(vec))
            )
            
    logger.info(f"Injected {len(memories)} {'GOLD' if gold else 'NOISE'} records with vector embeddings.")

def run_mini_eval(provider, llm, stage_name):
    logger.info(f"--- Running RAGAS Evaluation: {stage_name} ---")
    results = []
    for item in TARGET_QUESTIONS:
        question = item["question"]
        
        # Retrieval
        query_vec = provider._enrichment.generate_embedding(question)
        retrieved = provider._retriever.search(
            query_vec, 
            query_text=question,
            limit=3
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

    df = pd.DataFrame(results)
    dataset = Dataset.from_pandas(df)
    
    run_config = RunConfig(max_workers=2)
    eval_result = evaluate(
        dataset,
        metrics=[Faithfulness(), ContextPrecision()],
        llm=llm,
        run_config=run_config
    )
    
    return eval_result.scores

def main():
    provider, llm = setup_experiment_env()
    
    try:
        # 1. State: GOLD + NOISE
        inject_data(provider, gold=True)
        inject_data(provider, gold=False)
        
        before_scores = run_mini_eval(provider, llm, "BEFORE (Cluttered)")
        
        # State: RUN MESA
        logger.info(">>> Triggering Mesa Maintenance Intervention...")
        mesa = MesaService(provider._db, centrality_threshold=2, age_days=14, importance_cutoff=4.0)
        mesa.run_soft_prune()
        
        # Verify pruning worked
        cursor = provider._db.get_cursor()
        cursor.execute("SELECT count(*) FROM memories WHERE status = 'ARCHIVED'")
        archived = cursor.fetchone()[0]
        logger.info(f"Mesa Intervention Result: {archived} memories ARCHIVED.")
        
        after_scores = run_mini_eval(provider, llm, "AFTER (Sanitized)")
        
        # 3. Report Comparison
        print("\n" + "="*60)
        print("MESA SERVICE PERFORMANCE EXPERIMENT")
        print("="*60)
        print(f"{'Metric':<25} | {'Before (Cluttered)':<20} | {'After (Sanitized)':<20}")
        print("-" * 60)
        for metric in before_scores.keys():
            b = before_scores[metric]
            a = after_scores[metric]
            diff = a - b
            print(f"{metric:<25} | {b:<20.4f} | {a:<20.4f} ({'+' if diff >=0 else ''}{diff:.4f})")
        print("="*60)
        
    finally:
        provider.shutdown()
        if os.path.exists(EXP_DB):
            os.remove(EXP_DB)

if __name__ == "__main__":
    main()
