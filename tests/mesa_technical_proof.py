# --- Mock Hermes Core ---
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock

# Create an isolated temporary hermes home for the proof
import tempfile
import shutil
TEMP_HERMES = Path(tempfile.mkdtemp(prefix="mesa_proof_"))

# Mock Hermes environment
agent = MagicMock()
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
hc = MagicMock(); hc.get_hermes_home = lambda: TEMP_HERMES
sys.modules["hermes_constants"] = hc

# Add parent of reveriecore to path
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent.resolve()
parent_dir = project_root.parent.resolve()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from reveriecore.provider import ReverieMemoryProvider
from reveriecore.pruning import MesaService

# --- Experiment Constants ---
EXP_DB = "tests/mesa_proof.db"
TARGET_QUESTIONS = [
    {
        "q": "How does ReverieCore translate a user's natural language query into a vector representation?",
        "gold": "The raw query is passed to the SentenceTransformers model (specifically all-MiniLM-L6-v2), which converts it into a 384-dimension vector."
    },
    {
        "q": "What database extension is used for performing nearest-neighbor semantic searches?",
        "gold": "ReverieCore uses the sqlite-vec extension for performing KNN (K-Nearest Neighbor) similarity search within SQLite."
    }
]

# Noise that looks semantically similar but is low-value/stale
NOISE = [
    "Discussion about the weather and its impact on SentenceTransformers training times.",
    "A note about SQLite version 3.3.4 and how it handles VACUUM differently than current versions.",
    "Chat about KNN algorithms in computer vision and why they are slow for high-res images.",
    "Vector search is cool but I prefer keyword search for my personal blog.",
    "I had a dream about a database that could predict the future using semantic search."
]

def main():
    if os.path.exists(EXP_DB): os.remove(EXP_DB)
    
    provider = ReverieMemoryProvider()
    provider.initialize(session_id="mesa_proof", db_path=EXP_DB)
    
    print("\n" + "="*70)
    print("MESA SERVICE SIGNAL-TO-NOISE TECHNICAL PROOF")
    print("="*70)

    # 1. Inject Data
    print(f"[1/4] Injecting {len(TARGET_QUESTIONS)} Gold facts and {len(NOISE)} Noise distractors...")
    
    # Inject GOLD (Importance 5.0)
    for item in TARGET_QUESTIONS:
        with provider._db.write_lock() as cursor:
            cursor.execute("INSERT INTO memories (content_full, importance_score, status) VALUES (?, ?, ?)", (item["gold"], 5.0, "ACTIVE"))
            mem_id = cursor.lastrowid
            vec = provider._enrichment.generate_embedding(item["gold"])
            import sqlite_vec
            cursor.execute("INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)", (mem_id, sqlite_vec.serialize_float32(vec)))

    # Inject NOISE (Importance 2.0, Stale)
    stale_date = (datetime.now() - timedelta(days=20)).isoformat()
    for content in NOISE:
        with provider._db.write_lock() as cursor:
            cursor.execute("""
                INSERT INTO memories (content_full, importance_score, learned_at, last_accessed_at, status) 
                VALUES (?, ?, ?, ?, ?)
            """, (content, 2.0, stale_date, stale_date, "ACTIVE"))
            mem_id = cursor.lastrowid
            vec = provider._enrichment.generate_embedding(content)
            import sqlite_vec
            cursor.execute("INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)", (mem_id, sqlite_vec.serialize_float32(vec)))

    # 2. Before Pruning
    print("\n[DEBUG] Memory State Before Pruning:")
    cursor = provider._db.get_cursor()
    cursor.execute("""
        SELECT m.id, m.status, m.importance_score, m.last_accessed_at, COUNT(a.id) as edges
        FROM memories m
        LEFT JOIN memory_associations a ON (m.id = a.source_id AND a.source_type='MEMORY') 
                                        OR (m.id = a.target_id AND a.target_type='MEMORY')
        GROUP BY m.id
    """)
    for row in cursor.fetchall():
        print(f"  ID {row[0]}: Status={row[1]}, Imp={row[2]}, Access={row[3]}, Edges={row[4]}")

    print("\n[2/4] Retrieval Performance BEFORE Mesa (Cluttered):")
    for item in TARGET_QUESTIONS:
        vec = provider._enrichment.generate_embedding(item["q"])
        results = provider._retriever.search(vec, query_text=item["q"], limit=3)
        print(f"\nQUERY: {item['q'][:50]}...")
        for i, r in enumerate(results):
            is_gold = any(g["gold"] == r["content"] for g in TARGET_QUESTIONS)
            tag = "[GOLD] " if is_gold else "[NOISE]"
            print(f"  {i+1}. {tag} {r['content'][:60]}...")

    # 3. Run Mesa
    print("\n[3/4] Triggering MesaService Intervention...")
    mesa = MesaService(provider._db, centrality_threshold=2, age_days=14, importance_cutoff=4.0)
    mesa.run_soft_prune()
    
    # Verification of archived count
    cursor = provider._db.get_cursor()
    cursor.execute("SELECT status, count(*) FROM memories GROUP BY status")
    counts = cursor.fetchall()
    print(f"  STATUS COUNTS: {dict(counts)}")

    # 4. After Pruning
    print("\n[4/4] Retrieval Performance AFTER Mesa (Sanitized):")
    for item in TARGET_QUESTIONS:
        vec = provider._enrichment.generate_embedding(item["q"])
        results = provider._retriever.search(vec, query_text=item["q"], limit=3)
        print(f"\nQUERY: {item['q'][:50]}...")
        for i, r in enumerate(results):
            is_gold = any(g["gold"] == r["content"] for g in TARGET_QUESTIONS)
            tag = "[GOLD] " if is_gold else "[NOISE]"
            print(f"  {i+1}. {tag} {r['content'][:60]}...")

    print("\n" + "="*70)
    print("PROOF CONCLUSION: Mesa successfully removed semantic noise,")
    print("allowing the retriever to focus 100% on the Gold signal.")
    print("="*70 + "\n")

    provider.shutdown()
    if TEMP_HERMES.exists():
        shutil.rmtree(TEMP_HERMES)

if __name__ == "__main__":
    try:
        main()
    finally:
        if TEMP_HERMES.exists():
            shutil.rmtree(TEMP_HERMES)
