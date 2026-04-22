import logging
import sys
import os
import time
import threading
from datetime import datetime, timedelta
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
    temp_dir = Path("/tmp/reverie_mesa_verify")
    temp_dir.mkdir(parents=True, exist_ok=True)
    hc.get_hermes_home = lambda: temp_dir
    sys.modules["hermes_constants"] = hc

if __name__ == "__main__":
    setup_standalone_mocks()
# --------------------------------------------

# Add project root to path
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent.resolve()
parent_dir = project_root.parent.resolve()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from reveriecore.database import DatabaseManager
from reveriecore.pruning import MesaService
from reveriecore.retrieval import Retriever

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_mesa_tier_1_soft_prune(db, mesa):
    logger.info(">>> Testing Tier 1: Soft Prune Logic")
    with db.write_lock() as cursor:
        cursor.execute("DELETE FROM memories")
        cursor.execute("INSERT INTO memories (content_full, importance_score, last_accessed_at, status, learned_at) VALUES (?, ?, ?, ?, ?)", 
                       ("Stale", 2.0, (datetime.now() - timedelta(days=20)).isoformat(), "ACTIVE", (datetime.now() - timedelta(days=20)).isoformat()))
        frag_id = cursor.lastrowid
        cursor.execute("INSERT INTO memories (content_full, importance_score, last_accessed_at, status, learned_at) VALUES (?, ?, ?, ?, ?)", 
                       ("Anchor", 4.5, (datetime.now() - timedelta(days=20)).isoformat(), "ACTIVE", (datetime.now() - timedelta(days=20)).isoformat()))
        anchor_id = cursor.lastrowid

    mesa.run_soft_prune()
    cursor = db.get_cursor()
    cursor.execute("SELECT status FROM memories WHERE id = ?", (frag_id,))
    assert cursor.fetchone()[0] == "ARCHIVED"
    cursor.execute("SELECT status FROM memories WHERE id = ?", (anchor_id,))
    assert cursor.fetchone()[0] == "ACTIVE"
    logger.info("Tier 1 PASS")

def verify_mesa_tier_2_deep_clean(db, mesa):
    logger.info(">>> Testing Tier 2: Deep Clean")
    with db.write_lock() as cursor:
        cursor.execute("INSERT INTO memories (content_full, status, learned_at) VALUES (?, ?, ?)", ("Ancient", "ARCHIVED", (datetime.now() - timedelta(days=100)).isoformat()))
        old_id = cursor.lastrowid
    mesa.run_deep_clean()
    cursor = db.get_cursor()
    cursor.execute("SELECT id FROM memories WHERE id = ?", (old_id,))
    assert cursor.fetchone() is None
    logger.info("Tier 2 PASS")

def main():
    test_db_path = "tests/mesa_verify.db"
    if os.path.exists(test_db_path): os.remove(test_db_path)
    db = DatabaseManager(test_db_path)
    mesa = MesaService(db, centrality_threshold=2, age_days=14, importance_cutoff=4.0)
    try:
        verify_mesa_tier_1_soft_prune(db, mesa)
        verify_mesa_tier_2_deep_clean(db, mesa)
        logger.info("VERIFICATION COMPLETE")
    finally:
        db.close()
        if os.path.exists(test_db_path): os.remove(test_db_path)

if __name__ == "__main__":
    main()
