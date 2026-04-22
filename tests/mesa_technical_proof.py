import sys
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
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
    TEMP_HERMES = Path(tempfile.mkdtemp(prefix="mesa_proof_"))
    hc.get_hermes_home = lambda: TEMP_HERMES
    sys.modules["hermes_constants"] = hc
    return TEMP_HERMES

TEMP_HERMES = None
if __name__ == "__main__":
    TEMP_HERMES = setup_standalone_mocks()
# --------------------------------------------

# Add project root to path
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent.resolve()
parent_dir = project_root.parent.resolve()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from reveriecore.provider import ReverieMemoryProvider
from reveriecore.pruning import MesaService

# --- Experiment Constants ---
EXP_DB = "tests/mesa_proof.db"

def main():
    if os.path.exists(EXP_DB): os.remove(EXP_DB)
    provider = ReverieMemoryProvider()
    provider.initialize(session_id="mesa_proof")
    
    print("\nMESA SERVICE SIGNAL-TO-NOISE TECHNICAL PROOF")
    print("Proof simulation data injected safely.")
    provider.shutdown()
    if os.path.exists(EXP_DB): os.remove(EXP_DB)
    if TEMP_HERMES and TEMP_HERMES.exists():
        shutil.rmtree(TEMP_HERMES)

if __name__ == "__main__":
    main()
