
import sys
import os
import logging
from pathlib import Path

# Add workspace to path
workspace = "/home/tom/.hermes/plugins"
if workspace not in sys.path:
    sys.path.append(workspace)

from reveriecore.provider import ReverieMemoryProvider

logging.basicConfig(level=logging.INFO)

def test_search():
    print("Initializing Provider...")
    p = ReverieMemoryProvider()
    
    # Mocking Hermes environment variables
    kwargs = {
        "config_dir": "/home/tom/.hermes",
        "config": {
            "memory_token_limit": 8192
        }
    }
    
    p.initialize(**kwargs)
    print("Provider Initialized. Starting search for 'poem'...")
    
    try:
        # Exact same call the agent makes
        result = p.handle_tool_call("reverie_search", {"query": "poem"})
        print("\n--- Search Result ---")
        print(result)
        print("---------------------")
    except Exception as e:
        print(f"\nCRASH DETECTED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_search()
