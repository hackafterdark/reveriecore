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
# Add parent of reveriecore to path so it can be imported as a package
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent.resolve()
parent_dir = project_root.parent.resolve()

if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from reveriecore.provider import ReverieMemoryProvider

def ingest_docs():
    provider = ReverieMemoryProvider()
    # Initialize with default settings
    provider.initialize(session_id="benchmark_ingestion")
    
    docs_dir = project_root / "AGENT_DOCS"
    prd_dir = project_root / "PRD"
    
    doc_files = list(docs_dir.glob("*.md")) + list(prd_dir.glob("*.md"))
    
    print(f"Found {len(doc_files)} documentation files for ingestion.")
    
    for doc_path in doc_files:
        print(f"Ingesting {doc_path.name}...")
        content = doc_path.read_text()
        
        # We split the content into paragraphs or sections to create multiple memories
        # This is better for RAG than ingesting a massive file as one record.
        sections = content.split("##")
        for i, section in enumerate(sections):
            if not section.strip():
                continue
            
            section_text = section.strip()
            if i > 0:
                section_text = "## " + section_text
            
            # Use sync_turn to simulate a storage event
            # We treat the assistant as providing the documentation
            provider.sync_turn(
                user_content=f"System Documentation: {doc_path.name}",
                assistant_content=section_text,
                session_id="benchmark_ingestion"
            )
            
    print("Ingestion queued. Waiting for background sync to complete...")
    # Give it some time to finish background threads
    import time
    provider.shutdown()
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_docs()
