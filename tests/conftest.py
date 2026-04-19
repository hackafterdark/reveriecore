import pytest
import os
import sys
from pathlib import Path

# Add the parent directory of reveriecore to path so we can import it
# assuming tests/ is inside reveriecore/
sys.path.append(str(Path(__file__).parent.parent.parent))

from reveriecore.database import DatabaseManager
from reveriecore.enrichment import EnrichmentService

@pytest.fixture(scope="session")
def db_manager():
    """In-memory database fixture."""
    # Use :memory: for fast, isolated tests
    db = DatabaseManager(":memory:")
    yield db
    db.close()

@pytest.fixture(scope="session")
def enrichment_service():
    """Real model-loading enrichment service fixture."""
    # This will load the actual models (all-MiniLM-L6-v2)
    service = EnrichmentService()
    return service
