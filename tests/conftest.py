import pytest
import os
import sys
from pathlib import Path

# Add the project root to sys.path
# If this file is at .../reveriecore/tests/conftest.py
# The project root is two levels up: .../reveriecore/
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

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
