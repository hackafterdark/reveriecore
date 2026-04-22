import sys
import os
from pathlib import Path
import pytest
from unittest.mock import MagicMock

# 1. Authoritative Path Sanitization
# Ensure that the directory containing the 'reveriecore' package is the FIRST in sys.path
# This prevents duplicate module objects (e.g. 'provider' vs 'reveriecore.provider')
project_root = Path(__file__).parent.resolve()
parent_dir = project_root.parent.resolve()

if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
if str(project_root) in sys.path:
    # Remove project root itself to force all imports to use 'reveriecore.' namespace
    sys.path.remove(str(project_root))

# 2. Early Global Mocking
# We mock these BEFORE importing any reveriecore modules
@pytest.fixture(scope="session", autouse=True)
def mock_hermes_env(tmp_path_factory):
    """Ensure hermes_constants and agent are mocked globally for the entire session."""
    tmp_home = tmp_path_factory.mktemp("hermes_home_base")
    (tmp_home / "reveriecore").mkdir(parents=True, exist_ok=True)
    
    # Force sync service OFF for tests to prevent background thread interference
    os.environ["REVERIE_SYNC_SERVICE"] = "false"
    
    # Mock hermes_constants
    try:
        import hermes_constants
    except ImportError:
        hermes_constants = MagicMock()
        sys.modules["hermes_constants"] = hermes_constants

    mock_constants = MagicMock()
    def get_isolated_home():
        env_path = os.getenv("HERMES_HOME")
        return Path(env_path) if env_path else tmp_home
        
    mock_constants.get_hermes_home = get_isolated_home
    sys.modules["hermes_constants"] = mock_constants
    
    # Mock agent and tools
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
    
    yield tmp_home

# 3. Deferred Imports for Fixtures
@pytest.fixture(scope="session")
def db_manager(mock_hermes_env):
    """In-memory database fixture."""
    from reveriecore.database import DatabaseManager
    db = DatabaseManager(":memory:")
    yield db
    db.close()

@pytest.fixture(scope="session")
def enrichment_service():
    """Real model-loading enrichment service fixture."""
    from reveriecore.enrichment import EnrichmentService
    return EnrichmentService()
