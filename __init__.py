"""ReverieCore Memory Plugin — Local RAG MemoryProvider entry point.
"""
import sys
import logging
from types import ModuleType

# Initialize the virtual parent namespace for user-installed plugins.
# This ensures that relative imports like `from .provider import ...` work 
# correctly within the Hermes discovery system.
parent_pkg = "_hermes_user_memory"
if parent_pkg not in sys.modules:
    m = ModuleType(parent_pkg)
    m.__path__ = []
    sys.modules[parent_pkg] = m

from .provider import ReverieMemoryProvider
from .schemas import MesaConfig, RetrievalConfig, RetrievalContext

def register(ctx):
    """Register the reverie memory provider with the plugin system."""
    try:
        provider = ReverieMemoryProvider()
        ctx.register_memory_provider(provider)
    except Exception as e:
        logging.getLogger(__name__).error("FATAL REVERIE INSTANTIATION ERROR: %s", e, exc_info=True)