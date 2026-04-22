"""ReverieCore Memory Plugin — Local RAG MemoryProvider entry point.
"""
import logging
from .provider import ReverieMemoryProvider

def register(ctx):
    """Register the reverie memory provider with the plugin system."""
    try:
        provider = ReverieMemoryProvider()
        ctx.register_memory_provider(provider)
    except Exception as e:
        logging.getLogger(__name__).error("FATAL REVERIE INSTANTIATION ERROR: %s", e, exc_info=True)