"""ReverieCore Memory Plugin — Local RAG MemoryProvider entry point.
"""

from .provider import ReverieMemoryProvider

# No 'register' function needed anymore—the agent discovers it by the class inheritance
# Dummy registration to silence the plugin loader
def register(ctx):
    """Placeholder to satisfy the generic plugin loader."""
    pass