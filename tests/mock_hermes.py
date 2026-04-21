import sys
from unittest.mock import MagicMock

# 1. Mock 'agent' module
agent = MagicMock()
agent.memory_provider = MagicMock()
class BaseMemoryProvider:
    def __init__(self, *args, **kwargs):
        pass
    def initialize(self, *args, **kwargs):
        pass
    def shutdown(self, *args, **kwargs):
        pass

agent.memory_provider.MemoryProvider = BaseMemoryProvider
sys.modules["agent"] = agent
sys.modules["agent.memory_provider"] = agent.memory_provider

# 2. Mock 'tools' module
tools = MagicMock()
tools.registry = MagicMock()
tools.registry.tool_error = lambda x: f"Error: {x}"
sys.modules["tools"] = tools
sys.modules["tools.registry"] = tools.registry

# 3. Mock 'hermes_constants'
hermes_constants = MagicMock()
from pathlib import Path
hermes_constants.get_hermes_home = lambda: Path.home() / ".hermes"
sys.modules["hermes_constants"] = hermes_constants

print("Hermes core modules mocked successfully.")
