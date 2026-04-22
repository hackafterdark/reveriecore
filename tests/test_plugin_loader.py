import sys
import os
import pytest
from pathlib import Path

def test_plugin_package_importability():
    """
    Regression Test: Simulates the Hermes loading process by importing 
    the provider from the package. This ensures that relative imports 
    work correctly and don't raise ModuleNotFoundError.
    """
    # 1. Setup path to simulate package loading from ~/.hermes/plugins/
    # The plugin is located at .../.hermes/plugins/reveriecore
    # We want to add .../.hermes/plugins/ to sys.path
    project_root = Path(__file__).parent.parent
    plugins_dir = str(project_root.parent)
    
    # Pre-check if it's already there (it shouldn't be for a clean test)
    # But if we are running in the current environment it might be.
    
    original_path = list(sys.path)
    if plugins_dir not in sys.path:
        sys.path.insert(0, plugins_dir)
        
    try:
        # 2. Attempt import from package path
        # We use __import__ to avoid issues with static analysis if it were a direct import
        module = __import__("reveriecore.provider", fromlist=["ReverieMemoryProvider"])
        provider_class = getattr(module, "ReverieMemoryProvider")
        
        assert provider_class is not None
        print("Plugin successfully loaded as package.")
        
    except ImportError as e:
        pytest.fail(f"Plugin failed to load as package (Relative import issue?): {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error during plugin package import: {e}")
    finally:
        # Cleanup
        sys.path = original_path
        # Remove from sys.modules to ensure re-import in other tests if needed
        # (Though pytest usually isolates this enough)
        for key in list(sys.modules.keys()):
            if key.startswith("reveriecore"):
                del sys.modules[key]
