import sys
import importlib.util
from pathlib import Path
import unittest

class TestHermesPluginLoader(unittest.TestCase):
    """
    Simulates the exact dynamic plugin module loader used by Hermes.
    This prevents subtle 'ModuleNotFoundError' and 'ImportError' issues
    caused by absolute imports within the package colliding with the 
    submodule_search_locations constraints during dynamic execution.
    """

    def test_dynamic_package_loader(self):
        provider_dir = Path(__file__).parent.parent
        module_name = "_hermes_user_memory.reveriecore"
        
        # Mock Hermes parent dependency
        class MockParent: pass
        if "_hermes_user_memory" not in sys.modules:
            sys.modules["_hermes_user_memory"] = MockParent()

        init_file = provider_dir / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            module_name, str(init_file),
            submodule_search_locations=[str(provider_dir)]
        )
        
        self.assertIsNotNone(spec, "__init__.py spec should not be None")
        
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        
        # Dynamic submodule execution (Hermes style)
        for sub_file in sorted(provider_dir.glob("*.py")):
            if sub_file.name == "__init__.py":
                continue
            
            sub_name = sub_file.stem
            full_sub_name = f"{module_name}.{sub_name}"
            
            if full_sub_name not in sys.modules:
                sub_spec = importlib.util.spec_from_file_location(
                    full_sub_name, str(sub_file)
                )
                if sub_spec:
                    sub_mod = importlib.util.module_from_spec(sub_spec)
                    sys.modules[full_sub_name] = sub_mod
                    try:
                        sub_spec.loader.exec_module(sub_mod)
                    except Exception as e:
                        self.fail(f"Submodule loading for {sub_file.name} threw {type(e).__name__}: {e}")
        
        # Finally execute __init__ which cascades the imports
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            self.fail(f"__init__.py execution threw {type(e).__name__}: {e}")

        # Check that it properly exports the register function
        self.assertTrue(hasattr(mod, "register"), "Plugin must export a 'register(ctx)' function")

if __name__ == '__main__':
    unittest.main()
