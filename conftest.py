import sys
import os
from pathlib import Path

# Add the directory containing 'reveriecore' to sys.path so that tests can 
# import it as a package. This allows relative imports inside the package
# modules to work correctly during local pytest runs.
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
