"""pytest fixtures shared across the suite."""

import sys
from pathlib import Path

# Make ``import z_jev`` work without ``pip install -e .`` in CI.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
