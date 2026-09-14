"""Render one Star in a Bottle run.

Pass a mode to choose the solver: `passive` (the default) evolves the plasma
field alone, `guardian` adds the trained neural controller and the confined
markers it is holding inside the torus.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_demo import run

method = sys.argv[1] if len(sys.argv) > 1 else "passive"
run("fusion_plasma", profile="local", frames=70, method=method)
