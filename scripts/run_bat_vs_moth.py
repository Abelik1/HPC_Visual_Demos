from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_demo import run
run("bat_vs_moth", profile="local", frames=70)
