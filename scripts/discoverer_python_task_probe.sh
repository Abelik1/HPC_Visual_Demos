#!/usr/bin/env bash

source /etc/profile
set -euo pipefail
module load python312

VENV=/weka/ehpc-school-2026/abelik/venvs/visual-demos
echo "Host: $(hostname)"
echo "Python command: $(command -v python3)"
python3 --version
module list
ls -l /cm/local/apps/python312/bin/python3 "$VENV/bin/python" "$VENV/bin/python3"
"$VENV/bin/python" --version
