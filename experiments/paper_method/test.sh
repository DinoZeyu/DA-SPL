#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate da-spl-repro
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
exec python -m unittest discover -s experiments/paper_method/tests -p 'test_*.py' -v
