#!/usr/bin/env bash
# Run all three training steps in sequence, each in a fresh Python process
# to avoid the torch+xgboost OpenMP clash on macOS.
set -e
cd "$(dirname "$0")/.."
source venv/bin/activate
python -u scripts/01_train_glm_xgb.py
python -u scripts/02_train_nn.py
python -u scripts/03_compute_stats.py
echo "All artifacts written to models/."
