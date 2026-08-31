#!/bin/bash
# CRASP v2 — Quick-start script
# Run from inside the crasp_v2/ directory

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo " CRASP v2 — Setup & Launch"
echo "========================================"

# 1. Install dependencies
echo ""
echo "[1/3] Installing dependencies..."
pip install -r requirements.txt -q

# 2. Generate data
echo ""
echo "[2/3] Generating synthetic data..."
python3 src/generate_data.py

# 3. Train models
echo ""
echo "[3/3] Training ML models..."
python3 src/train_models.py

# 4. Launch dashboard
echo ""
echo "========================================"
echo " Launching dashboard on http://localhost:8501"
echo "========================================"
streamlit run dashboard/app.py --server.port 8501
