#!/usr/bin/env bash
# setup.sh — One-command environment setup for hllset_cortex notebooks
#
# Usage:
#   source setup.sh          # sets up current shell
#   bash setup.sh            # just builds/installs
#
# Sets up:
#   1. Python virtualenv (.venv) with Jupyter + ipykernel
#   2. Rust hllset_py crate (maturin build + install)
#   3. hllset_cortex Python package (editable install)
#   4. Jupyter kernel registration
#
# After setup:
#   jupyter notebook notebooks/
#   → select kernel "Python (hllset-cortex)"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== hllset_cortex environment setup ==="
echo ""

# ── 1. Python virtualenv ─────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "[1/5] Creating Python virtualenv..."
    python3 -m venv .venv
else
    echo "[1/5] Virtualenv exists: .venv/"
fi

source .venv/bin/activate
pip install -q maturin jupyter ipykernel
echo "       Python: $(python3 --version)"

# ── 2. Build Rust crate ──────────────────────────────────────────────
echo "[2/5] Building hllset_py (Rust → wheel)..."
cd crates/hllset_py
maturin build --release -q
WHEEL=$(ls -t target/wheels/hllset_py-*.whl 2>/dev/null | head -1)
echo "       Wheel: $(basename "$WHEEL")"

# ── 3. Install into .venv ────────────────────────────────────────────
echo "[3/5] Installing into .venv..."
pip install -q "$WHEEL" --force-reinstall
cd "$SCRIPT_DIR"

# ── 4. Install hllset_cortex Python package ───────────────────────────
echo "[4/5] Installing hllset_cortex (editable)..."
pip install -q -e .

# ── 5. Register Jupyter kernel ───────────────────────────────────────
echo "[5/5] Registering Jupyter kernel..."
python3 -m ipykernel install --user --name hllset-cortex --display-name "Python (hllset-cortex)" 2>/dev/null || true

# ── Done ─────────────────────────────────────────────────────────────
echo ""
echo "=== Setup complete ==="
echo ""
echo "  Start notebook:  jupyter notebook notebooks/"
echo "  Kernel:          Python (hllset-cortex)"
echo ""
echo "  Or test directly:"
echo "    source .venv/bin/activate"
echo "    python3 -c 'from hllset_cortex import HLLSetFilter; print(\"OK\")'"
echo ""

# Also install into conda deepseek-ocr if present
if [ -d "$HOME/.conda/envs/deepseek-ocr" ]; then
    echo "Conda deepseek-ocr found — installing there too..."
    $HOME/.conda/envs/deepseek-ocr/bin/pip install -q "$SCRIPT_DIR/crates/hllset_py/$WHEEL" --force-reinstall 2>/dev/null || true
    $HOME/.conda/envs/deepseek-ocr/bin/pip install -q -e "$SCRIPT_DIR" 2>/dev/null || true
    echo "       Installed into conda deepseek-ocr"
fi
