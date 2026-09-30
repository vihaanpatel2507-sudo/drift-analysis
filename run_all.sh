#!/usr/bin/env bash
# Re-run the full B -> C -> D -> E -> F pipeline after the look-ahead leakage
# fix in phaseB. Backs up the existing results_phase* directories first so
# the pre-fix numbers can be diffed against the new ones.
#
# Usage:  ./run_all.sh [path/to/SAML-D.csv]
#
# Requires Python 3.12+ (numpy and xgboost both declare >=3.12).

set -euo pipefail

DATA_PATH="${1:-/Users/vihaanpatel/Downloads/SAML-D 2.csv}"
PY="${PYTHON:-python3}"

if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: dataset not found at: $DATA_PATH"
    echo "Pass the path as the first argument: ./run_all.sh /path/to/SAML-D.csv"
    exit 1
fi

echo "=============================================================="
echo "AML DRIFT PIPELINE — FULL RE-RUN"
echo "  data:   $DATA_PATH"
echo "  python: $($PY --version 2>&1)"
echo "=============================================================="

# ---- backup existing results so the pre-leakfix numbers survive -------
BACKUP_DIR="results_pre_leakfix"
if [ -d "$BACKUP_DIR" ] && [ -n "$(ls -A "$BACKUP_DIR" 2>/dev/null)" ]; then
    BACKUP_DIR="results_pre_leakfix_$(date +%Y%m%d_%H%M%S)"
    echo "Backup dir already exists; using $BACKUP_DIR"
fi
if ls -d results_phase* >/dev/null 2>&1; then
    mkdir -p "$BACKUP_DIR"
    cp -R results_phase* "$BACKUP_DIR"/
    echo "Backed up existing results -> $BACKUP_DIR/"
fi

# ---- dependency check --------------------------------------------------
$PY - <<'EOF'
import sys
missing = []
for mod in ["numpy", "pandas", "matplotlib", "sklearn", "scipy", "xgboost"]:
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
if missing:
    sys.exit("ERROR: missing packages: " + ", ".join(missing) +
             "\nInstall with:  pip install -r requirements.txt")
print("Dependencies OK")
EOF

# ---- phase B: chunking (MUST run first — it regenerates chunks/) ------
echo ""
echo ">>> PHASE B — temporal chunking (this is the step that changes results)"
$PY phaseB_chunking.py --data_path "$DATA_PATH"

# ---- phases C, D, E: the measurement phases ---------------------------
echo ""
echo ">>> PHASE C — baseline training + decay measurement"
$PY phaseC_baseline_model.py

echo ""
echo ">>> PHASE D — label-free drift detectors"
$PY phaseD_drift_detectors.py

echo ""
echo ">>> PHASE E — label scarcity vs matched no-drift control"
$PY phaseE_label_scarcity.py

echo ""
echo "PHASES B-E COMPLETE — results now regenerable for the docs."
echo "Check the decay table and Phase E ratio before committing compute to F:"
echo "  cat results_phaseC/baseline_decay_summary.csv"
echo "  cat results_phaseE/label_scarcity_summary.csv"
echo ""
echo ">>> PHASE F — retraining (expensive: 4 strategies + 4-point weight sweep)"
echo "    It holds all 10 chunks in memory. Expect it to be the long step."
$PY phaseF_retraining.py

echo ""
echo "=============================================================="
echo "PIPELINE COMPLETE"
echo "  new results:    results_phaseC/ results_phaseD/ results_phaseE/ results_phaseF/"
echo "  pre-fix backup: $BACKUP_DIR/"
echo ""
echo "Next: reconcile the number tables in README.md and results_explained.txt"
echo "against the new CSVs."
echo "=============================================================="
