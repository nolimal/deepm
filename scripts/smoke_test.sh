#!/usr/bin/env bash
# smoke_test.sh — Quick end-to-end smoke test of the full paper pipeline.
#
# ⚠ SMOKE RESULTS ARE NOT MEANINGFUL — they exist solely to verify that the
#   pipeline runs end-to-end without errors.  The reduced HP budget, fewer
#   seeds, and single short test window mean metrics will differ substantially
#   from the paper.  Use scripts/reproduce.sh for publishable results.
#
# Runs a lightweight version of reproduce.sh covering ALL model types:
#   - Traditional baselines (no training needed)
#   - Temporal baselines (Momentum Transformer)
#   - DeePM-GAT main model
#   - All architecture, feature, loss, and cost ablations
#
# Smoke overrides vs production:
#   - 5 HP trials (instead of 50–100)
#   - 5 seeds (instead of 10/25)
#   - 1 test window: [2020] (instead of 3)
#
# Prerequisites:
#   1. pip install -e .  (or uv sync)
#   2. Place raw price data at data/data_dec25.parquet
#   3. Set WANDB_ENTITY and WANDB_API_KEY (or WANDB_MODE=offline)
#
# Usage:
#   bash scripts/smoke_test.sh                # full smoke pipeline
#   bash scripts/smoke_test.sh --step N       # run a single step (1–7)
#   bash scripts/smoke_test.sh --baselines    # traditional baselines only (no training)

set -euo pipefail

BASELINES_ONLY=false
STEP=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --baselines)  BASELINES_ONLY=true; shift ;;
        --step)       STEP="$2"; shift 2 ;;
        *)            echo "Unknown arg: $1"; exit 1 ;;
    esac
done

run_step() {
    local n=$1
    shift
    if [ "$STEP" -eq 0 ] || [ "$STEP" -eq "$n" ]; then
        echo ""
        echo "=========================================="
        echo "  Step $n: $1"
        echo "=========================================="
        shift
        "$@"
    fi
}

# ──────────────────────────────────────────────
# Step 1: Generate adjacency matrix
# ──────────────────────────────────────────────
run_step 1 "Generate adjacency matrix" \
    python scripts/build_graph.py

# ──────────────────────────────────────────────
# Step 2: Prepare features (skip if already exists)
# ──────────────────────────────────────────────
step2() {
    FEATS_FILE="data/feats-data_dec25.parquet"
    if [ -f "$FEATS_FILE" ]; then
        echo "  Features file already exists: $FEATS_FILE — skipping."
    else
        python scripts/prepare_features.py
    fi
}
run_step 2 "Prepare features" step2

# ──────────────────────────────────────────────
# Traditional baselines helper (shared by both modes)
# ──────────────────────────────────────────────
backtest() {
    local config=$1
    echo "  Backtesting: $config"
    python -m deepm.backtest --name "$config" --diagnostics
}

BASELINE_CONFIGS=(
    bt-baseline-longonly
    bt-baseline-tsmom
    bt-baseline-tsmom-rm
    bt-baseline-tsmom-mvo
    bt-baseline-tsmom-mvo-tp
    bt-baseline-tsmom-erc
    bt-baseline-macd
    bt-baseline-macd-rm
    bt-baseline-macd-mvo-tp
)

run_baselines() {
    for cfg in "${BASELINE_CONFIGS[@]}"; do
        backtest "$cfg"
    done
}

aggregate_baselines() {
    python scripts/aggregate_metrics.py \
        --title "Traditional Baselines (Smoke)" \
        --csv backtest_results/smoke_baselines_metrics.csv \
        "${BASELINE_CONFIGS[@]}"
}

# ──────────────────────────────────────────────
# --baselines: fast path (no training needed)
# ──────────────────────────────────────────────
if [ "$BASELINES_ONLY" = true ]; then
    echo ""
    echo "=========================================="
    echo "  Baselines-only mode (steps 1–2 already ran above)"
    echo "=========================================="

    echo ""
    echo "--- Backtest traditional baselines ---"
    run_baselines

    echo ""
    echo "--- Aggregate metrics ---"
    aggregate_baselines

    echo ""
    echo "=========================================="
    echo "  Baselines complete."
    echo "=========================================="
    exit 0
fi

# ──────────────────────────────────────────────
# Step 3: Generate smoke configs from production
# ──────────────────────────────────────────────
run_step 3 "Generate smoke configs" \
    python scripts/generate_smoke_configs.py

# ──────────────────────────────────────────────
# Step 4: Train DL models (smoke settings)
# ──────────────────────────────────────────────
train_model() {
    local config=$1
    local arch=$2
    echo "  Training: $config / $arch"
    python -m deepm.training -r "$config" -a "$arch"
}

step4() {
    # DeePM-GAT (main model)
    train_model "smoke-deepm-gat" "DeePM"

    # Architecture ablations
    train_model "smoke-deepm-gcn" "DeePM"
    train_model "smoke-deepm-no-graph" "DeePM"
    train_model "smoke-deepm-graph-only" "DeePM"
    train_model "smoke-deepm-independent" "DeePM"
    train_model "smoke-deepm-gat-flip" "DeePM"
    train_model "smoke-deepm-gat-no-rezero" "DeePM"

    # Feature ablations
    train_model "smoke-deepm-gat-macd" "DeePM"
    train_model "smoke-deepm-gat-cascading" "DeePM"

    # Loss ablations
    train_model "smoke-deepm-gat-pooled-only" "DeePM"
    train_model "smoke-deepm-gat-softmin-t1" "DeePM"
    train_model "smoke-deepm-gat-softmin-t005" "DeePM"

    # Cost ablations
    train_model "smoke-deepm-gat-zero-cost" "DeePM"
    train_model "smoke-deepm-gat-full-cost" "DeePM"

    # Temporal baselines
    train_model "smoke-temporal-baseline" "AdvancedTemporalBaseline"
    train_model "smoke-temporal-baseline-zero-cost" "AdvancedTemporalBaseline"
}
run_step 4 "Train DL models" step4

# ──────────────────────────────────────────────
# Step 5: Run backtests
# ──────────────────────────────────────────────
step5() {
    # DL model backtests (smoke configs)
    backtest "bt-smoke-deepm-gat"
    backtest "bt-smoke-deepm-gcn"
    backtest "bt-smoke-deepm-no-graph"
    backtest "bt-smoke-deepm-graph-only"
    backtest "bt-smoke-deepm-independent"
    backtest "bt-smoke-deepm-gat-flip"
    backtest "bt-smoke-deepm-gat-no-rezero"
    backtest "bt-smoke-deepm-gat-macd"
    backtest "bt-smoke-deepm-gat-cascading"
    backtest "bt-smoke-deepm-gat-pooled-only"
    backtest "bt-smoke-deepm-gat-softmin-t1"
    backtest "bt-smoke-deepm-gat-softmin-t005"
    backtest "bt-smoke-deepm-gat-zero-cost"
    backtest "bt-smoke-deepm-gat-full-cost"

    # Temporal baselines (smoke configs)
    backtest "bt-smoke-temporal-baseline"
    backtest "bt-smoke-temporal-baseline-zero-cost"

    # Traditional baselines (production configs — no training needed)
    run_baselines
}
run_step 5 "Run backtests" step5

# ──────────────────────────────────────────────
# Step 6: Aggregate metrics tables
# ──────────────────────────────────────────────
step6() {
    echo ""
    python scripts/aggregate_metrics.py \
        --title "Table 1: Main Results (Smoke)" \
        "${BASELINE_CONFIGS[@]}" \
        bt-smoke-temporal-baseline-zero-cost \
        bt-smoke-temporal-baseline \
        bt-smoke-deepm-gat

    echo ""
    python scripts/aggregate_metrics.py \
        --title "Table 2: Ablation Studies (Smoke)" \
        bt-smoke-deepm-gat \
        bt-smoke-deepm-gcn \
        bt-smoke-deepm-no-graph \
        bt-smoke-deepm-graph-only \
        bt-smoke-deepm-independent \
        bt-smoke-deepm-gat-flip \
        bt-smoke-deepm-gat-no-rezero \
        bt-smoke-deepm-gat-macd \
        bt-smoke-deepm-gat-cascading \
        bt-smoke-deepm-gat-pooled-only \
        bt-smoke-deepm-gat-softmin-t1 \
        bt-smoke-deepm-gat-softmin-t005 \
        bt-smoke-deepm-gat-zero-cost \
        bt-smoke-deepm-gat-full-cost

    # Save CSV (suppress table reprint)
    python scripts/aggregate_metrics.py --auto \
        --csv backtest_results/smoke_all_metrics.csv > /dev/null
}
run_step 6 "Aggregate metrics tables" step6

echo ""
echo "=========================================="
echo "  Smoke test complete."
echo "=========================================="
