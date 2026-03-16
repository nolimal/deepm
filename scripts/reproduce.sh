#!/usr/bin/env bash
# reproduce.sh — Reproduce all main results from the paper.
#
# Prerequisites:
#   1. pip install -e .
#   2. Place raw price data at data/data_dec25.parquet
#   3. Set WANDB_ENTITY and WANDB_API_KEY environment variables
#      (or use WANDB_MODE=offline to skip wandb entirely)
#
# Usage:
#   bash scripts/reproduce.sh                # full pipeline
#   bash scripts/reproduce.sh --step N       # run a single step (1-5)
#   bash scripts/reproduce.sh --baselines    # traditional baselines only (no training)

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
# Traditional baselines list (shared)
# ──────────────────────────────────────────────
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

# ──────────────────────────────────────────────
# Step 1: Generate graph adjacency matrix
# ──────────────────────────────────────────────
run_step 1 "Generate adjacency matrix" \
    python scripts/build_graph.py

# ──────────────────────────────────────────────
# Step 2: Prepare features
# ──────────────────────────────────────────────
run_step 2 "Prepare features" \
    python scripts/prepare_features.py

# ──────────────────────────────────────────────
# --baselines: fast path (no training needed)
# ──────────────────────────────────────────────
if [ "$BASELINES_ONLY" = true ]; then
    echo ""
    echo "=========================================="
    echo "  Baselines-only mode"
    echo "=========================================="

    for cfg in "${BASELINE_CONFIGS[@]}"; do
        echo "  Backtesting: $cfg"
        python -m deepm.backtest --name "$cfg" --diagnostics
    done

    echo ""
    python scripts/aggregate_metrics.py \
        --title "Traditional Baselines" \
        --csv backtest_results/baselines_metrics.csv \
        "${BASELINE_CONFIGS[@]}"

    echo ""
    echo "Baselines complete."
    exit 0
fi

# ──────────────────────────────────────────────
# Step 3: Train models (all architectures)
# ──────────────────────────────────────────────
train_model() {
    local config=$1
    local arch=$2
    echo "  Training: $config / $arch"
    python -m deepm.training -r "$config" -a "$arch"
}

step3() {
    # DeePM-GAT (main model)
    train_model "deepm-gat" "DeePM"
    # DeePM-GCN ablation
    train_model "deepm-gcn" "DeePM"
    # Cross-Attention Only (no graph)
    train_model "deepm-no-graph" "DeePM"
    # Graph Only (no cross-attention)
    train_model "deepm-graph-only" "DeePM"
    # Independent (no structure)
    train_model "deepm-independent" "DeePM"
    # Flip graph/cross-attention order
    train_model "deepm-gat-flip" "DeePM"
    # No ReZero
    train_model "deepm-gat-no-rezero" "DeePM"
    # MACD features
    train_model "deepm-gat-macd" "DeePM"
    # Cascading lag
    train_model "deepm-gat-cascading" "DeePM"
    # No SoftMin (Pooled Only)
    train_model "deepm-gat-pooled-only" "DeePM"
    # SoftMin tau=1
    train_model "deepm-gat-softmin-t1" "DeePM"
    # SoftMin tau=0.05
    train_model "deepm-gat-softmin-t005" "DeePM"
    # Zero Cost (gamma=0)
    train_model "deepm-gat-zero-cost" "DeePM"
    # Full Cost (gamma=1)
    train_model "deepm-gat-full-cost" "DeePM"
    # Temporal baseline (Momentum Transformer)
    train_model "temporal-baseline" "AdvancedTemporalBaseline"
    train_model "temporal-baseline-zero-cost" "AdvancedTemporalBaseline"
}
run_step 3 "Train models" step3

# ──────────────────────────────────────────────
# Step 4: Run backtests
# ──────────────────────────────────────────────
backtest() {
    local config=$1
    echo "  Backtesting: $config"
    python -m deepm.backtest --name "$config" --diagnostics
}

step4() {
    # Main model + seed ablations
    backtest "bt-deepm-gat"
    backtest "bt-deepm-gat-1seed"
    backtest "bt-deepm-gat-10seed"
    backtest "bt-deepm-gat-50seed"
    backtest "bt-deepm-gat-100seed"

    # Architecture ablations
    backtest "bt-deepm-gcn"
    backtest "bt-deepm-no-graph"
    backtest "bt-deepm-graph-only"
    backtest "bt-deepm-independent"
    backtest "bt-deepm-gat-flip"
    backtest "bt-deepm-gat-no-rezero"

    # Feature ablations
    backtest "bt-deepm-gat-macd"
    backtest "bt-deepm-gat-cascading"

    # Loss ablations
    backtest "bt-deepm-gat-pooled-only"
    backtest "bt-deepm-gat-softmin-t1"
    backtest "bt-deepm-gat-softmin-t005"

    # Cost ablations
    backtest "bt-deepm-gat-zero-cost"
    backtest "bt-deepm-gat-full-cost"

    # Temporal baseline
    backtest "bt-temporal-baseline"
    backtest "bt-temporal-baseline-zero-cost"

    # Traditional baselines
    backtest "bt-baseline-longonly"
    backtest "bt-baseline-tsmom"
    backtest "bt-baseline-tsmom-rm"
    backtest "bt-baseline-tsmom-mvo"
    backtest "bt-baseline-tsmom-mvo-tp"
    backtest "bt-baseline-tsmom-erc"
    backtest "bt-baseline-macd"
    backtest "bt-baseline-macd-rm"
    backtest "bt-baseline-macd-mvo-tp"

    # Post-2020 backtests (repeat all with post2020 prefix)
    backtest "bt-post2020-deepm-gat"
    backtest "bt-post2020-deepm-gcn"
    backtest "bt-post2020-deepm-no-graph"
    backtest "bt-post2020-deepm-graph-only"
    backtest "bt-post2020-deepm-independent"
    backtest "bt-post2020-deepm-gat-flip"
    backtest "bt-post2020-deepm-gat-no-rezero"
    backtest "bt-post2020-deepm-gat-macd"
    backtest "bt-post2020-deepm-gat-cascading"
    backtest "bt-post2020-deepm-gat-pooled-only"
    backtest "bt-post2020-deepm-gat-softmin-t1"
    backtest "bt-post2020-deepm-gat-softmin-t005"
    backtest "bt-post2020-deepm-gat-zero-cost"
    backtest "bt-post2020-deepm-gat-full-cost"
    backtest "bt-post2020-temporal-baseline"
    backtest "bt-post2020-temporal-baseline-zero-cost"
    backtest "bt-post2020-baseline-longonly"
    backtest "bt-post2020-baseline-tsmom"
    backtest "bt-post2020-baseline-tsmom-rm"
    backtest "bt-post2020-baseline-tsmom-mvo"
    backtest "bt-post2020-baseline-tsmom-mvo-tp"
    backtest "bt-post2020-baseline-tsmom-erc"
    backtest "bt-post2020-baseline-macd"
    backtest "bt-post2020-baseline-macd-rm"
    backtest "bt-post2020-baseline-macd-mvo-tp"
}
run_step 4 "Run backtests" step4

# ──────────────────────────────────────────────
# Step 5: Aggregate metrics tables
# ──────────────────────────────────────────────
step5() {
    echo ""
    python scripts/aggregate_metrics.py \
        --title "Table 1: Main Results (2010–2025)" \
        bt-baseline-longonly \
        bt-baseline-tsmom \
        bt-baseline-tsmom-rm \
        bt-baseline-tsmom-mvo \
        bt-baseline-tsmom-mvo-tp \
        bt-baseline-tsmom-erc \
        bt-baseline-macd \
        bt-baseline-macd-rm \
        bt-baseline-macd-mvo-tp \
        bt-temporal-baseline-zero-cost \
        bt-temporal-baseline \
        bt-deepm-gat

    echo ""
    python scripts/aggregate_metrics.py \
        --title "Table 2: Ablation Studies (2010–2025)" \
        bt-deepm-gat \
        bt-deepm-gcn \
        bt-deepm-no-graph \
        bt-deepm-graph-only \
        bt-deepm-independent \
        bt-deepm-gat-flip \
        bt-deepm-gat-no-rezero \
        bt-deepm-gat-macd \
        bt-deepm-gat-cascading \
        bt-deepm-gat-pooled-only \
        bt-deepm-gat-softmin-t1 \
        bt-deepm-gat-softmin-t005 \
        bt-deepm-gat-zero-cost \
        bt-deepm-gat-full-cost

    echo ""
    python scripts/aggregate_metrics.py \
        --title "Table 3: Seed Sensitivity" \
        bt-deepm-gat-1seed \
        bt-deepm-gat-10seed \
        bt-deepm-gat \
        bt-deepm-gat-50seed \
        bt-deepm-gat-100seed

    echo ""
    python scripts/aggregate_metrics.py \
        --title "Table 4: Post-2020 Results" \
        bt-post2020-baseline-longonly \
        bt-post2020-baseline-tsmom \
        bt-post2020-baseline-tsmom-rm \
        bt-post2020-baseline-tsmom-mvo-tp \
        bt-post2020-baseline-tsmom-erc \
        bt-post2020-baseline-macd \
        bt-post2020-baseline-macd-rm \
        bt-post2020-baseline-macd-mvo-tp \
        bt-post2020-temporal-baseline-zero-cost \
        bt-post2020-temporal-baseline \
        bt-post2020-deepm-gat

    # Save full CSV (suppress table reprint)
    python scripts/aggregate_metrics.py --auto \
        --csv backtest_results/all_metrics.csv > /dev/null
}
run_step 5 "Aggregate metrics tables" step5

echo ""
echo "All steps complete."
