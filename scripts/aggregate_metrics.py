"""Aggregate backtest metrics into paper-style results tables.

Reads ``metrics.json`` from each backtest's directory under ``backtest_diagnostics/`` and prints a
formatted table of key performance metrics.

Usage::

    # Explicit list of backtests
    python scripts/aggregate_metrics.py \\
        --title "Table 1: Main Results" \\
        bt-deepm-gat bt-baseline-tsmom bt-baseline-longonly

    # Auto-discover all diagnostics
    python scripts/aggregate_metrics.py --auto

    # Save to CSV
    python scripts/aggregate_metrics.py --auto --csv results_table.csv
"""

import argparse
import json
import os
import sys

RESULTS_DIR_DEFAULT = "backtest_diagnostics"

DISPLAY_NAMES = {
    # Main model
    "bt-deepm-gat": "DeePM-GAT (K=25)",
    "bt-smoke-deepm-gat": "DeePM-GAT",
    # Architecture ablations
    "bt-deepm-gcn": "GCN (Isotropic)",
    "bt-smoke-deepm-gcn": "GCN (Isotropic)",
    "bt-deepm-no-graph": "Cross-Attn Only",
    "bt-smoke-deepm-no-graph": "Cross-Attn Only",
    "bt-deepm-graph-only": "Graph Only",
    "bt-smoke-deepm-graph-only": "Graph Only",
    "bt-deepm-independent": "Independent",
    "bt-smoke-deepm-independent": "Independent",
    "bt-deepm-gat-flip": "Flip Graph/Cross-Attn",
    "bt-smoke-deepm-gat-flip": "Flip Graph/Cross-Attn",
    "bt-deepm-gat-no-rezero": "No ReZero",
    "bt-smoke-deepm-gat-no-rezero": "No ReZero",
    # Feature ablations
    "bt-deepm-gat-macd": "MACD Features",
    "bt-smoke-deepm-gat-macd": "MACD Features",
    "bt-deepm-gat-cascading": "Cascading Lag",
    "bt-smoke-deepm-gat-cascading": "Cascading Lag",
    # Loss ablations
    "bt-deepm-gat-pooled-only": "No SoftMin (Pooled Only)",
    "bt-smoke-deepm-gat-pooled-only": "No SoftMin (Pooled Only)",
    "bt-deepm-gat-softmin-t1": "SoftMin τ=1",
    "bt-smoke-deepm-gat-softmin-t1": "SoftMin τ=1",
    "bt-deepm-gat-softmin-t005": "SoftMin τ=0.05",
    "bt-smoke-deepm-gat-softmin-t005": "SoftMin τ=0.05",
    # Cost ablations
    "bt-deepm-gat-zero-cost": "Zero Cost (γ=0)",
    "bt-smoke-deepm-gat-zero-cost": "Zero Cost (γ=0)",
    "bt-deepm-gat-full-cost": "Full Cost (γ=1)",
    "bt-smoke-deepm-gat-full-cost": "Full Cost (γ=1)",
    # Seed ablations
    "bt-deepm-gat-1seed": "DeePM-GAT (K=1)",
    "bt-deepm-gat-10seed": "DeePM-GAT (K=10)",
    "bt-deepm-gat-50seed": "DeePM-GAT (K=50)",
    "bt-deepm-gat-100seed": "DeePM-GAT (K=100)",
    # Temporal baselines
    "bt-temporal-baseline": "Mom. Transformer (γ=0.5)",
    "bt-smoke-temporal-baseline": "Mom. Transformer (γ=0.5)",
    "bt-temporal-baseline-zero-cost": "Mom. Transformer (γ=0)",
    "bt-smoke-temporal-baseline-zero-cost": "Mom. Transformer (γ=0)",
    # Traditional baselines
    "bt-baseline-longonly": "Long Only",
    "bt-baseline-tsmom": "TSMOM",
    "bt-baseline-tsmom-rm": "Risk-Managed TSMOM",
    "bt-baseline-tsmom-mvo": "MVO Trend",
    "bt-baseline-tsmom-mvo-tp": "MVO-TP Trend",
    "bt-baseline-tsmom-erc": "ERC Trend",
    "bt-baseline-macd": "MACD",
    "bt-baseline-macd-rm": "Risk-Managed MACD",
    "bt-baseline-macd-mvo-tp": "MVO-TP MACD",
}

METRICS_COLS = [
    ("Sharpe Ratio", "Sharpe", ".2f"),
    ("Calmar Ratio", "Calmar", ".2f"),
    ("CAGR", "CAGR%", ".1f"),
    ("Annualized Vol", "Vol%", ".1f"),
    ("Max Drawdown", "MaxDD%", ".1f"),
    ("t-statistic", "t-stat", ".2f"),
    ("Hit Rate", "Hit%", ".1f"),
]


def _pct(v):
    """Convert a ratio to percentage if it looks like a ratio (|v| <= 1)."""
    if v is None:
        return None
    return v * 100 if abs(v) <= 2.0 else v


def load_metrics(name: str, results_dir: str) -> dict | None:
    """Load metrics.json for a given backtest name."""
    path = os.path.join(results_dir, name, "metrics.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_row(metrics: dict, section: str = "net") -> dict:
    """Extract key metric values from a metrics dict section."""
    m = metrics.get(section, {})
    row = {}
    for json_key, col_name, fmt in METRICS_COLS:
        val = m.get(json_key)
        if val is not None and "%" in col_name:
            val = _pct(val)
        row[col_name] = val
    return row


def format_table(
    rows: list[tuple[str, dict]],
    title: str | None = None,
) -> str:
    """Format rows as an aligned ASCII table."""
    col_names = [c for _, c, _ in METRICS_COLS]
    col_fmts = {c: f for _, c, f in METRICS_COLS}
    name_width = max(len(name) for name, _ in rows) if rows else 20
    name_width = max(name_width, 8)

    lines = []
    if title:
        lines.append("")
        lines.append(title)
        lines.append("=" * len(title))

    col_widths = {}
    for c in col_names:
        col_widths[c] = max(len(c), 8)

    header = f"{'Model':<{name_width}}  " + "  ".join(
        f"{c:>{col_widths[c]}}" for c in col_names
    )
    lines.append(header)
    lines.append("-" * len(header))

    for display_name, vals in rows:
        parts = [f"{display_name:<{name_width}}"]
        for c in col_names:
            v = vals.get(c)
            if v is None:
                parts.append(f"{'n/a':>{col_widths[c]}}")
            else:
                parts.append(f"{v:{col_widths[c]}{col_fmts[c]}}")
        lines.append("  ".join(parts))

    lines.append("")
    return "\n".join(lines)


def auto_discover(results_dir: str) -> list[str]:
    """Find all backtest names that have a metrics.json."""
    if not os.path.isdir(results_dir):
        return []
    names = []
    for d in sorted(os.listdir(results_dir)):
        if os.path.isfile(os.path.join(results_dir, d, "metrics.json")):
            names.append(d)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate backtest metrics into a results table."
    )
    parser.add_argument(
        "backtests",
        nargs="*",
        help="Backtest config names (e.g. bt-deepm-gat bt-baseline-tsmom)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-discover all backtests with metrics.json",
    )
    parser.add_argument(
        "--results_dir",
        default=RESULTS_DIR_DEFAULT,
        help="Root diagnostics directory (default: backtest_diagnostics)",
    )
    parser.add_argument("--title", default=None, help="Table title")
    parser.add_argument(
        "--section",
        default="net",
        choices=["gross", "net"],
        help="Metrics section to display (default: net)",
    )
    parser.add_argument("--csv", default=None, help="Save table as CSV to this path")
    args = parser.parse_args()

    if args.auto:
        names = auto_discover(args.results_dir)
    elif args.backtests:
        names = args.backtests
    else:
        parser.error("Provide backtest names or use --auto")

    if not names:
        print("No backtests found.", file=sys.stderr)
        sys.exit(1)

    rows = []
    missing = []
    for name in names:
        metrics = load_metrics(name, args.results_dir)
        if metrics is None:
            missing.append(name)
            continue
        display = DISPLAY_NAMES.get(name, name)
        row = extract_row(metrics, section=args.section)
        rows.append((display, row))

    if missing:
        print(
            f"Warning: metrics.json not found for: {', '.join(missing)}",
            file=sys.stderr,
        )

    if not rows:
        print("No metrics data found.", file=sys.stderr)
        sys.exit(1)

    table = format_table(rows, title=args.title)
    print(table)

    if args.csv:
        import csv

        col_names = [c for _, c, _ in METRICS_COLS]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Model"] + col_names)
            for display, vals in rows:
                writer.writerow(
                    [display] + [vals.get(c, "") for c in col_names]
                )
        print(f"Saved CSV to {args.csv}")


if __name__ == "__main__":
    main()
