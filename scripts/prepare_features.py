#!/usr/bin/env python
"""Prepare model-ready features from raw price data.

Usage:
    python scripts/prepare_features.py --input data/data_dec25.parquet
"""

import argparse
import os

import pandas as pd

from deepm._paths import DATA_DIR
from deepm.data.build_features import add_cross_sectional_features, prepare_features


def main():
    """Parse CLI arguments and run feature preparation pipeline."""
    parser = argparse.ArgumentParser(description="Prepare features for DeePM")
    parser.add_argument(
        "--input",
        type=str,
        default=str(DATA_DIR / "data_dec25.parquet"),
        help="Path to raw price parquet file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: data/feats-{input_basename})",
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        basename = os.path.basename(args.input)
        output = str(DATA_DIR / f"feats-{basename}")

    print(f"Reading raw data from {args.input}")
    data = pd.read_parquet(args.input)
    data.index = pd.to_datetime(data.index)

    print("Computing features...")
    data = prepare_features(data)
    data = data.drop(columns="date", errors="ignore")

    print("Adding cross-sectional features...")
    data = add_cross_sectional_features(data)

    print(f"Saving to {output}")
    data.to_parquet(output)
    print(f"Done. Shape: {data.shape}")


if __name__ == "__main__":
    main()
