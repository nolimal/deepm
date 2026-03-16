import logging
from typing import Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --- ASSET LIST (50 Assets) ---
ASSET_LIST = [
    # Equities (9)
    "NQ1 Index", "RTY1 Index", "ES1 Index", "DM1 Index",  # US
    "Z 1 Index", "CF1 Index", "VG1 Index",               # EU (CF1 = CAC40)
    "NK1 Index", "HI1 Index",                             # APAC (HI1 = Hang Seng)

    # Rates (9)
    "FV1 Comdty", "TU1 Comdty", "TY1 Comdty", "US1 Comdty", # US Treasury
    "RX1 Comdty", "G 1 Comdty", "CN1 Comdty",             # Global
    "OE1 Comdty", "DU1 Comdty",                           # EU

    # Commodities (24)
    "CO1 Comdty", "QS1 Comdty", "XB1 Comdty", "NG1 Comdty", "CL1 Comdty", # Energy
    "PA1 Comdty", "GC1 Comdty", "SI1 Comdty", "PL1 Comdty", # Prec Metals
    "HG1 Comdty",                                         # Base Metals
    "KW1 Comdty", "C 1 Comdty", "BO1 Comdty", "SM1 Comdty", "S 1 Comdty", "W 1 Comdty", # Grains
    "CC1 Comdty", "CT1 Comdty", "JO1 Comdty", "KC1 Comdty", "SB1 Comdty", # Softs
    "FC1 Comdty", "LC1 Comdty", "LH1 Comdty",             # Livestock

    # FX (8)
    "AD1 Curncy", "BP1 Curncy", "CD1 Curncy", "EC1 Curncy", "JY1 Curncy", "SF1 Curncy", # G10
    "DX1 Curncy",                                         # DXY
    "PE1 Curncy"                                          # MXN
]

# --- MACRO GROUPS ---
MACRO_GROUPS: dict[str, Set[str]] = {
    "EQUITY_US": {"NQ1 Index", "RTY1 Index", "ES1 Index", "DM1 Index"},
    "EQUITY_EU": {"Z 1 Index", "CF1 Index", "VG1 Index"},
    "EQUITY_APAC": {"NK1 Index", "HI1 Index"},

    "RATES_US_TREASURY": {"FV1 Comdty", "TU1 Comdty", "TY1 Comdty", "US1 Comdty"},

    "RATES_EU_BUND": {"RX1 Comdty", "OE1 Comdty", "DU1 Comdty"},
    "RATES_GLOBAL_OTHER": {"G 1 Comdty", "CN1 Comdty"},

    "COMM_ENERGY": {"CO1 Comdty", "QS1 Comdty", "XB1 Comdty", "NG1 Comdty", "CL1 Comdty"},
    "COMM_PREC_METALS": {"PA1 Comdty", "GC1 Comdty", "SI1 Comdty", "PL1 Comdty"},
    "COMM_BASE_METALS": {"HG1 Comdty"},
    "COMM_AGRI_GRAINS": {"KW1 Comdty", "C 1 Comdty", "BO1 Comdty", "SM1 Comdty", "S 1 Comdty", "W 1 Comdty"},
    "COMM_AGRI_SOFTS": {"CC1 Comdty", "CT1 Comdty", "JO1 Comdty", "KC1 Comdty", "SB1 Comdty"},
    "COMM_LIVESTOCK": {"FC1 Comdty", "LC1 Comdty", "LH1 Comdty"},

    "FX_G10": {"AD1 Curncy", "BP1 Curncy", "CD1 Curncy", "EC1 Curncy", "JY1 Curncy", "SF1 Curncy", "DX1 Curncy"},
    "FX_EM": {"PE1 Curncy"}
}

# --- REGIONAL LINKS ---
REGIONAL_LINKS: dict[str, dict[str, Set[str]]] = {
    "US": {
        "equity": {"ES1 Index", "NQ1 Index", "DM1 Index", "RTY1 Index"},
        "bonds": {"US1 Comdty", "TY1 Comdty", "FV1 Comdty", "TU1 Comdty"},
        "fx": {"DX1 Curncy"}
    },
    "JP": {"equity": {"NK1 Index"}, "bonds": set(), "fx": {"JY1 Curncy"}},
    "DE/EUR": {
        "equity": {"VG1 Index", "CF1 Index"},
        "bonds": {"RX1 Comdty", "OE1 Comdty", "DU1 Comdty"},
        "fx": {"EC1 Curncy"}
    },
    "UK": {"equity": {"Z 1 Index"}, "bonds": {"G 1 Comdty"}, "fx": {"BP1 Curncy"}},
    "CA": {
        "equity": set(),
        "bonds": {"CN1 Comdty"},
        "fx": {"CD1 Curncy"}
    },
}

# Convenience Sets
RISK_FX: Set[str] = {"AD1 Curncy", "CD1 Curncy", "PE1 Curncy"}
SAFE_FX: Set[str] = {"JY1 Curncy", "SF1 Curncy"}

# --- HELPER FUNCTIONS ---
def add_links_between_sets(matrix: np.ndarray, set1: Set[str], set2: Set[str], ticker_to_index_map: dict):
    """Adds symmetric links between two sets of tickers."""
    for ticker1 in set1:
        for ticker2 in set2:
            if ticker1 in ticker_to_index_map and ticker2 in ticker_to_index_map:
                idx1, idx2 = ticker_to_index_map[ticker1], ticker_to_index_map[ticker2]
                matrix[idx1, idx2] = 1
                matrix[idx2, idx1] = 1

def add_regional_triplets(matrix: np.ndarray, regional_map: dict, ticker_to_index_map: dict):
    """Connects Equity <-> Bond <-> FX within specific regions."""
    for _, comp in regional_map.items():
        eq, bd, fx = comp.get("equity", set()), comp.get("bonds", set()), comp.get("fx", set())
        if eq and bd: add_links_between_sets(matrix, eq, bd, ticker_to_index_map)
        if eq and fx: add_links_between_sets(matrix, eq, fx, ticker_to_index_map)
        if bd and fx: add_links_between_sets(matrix, bd, fx, ticker_to_index_map)

# --- MAIN GRAPH GENERATION ---
def create_macro_story_graph(asset_list: list, macro_groups: dict, regional_links: dict) -> np.ndarray:
    """Build a binary adjacency matrix encoding macro-economic relationships between assets."""
    logger.info("Building Adjacency Matrix...")
    num_assets = len(asset_list)
    adj = np.zeros((num_assets, num_assets), dtype=np.float32)
    t2i = {ticker: i for i, ticker in enumerate(asset_list)}

    # Rule 1: Intra-group cliques
    for _, tickers in macro_groups.items():
        valid = tickers.intersection(set(asset_list))
        add_links_between_sets(adj, valid, valid, t2i)

    # Rule 2: Macro Stories
    G_ALL_EQ = macro_groups["EQUITY_US"] | macro_groups["EQUITY_EU"] | macro_groups["EQUITY_APAC"]
    BASE = macro_groups["COMM_BASE_METALS"]
    ENERGY = macro_groups["COMM_ENERGY"]
    PREC = macro_groups["COMM_PREC_METALS"]
    UST = macro_groups["RATES_US_TREASURY"]

    # 2a. Risk-on
    add_links_between_sets(adj, G_ALL_EQ, BASE, t2i)
    add_links_between_sets(adj, G_ALL_EQ, RISK_FX, t2i)
    add_links_between_sets(adj, BASE, RISK_FX, t2i)

    # 2b. Inflation
    add_links_between_sets(adj, ENERGY, UST, t2i)
    add_links_between_sets(adj, UST, PREC, t2i)
    add_links_between_sets(adj, ENERGY, PREC, t2i)

    # 2c. Safe Haven
    add_links_between_sets(adj, UST, SAFE_FX, t2i)
    add_links_between_sets(adj, PREC, SAFE_FX, t2i)

    # 2d. Equity/Rates
    add_links_between_sets(adj, G_ALL_EQ, UST, t2i)

    # 2e. Carry
    add_links_between_sets(adj, macro_groups["FX_G10"], UST, t2i)

    # 2f. Commodity Exporters
    add_links_between_sets(adj, ENERGY, {"CD1 Curncy", "PE1 Curncy"}, t2i)
    add_links_between_sets(adj, PREC, {"AD1 Curncy"}, t2i)
    add_links_between_sets(adj, ENERGY, {"JY1 Curncy"}, t2i)

    # 2g. EM Risk
    add_links_between_sets(adj, macro_groups["FX_EM"], G_ALL_EQ, t2i)

    # 2l. The "Island" Fix (USD Link)
    AGS_AND_LIVESTOCK = (
        macro_groups["COMM_AGRI_GRAINS"] |
        macro_groups["COMM_AGRI_SOFTS"] |
        macro_groups["COMM_LIVESTOCK"]
    )
    add_links_between_sets(adj, AGS_AND_LIVESTOCK, {"DX1 Curncy"}, t2i)

    # 2m. The Input Cost Fix (Energy Link)
    add_links_between_sets(adj, AGS_AND_LIVESTOCK, ENERGY, t2i)

    # 2k. Regional Triplets
    add_regional_triplets(adj, regional_links, t2i)

    # Remove self-loops
    np.fill_diagonal(adj, 0.0)

    return adj

# --- GCN NORMALIZATION ---
def create_normalized_adjacency(adj: np.ndarray) -> np.ndarray:
    """Compute the symmetric normalized adjacency matrix with self-loops."""
    # A_hat = D^-0.5 * (A + I) * D^-0.5
    A_tilde = adj + np.eye(adj.shape[0])
    D_tilde_vec = np.sum(A_tilde, axis=1)

    with np.errstate(divide='ignore'):
        D_inv_sqrt = np.power(D_tilde_vec, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0

    D_mat = np.diag(D_inv_sqrt)
    return (D_mat @ A_tilde @ D_mat).astype(np.float32)

# --- HELPER TO SAVE CSV ---
def save_matrix_to_csv(matrix: np.ndarray, asset_list: list, filename: str):
    """Saves the numpy matrix to a labeled CSV file."""
    logger.info("Saving matrix to %s...", filename)
    df = pd.DataFrame(matrix, index=asset_list, columns=asset_list)
    df.to_csv(filename)
    logger.info("Successfully saved %s", filename)

# --- EXECUTION ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate macro adjacency matrix for GNN")
    parser.add_argument("--verify", action="store_true", help="Run sanity checks on the adjacency matrix")
    args = parser.parse_args()

    raw_A = create_macro_story_graph(ASSET_LIST, MACRO_GROUPS, REGIONAL_LINKS)
    norm_A = create_normalized_adjacency(raw_A)

    if args.verify:
        t2i = {t: i for i, t in enumerate(ASSET_LIST)}

        print("\n--- VERIFYING ADJACENCY MATRIX ---")

        res = raw_A[t2i["HI1 Index"], t2i["JY1 Curncy"]]
        print(f"1. Hang Seng (HI1) <-> JPY: {'NOT connected' if res == 0 else 'CONNECTED'} (expect: NOT connected — different regions)")

        res = raw_A[t2i["DM1 Index"], t2i["ES1 Index"]]
        print(f"2. Dow (DM1) <-> S&P (ES1): {'CONNECTED' if res > 0 else 'NOT connected'} (expect: CONNECTED)")

        res = raw_A[t2i["CF1 Index"], t2i["EC1 Curncy"]]
        print(f"3. CAC40 (CF1) <-> EUR: {'CONNECTED' if res > 0 else 'NOT connected'} (expect: CONNECTED)")

    from deepm._paths import PROJECT_ROOT

    save_matrix_to_csv(raw_A, ASSET_LIST, str(PROJECT_ROOT / "macro_adjacency_raw.csv"))
    save_matrix_to_csv(norm_A, ASSET_LIST, str(PROJECT_ROOT / "macro_adjacency_normalized.csv"))

    print("\nDone.")
