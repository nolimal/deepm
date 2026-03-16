import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


TARGET_MAX_ABS = 20.0
VOL_LOOKBACK = 63

MACD_PAIRS = [(4, 12), (6.5, 13.5), (8, 24), (16, 48), (32, 96)]

FEATURE_COLS = [
    "r1d", "r1w", "r1m", "r3m", "r6m", "r1y",
    "macd_4_12", "macd_6.5_13.5", "macd_8_24", "macd_16_48", "macd_32_96",
    "z_price_21d", "z_price_63d", "z_price_252d",
    "z_ret_5d", "z_ret_10d",
    "log_vol", "z_log_vol_252d",
    "skew_63d", "kurt_63d",
    "dd_63d", "dd_252d",
    "time_since_1y_high",
]


# ── Base calculations ─────────────────────────────────────────


def calc_returns(srs: pd.Series, day_offset: int = 1) -> pd.Series:
    """For each element of a pandas time-series, calculate the return over
    the past *day_offset* days.
    """
    return srs / srs.shift(day_offset) - 1.0


def calc_daily_vol(daily_returns: pd.Series) -> pd.Series:
    """Exponentially weighted daily volatility with lookback VOL_LOOKBACK."""
    return daily_returns.ewm(span=VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std()


# ── MACD signal functions ────────────────────────────────────


def _macd_halflife(timescale: float) -> float:
    return np.log(0.5) / np.log(1 - 1 / timescale)


def macd_signal(srs: pd.Series, short_timescale: float, long_timescale: float) -> pd.Series:
    """MACD signal for a single short/long timescale combination."""
    macd = (
        srs.ewm(halflife=_macd_halflife(short_timescale)).mean()
        - srs.ewm(halflife=_macd_halflife(long_timescale)).mean()
    )
    q = macd / srs.rolling(63).std()
    return q / q.rolling(252).std()


def macd_scale_signal(y):
    """Position-sizing function phi(x) = x * exp(-x^2/4) / 0.89."""
    return y * np.exp(-(y ** 2) / 4) / 0.89


def macd_combined_signal(
    srs: pd.Series,
    trend_combinations: List[Tuple[float, float]] = None,
) -> pd.Series:
    """Combined MACD signal averaged over multiple short/long pairs."""
    if trend_combinations is None:
        trend_combinations = [(4, 12), (8, 24), (16, 48), (32, 96)]
    return np.sum(
        [macd_signal(srs, s, l) for s, l in trend_combinations]
    ) / len(trend_combinations)


# ── Per-asset feature builders ────────────────────────────────


def _add_normalised_returns(df: pd.DataFrame) -> None:
    """Vol-normalised horizon returns so different horizons are comparable."""
    for col, h in [("r1d", 1), ("r1w", 5), ("r1m", 21), ("r3m", 63), ("r6m", 126), ("r1y", 252)]:
        ret_h = calc_returns(df["srs"], h)
        denom = df["daily_vol"] * np.sqrt(h)
        df[col] = ret_h / denom.replace(0.0, np.nan)


def _add_macd_features(df: pd.DataFrame) -> None:
    """Multi-scale MACD trend signals."""
    for short_w, long_w in MACD_PAIRS:
        df[f"macd_{short_w}_{long_w}"] = macd_signal(df["srs"], short_w, long_w)


def _add_zscore_features(df: pd.DataFrame) -> None:
    """Price z-scores, return z-scores, and log-vol z-scores."""
    for col, window in [("z_price_21d", 21), ("z_price_63d", 63), ("z_price_252d", 252)]:
        min_p = max(1, window // 2)
        roll_mean = df["log_price"].rolling(window, min_periods=min_p).mean()
        roll_std = df["log_price"].rolling(window, min_periods=min_p).std()
        df[col] = (df["log_price"] - roll_mean) / roll_std.replace(0.0, np.nan)

    for col, h in [("z_ret_5d", 5), ("z_ret_10d", 10)]:
        ret_h = calc_returns(df["srs"], h)
        roll_mean = ret_h.rolling(252, min_periods=63).mean()
        roll_std = ret_h.rolling(252, min_periods=63).std()
        df[col] = (ret_h - roll_mean) / roll_std.replace(0.0, np.nan)

    eps = 1e-8
    df["log_vol"] = np.log(df["daily_vol"] + eps)
    roll_mean = df["log_vol"].rolling(252, min_periods=63).mean()
    roll_std = df["log_vol"].rolling(252, min_periods=63).std(ddof=0)
    df["log_vol_252d"] = df["log_vol"]
    df["z_log_vol_252d"] = (df["log_vol"] - roll_mean) / (roll_std + eps)


def _add_shape_features(df: pd.DataFrame) -> None:
    """Rolling skewness and kurtosis of daily returns."""
    rolling_shape = df["daily_returns"].rolling(63, min_periods=21)
    df["skew_63d"] = rolling_shape.skew().clip(-3.0, 3.0)
    df["kurt_63d"] = rolling_shape.kurt().clip(1.0, 10.0)


def _add_drawdown_features(df: pd.DataFrame) -> None:
    """Vol-scaled drawdown from rolling highs and time since 1-year high."""
    dd_scale = (df["daily_vol"] * np.sqrt(252.0)).replace(0.0, np.nan)

    for col, window in [("dd_252d", 252), ("dd_63d", 63)]:
        rolling_max = df["log_price"].rolling(window, min_periods=1).max()
        df[col] = (df["log_price"] - rolling_max) / dd_scale

    rolling_max_252 = df["log_price"].rolling(252, min_periods=1).max()
    log_p = df["log_price"].values
    roll_max = rolling_max_252.values
    time_since = np.full(len(df), np.nan, dtype=np.float32)
    last_high = -1
    for i, (p, m) in enumerate(zip(log_p, roll_max)):
        if np.isnan(p) or np.isnan(m):
            continue
        if np.isclose(p, m, atol=1e-10):
            last_high = i
            time_since[i] = 0.0
        elif last_high >= 0:
            time_since[i] = float(i - last_high)
    df["time_since_1y_high"] = time_since


def _clip_features_mad(df: pd.DataFrame, cols: List[str], window: int = 252, sigma: float = 5.0) -> None:
    """Rolling MAD clipping per feature (no look-ahead). MAD scaled by 1.4826 ≈ std."""
    for col in cols:
        median = df[col].rolling(window, min_periods=1).median()
        mad = (df[col] - median).abs().rolling(window, min_periods=1).median() * 1.4826
        df[col] = df[col].clip(lower=median - sigma * mad, upper=median + sigma * mad)


def _add_target(df: pd.DataFrame) -> None:
    """Vol-targeted next-day return."""
    df["vs_factor"] = 1.0 / df["daily_vol"].replace(0.0, np.nan)
    df["target"] = calc_returns(df["srs"]).shift(-1) * df["vs_factor"]
    df["target"] = df["target"].clip(-TARGET_MAX_ABS, TARGET_MAX_ABS)


# ── Feature pipeline entry points ────────────────────────────


def deep_momentum_strategy_features(df_asset: pd.DataFrame) -> pd.DataFrame:
    """Prepare input features for deep learning model.

    Args:
        df_asset: time-series for a single asset with column 'close'.

    Returns:
        DataFrame of input features (per-asset).
    """
    df_asset = df_asset[~df_asset["close"].isna() & (df_asset["close"] > 1e-8)].copy()
    df_asset["srs"] = df_asset["close"].astype(float)

    df_asset["daily_returns"] = calc_returns(df_asset["srs"])
    df_asset["daily_vol"] = calc_daily_vol(df_asset["daily_returns"])
    df_asset["log_price"] = np.log(df_asset["srs"])

    _add_normalised_returns(df_asset)
    _add_macd_features(df_asset)
    _add_zscore_features(df_asset)
    _add_shape_features(df_asset)
    _add_drawdown_features(df_asset)
    _clip_features_mad(df_asset, FEATURE_COLS)
    _add_target(df_asset)

    return df_asset.dropna()


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for deep learning model.

    Args:
        data: Wide-format price DataFrame with DatetimeIndex and ticker columns.

    Returns:
        Long-format DataFrame with all computed features.
    """
    long = data.stack().reset_index()
    long.columns = ["date", "ticker", "close"]
    data = long.sort_values(["ticker", "date"]).reset_index(drop=True)
    mask = (data.groupby("ticker")["close"].pct_change() != 0).values
    data = pd.concat(
        [
            deep_momentum_strategy_features(group)
            for _, group in data[mask].groupby("ticker")
        ],
        ignore_index=True,
    )
    data["date"] = pd.to_datetime(data["date"])
    data.index = data["date"]
    return data


def add_cross_sectional_features(
    data: pd.DataFrame,
    cs_cols: Optional[List[str]] = None,
    date_col: str = "date",
    vol_col: str = "daily_vol",
    z_prefix: str = "cs_z_",
    pct_prefix: str = "cs_pct_",
    eps: float = 1e-8,
) -> pd.DataFrame:
    """Add cross-sectional z-scores and percentiles by date for selected columns,
    plus a cross-sectional z-score of volatility.
    """
    if cs_cols is None:
        cs_cols = ["r1m", "r3m", "r6m", "r1y", "dd_252d", "z_price_252d", "log_vol_252d"]

    df = data.copy()
    grouped = df.groupby(date_col)

    for col in cs_cols:
        z_col = f"{z_prefix}{col}"
        pct_col = f"{pct_prefix}{col}"

        df[z_col] = grouped[col].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) + eps)
        )
        df[pct_col] = grouped[col].transform(
            lambda x: x.rank(method="average") / (len(x) + 1.0)
        )

    df[f"{z_prefix}vol"] = grouped[vol_col].transform(
        lambda x: (x - x.mean()) / (x.std(ddof=0) + eps)
    )

    return df


# ── Changepoint detection features ───────────────────────────


def read_changepoint_results_and_fill_na(
    file_path: str, lookback_window_length: int
) -> pd.DataFrame:
    """Read CPD output into a DataFrame, forward-filling failed rows."""
    return (
        pd.read_csv(file_path, index_col=0, parse_dates=True)
        .ffill()
        .dropna()
        .assign(
            cp_location_norm=lambda row: (row["t"] - row["cp_location"])
            / lookback_window_length
        )
    )


def prepare_cpd_features(folder_path: str, lookback_window_length: int) -> pd.DataFrame:
    """Read CPD results for all assets into a single DataFrame."""
    return pd.concat(
        [
            read_changepoint_results_and_fill_na(
                os.path.join(folder_path, f), lookback_window_length
            ).assign(ticker=os.path.splitext(f)[0])
            for f in os.listdir(folder_path)
        ]
    )


def include_changepoint_features(
    features: pd.DataFrame, cpd_folder_name: str, lookback_window_length: int
) -> pd.DataFrame:
    """Combine changepoint features with model features."""
    features = features.merge(
        prepare_cpd_features(cpd_folder_name, lookback_window_length)[
            ["ticker", "cp_location_norm", "cp_score"]
        ]
        .rename(
            columns={
                "cp_location_norm": f"cp_rl_{lookback_window_length}",
                "cp_score": f"cp_score_{lookback_window_length}",
            }
        )
        .reset_index(),
        on=["date", "ticker"],
    )

    features.index = features["date"]

    return features
