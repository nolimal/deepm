"""Performance metrics and volatility rescaling for backtest diagnostics."""

from typing import Dict, Tuple

import numpy as np
import pandas as pd

TARGET_VOL = 0.10
TRADING_DAYS_PER_YEAR = 252


def _annualized_sharpe(returns: pd.Series) -> float:
    """Annualised Sharpe ratio from daily returns."""
    return (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(TRADING_DAYS_PER_YEAR)


def compute_hac_t_stat(returns: pd.Series) -> float:
    """Compute t-statistic for mean return != 0 using Newey-West HAC standard errors.

    Robust to heteroskedasticity and serial correlation.
    """
    data = returns.values
    T = len(data)
    if T <= 1:
        return 0.0

    mu = np.mean(data)
    demeaned = data - mu

    q = int(4 * (T / 100) ** (2 / 9))

    gamma_0 = np.sum(demeaned**2) / T

    weighted_sum = 0.0
    for k in range(1, q + 1):
        cov = np.dot(demeaned[k:], demeaned[:-k]) / T
        weight = 1 - k / (q + 1)
        weighted_sum += weight * cov

    long_run_var = gamma_0 + 2 * weighted_sum

    if long_run_var <= 0:
        return 0.0

    se_hac = np.sqrt(long_run_var / T)

    return mu / se_hac


def calculate_metrics_and_rescale(
    returns: pd.Series,
    turnover: float = None,
    avg_gmv: float = None,
    benchmark_returns: pd.Series = None,
) -> Tuple[Dict, pd.Series]:
    """Rescale returns to target volatility and calculate performance metrics.

    Args:
        returns: Daily returns (arithmetic).
        turnover: Annualised raw turnover scalar, per unit of capital.
        avg_gmv: Average raw Gross Market Value, per unit of capital.
        benchmark_returns: Benchmark returns (already rescaled) for alpha tests.

    Returns:
        Tuple of (metrics_dict, rescaled_returns_series).
    """
    ann_vol_raw = returns.std() * np.sqrt(252)
    if ann_vol_raw == 0 or np.isnan(ann_vol_raw):
        scaler = 0
    else:
        scaler = TARGET_VOL / ann_vol_raw

    rescaled_returns = returns * scaler

    ann_ret = rescaled_returns.mean() * 252

    cagr = (1 + rescaled_returns).prod() ** (252 / len(rescaled_returns)) - 1

    ann_vol = rescaled_returns.std() * np.sqrt(252)

    sharpe = (ann_ret / ann_vol) if ann_vol > 1e-6 else 0

    n_days = len(rescaled_returns)
    if n_days > 1 and ann_vol > 0:
        t_stat = rescaled_returns.mean() / (rescaled_returns.std() / np.sqrt(n_days))
    else:
        t_stat = 0.0

    t_stat_hac = compute_hac_t_stat(rescaled_returns)

    t_stat_benchmark_hac = None
    information_ratio = None
    correlation_vs_benchmark = None

    if benchmark_returns is not None:
        combined = pd.concat([rescaled_returns, benchmark_returns], axis=1).dropna()
        if not combined.empty:
            excess_returns = combined.iloc[:, 0] - combined.iloc[:, 1]

            t_stat_benchmark_hac = compute_hac_t_stat(excess_returns)

            tracking_error = excess_returns.std() * np.sqrt(252)
            if tracking_error > 0:
                information_ratio = (excess_returns.mean() * 252) / tracking_error
            else:
                information_ratio = 0.0

            correlation_vs_benchmark = combined.iloc[:, 0].corr(combined.iloc[:, 1])

    rolling_window = 63
    rolling_means = rescaled_returns.rolling(rolling_window).mean()
    rolling_stds = rescaled_returns.rolling(rolling_window).std()
    rolling_sharpes = (rolling_means / (rolling_stds + 1e-8)) * np.sqrt(252)
    worst_window_sharpe = rolling_sharpes.min()

    annual_sharpes = rescaled_returns.groupby(rescaled_returns.index.year).apply(
        _annualized_sharpe
    )
    min_annual_sharpe = annual_sharpes.min() if not annual_sharpes.empty else np.nan

    wealth_index = (1 + rescaled_returns).cumprod()
    peaks = wealth_index.cummax()
    drawdowns = (wealth_index - peaks) / peaks
    max_dd = drawdowns.min()

    calmar = (cagr / abs(max_dd)) if max_dd != 0 else 0

    var_95 = rescaled_returns.quantile(0.05)
    cvar_95 = -rescaled_returns[rescaled_returns <= var_95].mean()

    hit_rate = (rescaled_returns > 0).mean()

    metrics = {
        "CAGR": cagr,
        "Annualized Return": ann_ret,
        "Annualized Vol": ann_vol,
        "Sharpe Ratio": sharpe,
        "t-statistic": t_stat,
        "t-statistic (HAC)": t_stat_hac,
        "Worst Window Sharpe (3m)": worst_window_sharpe,
        "Minimum Annual Sharpe": min_annual_sharpe,
        "Calmar Ratio": calmar,
        "Max Drawdown": max_dd,
        "CVaR 5% (Loss)": cvar_95,
        "Hit Rate": hit_rate,
    }

    if t_stat_benchmark_hac is not None:
        metrics["t-statistic vs Passive (HAC)"] = t_stat_benchmark_hac
        metrics["Information Ratio"] = information_ratio
        metrics["Correlation vs Passive"] = correlation_vs_benchmark

    if turnover is not None:
        scaled_turnover = turnover * scaler
        metrics["Annualized Turnover (Scaled)"] = scaled_turnover

        if avg_gmv is not None and avg_gmv > 0:
            metrics["Turnover Ratio (xGMV)"] = turnover / avg_gmv
            metrics["Average Leverage (Scaled)"] = avg_gmv * scaler
            metrics["Avg Holding Period (Days)"] = (
                (2 * 252) / (turnover / avg_gmv) if turnover > 0 else 0
            )

        if scaled_turnover > 1e-6:
            breakeven_bps = (ann_ret / scaled_turnover) * 10000
        else:
            breakeven_bps = 0
        metrics["Breakeven Cost (bps)"] = breakeven_bps

    return metrics, rescaled_returns
