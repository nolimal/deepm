import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio from a series of periodic returns.

    Args:
        returns: Series of periodic (e.g. daily) returns.
        risk_free: Periodic risk-free rate (same frequency as returns).
        periods_per_year: Number of periods per year (252 for daily trading days).

    Returns:
        Annualised Sharpe ratio.
    """
    returns = returns.dropna()
    if returns.empty or returns.std() == 0:
        return 0.0

    excess = returns - risk_free
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std(ddof=1))


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised Calmar ratio from a series of periodic returns.

    Args:
        returns: Series of returns (e.g. daily), NOT price levels.
        periods_per_year: Number of periods per year.

    Returns:
        Annualised Calmar ratio (annual_return / |max_drawdown|).
    """
    returns = returns.dropna()
    if returns.empty:
        return np.nan

    # Equity curve
    equity = (1.0 + returns).cumprod()

    # Drawdown series in percent (0 at highs, negative in drawdowns)
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = drawdown.min()  # e.g. -0.25 for -25%

    # Annualised return
    annual_return = returns.mean() * periods_per_year

    if max_dd == 0:
        # No drawdowns: define Calmar as +inf if positive, else 0
        return np.inf if annual_return > 0 else 0.0

    # Use magnitude of max drawdown so sign makes sense
    calmar = annual_return / abs(max_dd)
    return float(calmar)
