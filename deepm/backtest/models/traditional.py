"""
Benchmark models for DeePM backtests.

Confirmed assumptions:
- `target` is aligned as next-day return: decision at t earns target for t -> t+1
- `target` is already vol-scaled (risk units), so PnL is position * target
- Runner supplies a long dataframe with columns: ['ticker', features..., 'target'] and datetime index

Each model returns a wide DataFrame (date x ticker) of positions p_{i,t}.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from sklearn.covariance import LedoitWolf
except ImportError:  # pragma: no cover
    LedoitWolf = None


# ---------------------------
# Defaults
# ---------------------------

DEFAULT_SIGN_FEATURES = ["r1y"]
DEFAULT_MACD_FEATURES = ["macd_8_24", "macd_16_48", "macd_32_96"]


# ---------------------------
# MACD position sizing
# ---------------------------

def position_sizing(x: np.ndarray) -> np.ndarray:
    """
    phi(x) = x * exp(-(x^2)/4) / 0.89
    """
    x = np.asarray(x, dtype=float)
    return x * np.exp(-(x**2) / 4.0) / 0.89


# ---------------------------
# Helpers
# ---------------------------

def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    return df


def _prepare_long_panel(
    data: pd.DataFrame,
    tickers: Sequence[str],
    start_date: str,
    required_cols: Sequence[str],
) -> pd.DataFrame:
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise KeyError(f"Missing columns in data: {missing}")

    df = _ensure_dt_index(data)
    df = df[df["ticker"].isin(tickers)].copy()
    df = df.loc[df.index>=pd.to_datetime(start_date)]

    keep = ["ticker", *required_cols]
    return df[keep].copy()


def _long_to_wide(df_long: pd.DataFrame, value_col: str, tickers: Sequence[str]) -> pd.DataFrame:
    out = (
        df_long[[value_col, "ticker"]]
        .assign(date=df_long.index)
        .set_index(["date", "ticker"])[value_col]
        .unstack("ticker")
        .sort_index()
        .reindex(columns=list(tickers))
    )
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    out.columns.name = "ticker"
    return out.fillna(0.0)


def _l1_project_rows(W: np.ndarray, max_l1: float = 1.0, eps: float = 1e-12) -> np.ndarray:
    l1 = np.sum(np.abs(W), axis=1, keepdims=True)
    scale = np.minimum(1.0, max_l1 / (l1 + eps))
    return W * scale


def _solve_erc(
    Sigma: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-8,
    damping: float = 0.5,
    eps: float = 1e-12,
) -> np.ndarray:
    """Solve Equal Risk Contribution (ERC) weights via fixed-point iteration."""
    n = Sigma.shape[0]
    q = np.ones(n, dtype=float) / n

    for _ in range(max_iter):
        s = Sigma @ q
        q_new = 1.0 / (s + eps)
        q_new = q_new / (np.sum(q_new) + eps)

        q_next = (1.0 - damping) * q + damping * q_new
        if np.max(np.abs(q_next - q)) < tol:
            q = q_next
            break
        q = q_next

    q = np.clip(q, 0.0, np.inf)
    q = q / (np.sum(q) + eps)
    return q


def _estimate_cov(window: np.ndarray, method: str) -> np.ndarray:
    if method == "ledoit_wolf":
        if LedoitWolf is None:
            raise ImportError("sklearn is required for cov_estimator='ledoit_wolf'")
        return LedoitWolf().fit(window).covariance_
    if method == "sample":
        return np.cov(window, rowvar=False)
    raise ValueError(f"Unknown cov_estimator='{method}' (use 'ledoit_wolf' or 'sample')")


# ---------------------------
# Base class
# ---------------------------

class AbstractBenchmark(ABC):
    """Base class for traditional (non-DL) benchmark strategies."""

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)

    @abstractmethod
    def predict(self, data: pd.DataFrame, tickers: List[str], start_date: str, **kwargs) -> pd.DataFrame:
        raise NotImplementedError


AbstractModel = AbstractBenchmark


# ============================================================
# A) Classical trend signals (signal-only)
# ============================================================

class LongOnly(AbstractBenchmark):
    """Constant long (+1). Under vol-scaling framework, behaves like equal-risk long."""

    def predict(self, data: pd.DataFrame, tickers: List[str], start_date: str, **kwargs) -> pd.DataFrame:
        df = _prepare_long_panel(data, tickers, start_date, required_cols=[])
        df = df.assign(position=1.0)
        return _long_to_wide(df, "position", tickers)


class SignTSMOM(AbstractBenchmark):
    """TSMOM sign rule on one or more momentum features (default: r1y)."""

    def __init__(self, features: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.features = features or list(DEFAULT_SIGN_FEATURES)

    def predict(self, data: pd.DataFrame, tickers: List[str], start_date: str, **kwargs) -> pd.DataFrame:
        df = _prepare_long_panel(data, tickers, start_date, required_cols=self.features)
        X = df[self.features].to_numpy(dtype=float)
        pos = np.nanmean(np.sign(X), axis=1)
        df = df.assign(position=np.clip(pos, -1.0, 1.0))
        return _long_to_wide(df, "position", tickers)


class MACD(AbstractBenchmark):
    """
    Multi-scale MACD baseline:
      p_{i,t} = (1/K) sum_k phi(macd_k(i,t)),  phi(x)=x exp(-x^2/4)/0.89
    """

    def __init__(self, features: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.features = features or list(DEFAULT_MACD_FEATURES)

    def predict(self, data: pd.DataFrame, tickers: List[str], start_date: str, **kwargs) -> pd.DataFrame:
        df = _prepare_long_panel(data, tickers, start_date, required_cols=self.features)
        X = df[self.features].to_numpy(dtype=float)
        pos = np.nanmean(position_sizing(X), axis=1)
        df = df.assign(position=np.clip(pos, -1.0, 1.0))
        return _long_to_wide(df, "position", tickers)


class EqualWeightSignal(AbstractBenchmark):
    """Continuous equal-weight average of selected features.

    If apply_sign is True, averages signs instead (like multi-horizon TSMOM).
    """

    def __init__(self, features: Optional[List[str]] = None, apply_sign: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.features = features or list(DEFAULT_SIGN_FEATURES)
        self.apply_sign = bool(apply_sign)

    def predict(self, data: pd.DataFrame, tickers: List[str], start_date: str, **kwargs) -> pd.DataFrame:
        df = _prepare_long_panel(data, tickers, start_date, required_cols=self.features)
        X = df[self.features].to_numpy(dtype=float)
        if self.apply_sign:
            X = np.sign(X)
        pos = np.nanmean(X, axis=1)
        df = df.assign(position=np.clip(pos, -1.0, 1.0))
        return _long_to_wide(df, "position", tickers)


# ============================================================
# B) Two-stage risk-managed scaling: signal -> cov -> scale
# ============================================================

class RiskManagedTrend(AbstractBenchmark):
    """
    Risk-managed trend baseline:
      1) Build raw signal s_t (TSMOM or MACD)
      2) Estimate cov Sigma_t on vol-scaled returns (target) using rolling window
      3) Scale p_t = (sigma_tgt / sqrt(s^T Sigma s)) * s_t
      4) Apply L1 cap and abs clip

    This is the "standard" trend + vol-risk-management two-stage baseline.
    """

    def __init__(
        self,
        signal: str = "tsmom",                 # "tsmom" | "macd" | "equal_weight"
        sign_features: Optional[List[str]] = None,
        macd_features: Optional[List[str]] = None,
        ew_features: Optional[List[str]] = None,
        cov_lookback: int = 252,
        cov_estimator: str = "ledoit_wolf",    # "ledoit_wolf" | "sample"
        ridge: float = 1e-3,
        rebalance_every: int = 1,
        sigma_tgt: float = 1.0,                # in target (risk) units
        max_gross_leverage: float = 1.0,       # L1 cap
        max_abs_pos: float = 2.0,              # hard clip per asset
        eps: float = 1e-12,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.signal = signal
        self.sign_features = sign_features or list(DEFAULT_SIGN_FEATURES)
        self.macd_features = macd_features or list(DEFAULT_MACD_FEATURES)
        self.ew_features = ew_features or ["r1d", "r1w", "r1m", "r3m", "r6m", "r1y"]

        self.cov_lookback = int(cov_lookback)
        self.cov_estimator = cov_estimator
        self.ridge = float(ridge)
        self.rebalance_every = int(rebalance_every)
        self.sigma_tgt = float(sigma_tgt)
        self.max_gross_leverage = float(max_gross_leverage)
        self.max_abs_pos = float(max_abs_pos)
        self.eps = float(eps)

    def _signal_wide(self, df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
        if self.signal == "tsmom":
            X = df[self.sign_features].to_numpy(dtype=float)
            s = np.nanmean(np.sign(X), axis=1)
            df2 = df.assign(sig=np.clip(s, -1.0, 1.0))
            return _long_to_wide(df2, "sig", tickers)

        if self.signal == "macd":
            X = df[self.macd_features].to_numpy(dtype=float)
            s = np.nanmean(position_sizing(X), axis=1)
            df2 = df.assign(sig=np.clip(s, -1.0, 1.0))
            return _long_to_wide(df2, "sig", tickers)

        if self.signal == "equal_weight":
            X = df[self.ew_features].to_numpy(dtype=float)
            s = np.nanmean(X, axis=1)
            df2 = df.assign(sig=np.clip(s, -1.0, 1.0))
            return _long_to_wide(df2, "sig", tickers)

        raise ValueError(f"Unknown signal='{self.signal}'")

    def predict(self, data: pd.DataFrame, tickers: List[str], start_date: str, **kwargs) -> pd.DataFrame:
        req = ["target"]
        if self.signal == "tsmom":
            req += self.sign_features
        elif self.signal == "macd":
            req += self.macd_features
        else:
            req += self.ew_features

        start_dt = pd.to_datetime(start_date)
        hist_start = (start_dt - pd.tseries.offsets.BDay(self.cov_lookback))
        df = _prepare_long_panel(data, tickers, hist_start, required_cols=req)

        # raw signal s_t (wide)
        sig_wide = self._signal_wide(df[["ticker"] + req[1:]].copy(), tickers)

        # vol-scaled returns for cov estimation
        ret_wide = (
            df.assign(date=df.index)
              .set_index(["date", "ticker"])["target"]
              .unstack("ticker")
              .reindex(columns=list(tickers))
              .sort_index()
        ).fillna(0.0)

        dates = ret_wide.index
        N = len(tickers)
        P = np.zeros((len(dates), N), dtype=float)
        last_p = np.zeros(N, dtype=float)

        for t_idx in range(len(dates)):
            if t_idx % self.rebalance_every != 0:
                P[t_idx] = last_p
                continue

            if t_idx < self.cov_lookback:
                P[t_idx] = 0.0
                last_p = P[t_idx]
                continue

            window = ret_wide.iloc[t_idx - self.cov_lookback : t_idx].to_numpy(dtype=float)
            Sigma = _estimate_cov(window, self.cov_estimator)
            Sigma = Sigma + self.ridge * np.eye(N)

            s = sig_wide.iloc[t_idx].to_numpy(dtype=float)

            # If all-zero signal, hold flat
            if np.all(np.abs(s) < 1e-15):
                p = np.zeros_like(s)
            else:
                # portfolio risk proxy sqrt(s^T Sigma s)
                port_var = float(s.T @ Sigma @ s)
                port_vol = np.sqrt(max(port_var, 0.0) + self.eps)

                scale = self.sigma_tgt / port_vol
                p = scale * s

                # L1 leverage cap + clip
                p = _l1_project_rows(p.reshape(1, -1), max_l1=self.max_gross_leverage)[0]
                p = np.clip(p, -self.max_abs_pos, self.max_abs_pos)

            P[t_idx] = p
            last_p = p

        return pd.DataFrame(P, index=dates, columns=list(tickers)).loc[start_dt:]


# ============================================================
# C) Two-stage allocators: MVO and Risk Parity (ERC)
# ============================================================

class TwoStageMVOTrend(AbstractBenchmark):
    """
    Two-stage Markowitz/MVO baseline:
      1) mu_t from features (default sign(r1y))
      2) Sigma_t from rolling cov on target
      3) w_raw = (Sigma + ridge I)^{-1} mu_t
      4) Scale w_raw to target portfolio volatility (sigma_tgt)
      5) Apply L1 leverage cap + abs clip
    """

    def __init__(
        self,
        mu_features: Optional[List[str]] = None,
        mu_transform: str = "sign",           # "sign" | "mean"
        cov_lookback: int = 252,
        cov_estimator: str = "ledoit_wolf",
        ridge: float = 1e-3,
        rebalance_every: int = 1,
        sigma_tgt: float = 1.0,
        tcost_penalty: float = 0.0,
        max_gross_leverage: Optional[float] = 1.0,
        max_abs_pos: float = 2.0,
        eps: float = 1e-12,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mu_features = mu_features or list(DEFAULT_SIGN_FEATURES)
        self.mu_transform = mu_transform
        self.cov_lookback = int(cov_lookback)
        self.cov_estimator = cov_estimator
        self.ridge = float(ridge)
        self.rebalance_every = int(rebalance_every)
        self.sigma_tgt = float(sigma_tgt)
        self.tcost_penalty = float(tcost_penalty)
        self.max_gross_leverage = float(max_gross_leverage) if max_gross_leverage is not None else float('inf')
        self.max_abs_pos = float(max_abs_pos)
        self.eps = float(eps)

    def _mu(self, X: np.ndarray) -> np.ndarray:
        if self.mu_transform == "sign":
            return np.nanmean(np.sign(X), axis=1)
        return np.nanmean(X, axis=1)

    def predict(self, data: pd.DataFrame, tickers: List[str], start_date: str, **kwargs) -> pd.DataFrame:
        req = list(set(["target"] + self.mu_features))
        start_dt = pd.to_datetime(start_date)
        hist_start = (start_dt - pd.tseries.offsets.BDay(self.cov_lookback))
        df = _prepare_long_panel(data, tickers, hist_start, required_cols=req)

        X = df[self.mu_features].to_numpy(dtype=float)
        mu = self._mu(X)
        mu_wide = _long_to_wide(df.assign(mu=mu), "mu", tickers)

        ret_wide = (
            df.assign(date=df.index)
              .set_index(["date", "ticker"])["target"]
              .unstack("ticker")
              .reindex(columns=list(tickers))
              .sort_index()
        ).fillna(0.0)

        dates = ret_wide.index
        N = len(tickers)
        P = np.zeros((len(dates), N), dtype=float)
        last_p = np.zeros(N, dtype=float)

        for t_idx in range(len(dates)):
            if t_idx % self.rebalance_every != 0:
                P[t_idx] = last_p
                continue

            if t_idx < self.cov_lookback:
                P[t_idx] = 0.0
                last_p = P[t_idx]
                continue

            # Check for valid signal before solving
            mu_t = mu_wide.iloc[t_idx].to_numpy(dtype=float)
            if np.all(np.abs(mu_t) < 1e-15):
                P[t_idx] = 0.0
                last_p = P[t_idx]
                continue

            window = ret_wide.iloc[t_idx - self.cov_lookback : t_idx].to_numpy(dtype=float)
            Sigma = _estimate_cov(window, self.cov_estimator)
            
            # Incorporate T-Cost Penalty (Quadratic Market Impact)
            # Solve: (Sigma + (ridge+tcost)*I) w = mu + tcost * w_prev
            effective_ridge = self.ridge + self.tcost_penalty
            Sigma_reg = Sigma + effective_ridge * np.eye(N)
            
            mu_adj = mu_t + self.tcost_penalty * last_p

            # 1. Unconstrained Tangency Portfolio
            try:
                w = np.linalg.solve(Sigma_reg, mu_adj)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(Sigma_reg, mu_adj, rcond=None)[0]

            # 2. Risk (Vol) Scaling
            # Calculate ex-ante volatility of the unscaled weights
            port_var = w.T @ Sigma @ w
            port_vol = np.sqrt(port_var)
            
            # Avoid division by zero if w is essentially zero
            if port_vol > self.eps:
                scale_factor = self.sigma_tgt / port_vol
                w = w * scale_factor

            # 3. Leverage Caps
            if self.max_gross_leverage < float('inf'):
                w = _l1_project_rows(w.reshape(1, -1), max_l1=self.max_gross_leverage)[0]
            
            w = np.clip(w, -self.max_abs_pos, self.max_abs_pos)

            P[t_idx] = w
            last_p = w

        return pd.DataFrame(P, index=dates, columns=list(tickers)).loc[start_dt:]


class TwoStageRiskParityERC(AbstractBenchmark):
    """
    Long-only risk parity (ERC) baseline:
      Sigma_t from rolling cov on target, solve ERC magnitudes q>=0,
      output p=q with L1 cap.
    """

    def __init__(
        self,
        cov_lookback: int = 252,
        cov_estimator: str = "ledoit_wolf",
        ridge: float = 1e-3,
        rebalance_every: int = 1,
        max_gross_leverage: float = 1.0,
        max_iter: int = 500,
        tol: float = 1e-8,
        damping: float = 0.5,
        eps: float = 1e-12,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cov_lookback = int(cov_lookback)
        self.cov_estimator = cov_estimator
        self.ridge = float(ridge)
        self.rebalance_every = int(rebalance_every)
        self.max_gross_leverage = float(max_gross_leverage)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.damping = float(damping)
        self.eps = float(eps)

    def predict(self, data: pd.DataFrame, tickers: List[str], start_date: str, **kwargs) -> pd.DataFrame:
        start_dt = pd.to_datetime(start_date)
        hist_start = (start_dt - pd.tseries.offsets.BDay(self.cov_lookback))
        df = _prepare_long_panel(data, tickers, hist_start, required_cols=["target"])

        ret_wide = (
            df.assign(date=df.index)
              .set_index(["date", "ticker"])["target"]
              .unstack("ticker")
              .reindex(columns=list(tickers))
              .sort_index()
        ).fillna(0.0)

        dates = ret_wide.index
        N = len(tickers)
        P = np.zeros((len(dates), N), dtype=float)
        last_p = np.zeros(N, dtype=float)

        for t_idx in range(len(dates)):
            if t_idx % self.rebalance_every != 0:
                P[t_idx] = last_p
                continue

            if t_idx < self.cov_lookback:
                P[t_idx] = 0.0
                last_p = P[t_idx]
                continue

            window = ret_wide.iloc[t_idx - self.cov_lookback : t_idx].to_numpy(dtype=float)
            Sigma = _estimate_cov(window, self.cov_estimator)
            Sigma = Sigma + self.ridge * np.eye(N)

            q = _solve_erc(Sigma, self.max_iter, self.tol, self.damping, self.eps)
            p = _l1_project_rows(q.reshape(1, -1), max_l1=self.max_gross_leverage)[0]

            P[t_idx] = p
            last_p = p

        return pd.DataFrame(P, index=dates, columns=list(tickers)).loc[start_dt:]


class TwoStageTrendRiskParity(AbstractBenchmark):
    """
    Trend direction + ERC magnitudes:
      direction = sign(TSMOM) or sign(MACD),
      magnitudes q from ERC, p = direction * q, then L1 cap + abs clip.
    """

    def __init__(
        self,
        signal: str = "tsmom",                 # "tsmom" | "macd"
        sign_features: Optional[List[str]] = None,
        macd_features: Optional[List[str]] = None,
        cov_lookback: int = 252,
        cov_estimator: str = "ledoit_wolf",
        ridge: float = 1e-3,
        rebalance_every: int = 1,
        max_gross_leverage: float = 1.0,
        max_abs_pos: float = 2.0,
        max_iter: int = 500,
        tol: float = 1e-8,
        damping: float = 0.5,
        eps: float = 1e-12,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.signal = signal
        self.sign_features = sign_features or list(DEFAULT_SIGN_FEATURES)
        self.macd_features = macd_features or list(DEFAULT_MACD_FEATURES)

        self.cov_lookback = int(cov_lookback)
        self.cov_estimator = cov_estimator
        self.ridge = float(ridge)
        self.rebalance_every = int(rebalance_every)
        self.max_gross_leverage = float(max_gross_leverage)
        self.max_abs_pos = float(max_abs_pos)

        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.damping = float(damping)
        self.eps = float(eps)

    def predict(self, data: pd.DataFrame, tickers: List[str], start_date: str, **kwargs) -> pd.DataFrame:
        req = ["target"]
        if self.signal == "tsmom":
            req += self.sign_features
        elif self.signal == "macd":
            req += self.macd_features
        else:
            raise ValueError("signal must be 'tsmom' or 'macd'")

        start_dt = pd.to_datetime(start_date)
        hist_start = (start_dt - pd.tseries.offsets.BDay(self.cov_lookback))
        df = _prepare_long_panel(data, tickers, hist_start, required_cols=req)


        # direction signal
        if self.signal == "tsmom":
            X = df[self.sign_features].to_numpy(dtype=float)
            d = np.nanmean(np.sign(X), axis=1)
        else:
            X = df[self.macd_features].to_numpy(dtype=float)
            d = np.nanmean(position_sizing(X), axis=1)

        dir_wide = _long_to_wide(df.assign(direction=np.sign(d)), "direction", tickers)

        # covariance from target
        ret_wide = (
            df.assign(date=df.index)
              .set_index(["date", "ticker"])["target"]
              .unstack("ticker")
              .reindex(columns=list(tickers))
              .sort_index()
        ).fillna(0.0)

        dates = ret_wide.index
        N = len(tickers)
        P = np.zeros((len(dates), N), dtype=float)
        last_p = np.zeros(N, dtype=float)

        for t_idx in range(len(dates)):
            if t_idx % self.rebalance_every != 0:
                P[t_idx] = last_p
                continue

            if t_idx < self.cov_lookback:
                P[t_idx] = 0.0
                last_p = P[t_idx]
                continue

            window = ret_wide.iloc[t_idx - self.cov_lookback : t_idx].to_numpy(dtype=float)
            Sigma = _estimate_cov(window, self.cov_estimator)
            Sigma = Sigma + self.ridge * np.eye(N)

            q = _solve_erc(Sigma, self.max_iter, self.tol, self.damping, self.eps)
            sgn = dir_wide.iloc[t_idx].to_numpy(dtype=float)
            p = sgn * q

            p = _l1_project_rows(p.reshape(1, -1), max_l1=self.max_gross_leverage)[0]
            p = np.clip(p, -self.max_abs_pos, self.max_abs_pos)

            P[t_idx] = p
            last_p = p

        return pd.DataFrame(P, index=dates, columns=list(tickers)).loc[start_dt:]


# ---------------------------
# Friendly aliases for YAML
# ---------------------------

TSMOM = SignTSMOM
MVO = TwoStageMVOTrend
ERC = TwoStageRiskParityERC
TrendERC = TwoStageTrendRiskParity
