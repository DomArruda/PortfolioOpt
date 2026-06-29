# -*- coding: utf-8 -*-
"""Performance metrics, factor analysis, bootstrapping, and stress testing."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from utils import TRADING_DAYS, align_rf


def calculate_max_drawdown(returns):
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    return (cum / peak - 1.0).min()


def calculate_performance(weights, returns, risk_free):
    port = (returns * weights).sum(axis=1)
    rf = align_rf(risk_free, port.index)
    excess = port - rf

    total_return = (1.0 + port).prod() - 1.0
    years = len(port) / TRADING_DAYS if len(port) else 0.0
    annualized_return = (
        (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else np.nan
    )
    vol = port.std(ddof=1) * np.sqrt(TRADING_DAYS) if len(port) > 1 else np.nan

    s_den = excess.std(ddof=1)
    sharpe = (
        (excess.mean() / s_den) * np.sqrt(TRADING_DAYS) if s_den and s_den > 0 else np.nan
    )

    downside = excess[excess < 0]
    d_den = downside.std(ddof=1) if len(downside) > 1 else np.nan
    sortino = (
        (excess.mean() / d_den) * np.sqrt(TRADING_DAYS) if d_den and d_den > 0 else np.nan
    )

    mdd = calculate_max_drawdown(port)
    calmar = (annualized_return / abs(mdd)) if mdd and mdd != 0 else np.nan

    return {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Volatility": vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": mdd,
        "Calmar Ratio": calmar,
    }


def factor_analysis(portfolio_returns, factor_df):
    aligned = pd.concat(
        [portfolio_returns, factor_df[["Mkt-RF", "SMB", "HML", "RF"]]], axis=1
    ).dropna()
    y = aligned.iloc[:, 0] - aligned["RF"]  # excess portfolio return
    X = sm.add_constant(aligned[["Mkt-RF", "SMB", "HML"]])
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})


def bootstrap_sharpe_ratio(returns, weights, risk_free, num_simulations=2000, seed=42):
    port = (returns * weights).sum(axis=1)
    rf = align_rf(risk_free, port.index)
    ex = (port - rf).values
    n = len(ex)
    if n == 0:
        return np.nan, (np.nan, np.nan)
    rng = np.random.default_rng(seed)  # fixed seed for reproducible CIs
    vals = []
    for _ in range(num_simulations):
        sample = rng.choice(ex, size=n, replace=True)
        s = sample.std(ddof=1)
        if s and s > 0:
            vals.append((sample.mean() / s) * np.sqrt(TRADING_DAYS))
    if not vals:
        return np.nan, (np.nan, np.nan)
    return np.mean(vals), np.percentile(vals, [2.5, 97.5])


def stress_test(weights, returns, scenarios):
    port = (returns * weights).sum(axis=1)
    out = {}
    for name, sc in scenarios.items():
        if isinstance(sc, str) and sc.startswith("oneday_"):
            shock = float(sc.split("_")[1]) / 100.0  # e.g. -10 => -0.10
            stressed = port.copy()
            if len(stressed) > 0:
                stressed.iloc[0] = stressed.iloc[0] + shock
            out[name] = (1.0 + stressed).prod() - 1.0
        else:
            out[name] = (1.0 + port * float(sc)).prod() - 1.0
    return out