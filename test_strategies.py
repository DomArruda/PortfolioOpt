# -*- coding: utf-8 -*-
"""Self-checks for strategies.py — run: python test_strategies.py"""
import numpy as np
import pandas as pd

from strategies import (
    STRATEGY_NAMES,
    build_all,
    risk_parity_portfolio,
    hierarchical_risk_parity_portfolio,
)

TRADING_DAYS = 252.0


def fake_returns(n=5, T=300, seed=0):
    rng = np.random.default_rng(seed)
    cols = [f"A{i}" for i in range(n)]
    idx = pd.date_range("2020-01-01", periods=T, freq="B")
    return pd.DataFrame(rng.normal(0.0005, 0.01, size=(T, n)), index=idx, columns=cols)


def risk_contributions(R, w):
    Sigma = R.cov().values * TRADING_DAYS
    v = float(np.sqrt(w.values @ Sigma @ w.values))
    return w.values * (Sigma @ w.values) / v


def check_risk_parity_is_actual_erc():
    R = fake_returns()
    w = risk_parity_portfolio(R)
    rc = risk_contributions(R, w) / risk_contributions(R, w).sum()
    target = np.full(len(R.columns), 1.0 / R.shape[1])
    err = np.abs(rc - target).max()
    assert err < 1e-3, f"risk contributions not equal: max deviation {err:.2e}"
    print(f"risk parity: contributions equal within {err:.2e}")


def check_build_all_subset():
    R = fake_returns()
    caps = pd.Series(1.0, index=R.columns)
    full = build_all(R, 0.0, caps)
    assert list(full) == STRATEGY_NAMES and len(full) == 11

    subset = ["Equal-Weight", "Risk Parity"]
    part = build_all(R, 0.0, caps, selected=subset)
    assert list(part) == subset, "subset order/content wrong"

    # canonical order kept even if caller passes a different order
    reordered = build_all(R, 0.0, caps, selected=["Risk Parity", "Equal-Weight"])
    assert list(reordered) == subset

    # every strategy: weights sum to 1, finite, non-negative
    for name, w in full.items():
        assert abs(w.sum() - 1.0) < 1e-9, f"{name} weights sum {w.sum()}"
        assert np.isfinite(w).all() and (w >= -1e-12).all(), f"{name} bad weights"
    print(f"build_all: 11 strategies OK, subset selection OK")


def check_small_universes():
    for n in (1, 2, 3):
        R = fake_returns(n=n)
        w = hierarchical_risk_parity_portfolio(R)
        assert abs(w.sum() - 1.0) < 1e-9 and np.isfinite(w).all()
    print("HRP: degenerate 1/2/3-asset universes OK")


if __name__ == "__main__":
    check_risk_parity_is_actual_erc()
    check_build_all_subset()
    check_small_universes()
    print("all checks passed")
