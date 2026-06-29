# -*- coding: utf-8 -*-
"""Portfolio construction strategies.

Each constructor takes a returns DataFrame (one column per asset) and returns a
weight Series indexed by asset, summing to 1 with long-only weights.

Primary reference:
    Daniel P. Palomar (2025). "Portfolio Optimization: Theory and Application."
    Cambridge University Press. Free online edition:
    https://bookdown.org/palomar/portfoliooptimizationbook/

Per-strategy section/equation references appear in each function's docstring.
The risk-based family (GMVP, Maximum Decorrelation, Most Diversified) is in
Section 6.5; heuristics (1/N) in Section 6.4; Maximum Sharpe in Section 7.2;
Risk Parity in Chapter 11; Hierarchical Risk Parity in Section 12.3.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from utils import TRADING_DAYS, as_frame, align_rf

# Shared optimizer setup --------------------------------------------------
SUM_TO_ONE = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)


def _long_only_bounds(n):
    return tuple((0, 1) for _ in range(n))


def _equal_start(n):
    return np.full(n, 1.0 / n)


def _normalize(w, columns):
    """Clip negatives, renormalize, fall back to equal weight if degenerate."""
    w = pd.Series(np.asarray(w, dtype=float), index=columns).clip(lower=0)
    return (w / w.sum()) if w.sum() > 0 else pd.Series(1.0 / len(columns), index=columns)


# Classic strategies ------------------------------------------------------
def equal_weight_portfolio(returns):
    """Equal-Weight (1/N) Portfolio.

    Palomar (2025), Section 6.4 (Heuristic Portfolios); DeMiguel, Garlappi &
    Uppal (2009). Notably shown to be a robust optimal portfolio under
    estimation error in the mean.
    """
    R = as_frame(returns)
    return pd.Series(1.0 / R.shape[1], index=R.columns)


def minimum_variance_portfolio(returns):
    """Global Minimum Variance Portfolio (GMVP).

    Palomar (2025), "Portfolio Optimization: Theory and Application",
    Section 6.5 (Risk-Based Portfolios), eq. (6.16).
    """
    R = as_frame(returns)
    n = R.shape[1]

    def obj(w):
        return (R * w).sum(axis=1).std(ddof=1) * np.sqrt(TRADING_DAYS)

    res = minimize(obj, _equal_start(n), method="SLSQP",
                   bounds=_long_only_bounds(n), constraints=SUM_TO_ONE)
    return pd.Series(res.x, index=R.columns)


def maximum_sharpe_ratio_portfolio(returns, risk_free):
    """Maximum Sharpe Ratio Portfolio (a.k.a. tangency portfolio).

    Palomar (2025), Section 7.2 (Maximum Sharpe Ratio Portfolio). The book
    details several exact solvers (bisection, Dinkelbach, Schaible); here we
    optimize the ratio directly with SLSQP.
    """
    R = as_frame(returns)
    n = R.shape[1]
    rf = align_rf(risk_free, R.index)

    def neg_sharpe(w):
        ex = (R * w).sum(axis=1) - rf
        s = ex.std(ddof=1)
        return -(ex.mean() / s) * np.sqrt(TRADING_DAYS) if s and s > 0 else 1e6

    res = minimize(neg_sharpe, _equal_start(n), method="SLSQP",
                   bounds=_long_only_bounds(n), constraints=SUM_TO_ONE)
    return pd.Series(res.x, index=R.columns)


def risk_parity_portfolio(returns):
    """Risk Parity / Equal Risk Contribution Portfolio.

    Palomar (2025), Chapter 11 (Risk Parity Portfolios); Maillard, Roncalli &
    Teiletche (2010). Minimizes squared deviation of risk contributions from
    equal (1/N) contribution.
    """
    R = as_frame(returns)
    n = R.shape[1]
    Sigma = R.cov().values * TRADING_DAYS

    def port_vol(w):
        return np.sqrt(w @ Sigma @ w)

    def risk_contrib(w):
        v = port_vol(w)
        return (w * (Sigma @ w)) / v if v > 0 else np.zeros_like(w)

    target = _equal_start(n)

    def obj(w):
        return ((risk_contrib(w) - target) ** 2).sum()

    res = minimize(obj, _equal_start(n), method="SLSQP",
                   bounds=_long_only_bounds(n), constraints=SUM_TO_ONE)
    return pd.Series(res.x, index=R.columns)


def market_cap_weighted_portfolio(returns, market_caps):
    """Market-cap (value-weighted) portfolio.

    The market portfolio of the CAPM; also the Black–Litterman prior when no
    investor views are supplied (Palomar 2025, Section 3.6 on Black–Litterman
    fusion). With no views, the BL posterior collapses to these market
    weights, which is what this returns.
    """
    w = market_caps.copy()
    if w.sum() <= 0 or w.isna().all():
        w = pd.Series(1.0 / len(returns.columns), index=returns.columns)
    return w / w.sum()


def momentum_portfolio(returns, lookback=252):
    """Cross-sectional momentum portfolio.

    Jegadeesh & Titman (1993); Grinblatt, Titman & Wermers (1995). Weights
    proportional to trailing cumulative return over the lookback window,
    long-only (negative-momentum names dropped).
    """
    R = as_frame(returns)
    lookback = min(lookback, len(R))
    mom = (1.0 + R.iloc[-lookback:]).prod() - 1.0
    return _normalize(mom, R.columns)


def eigen_portfolio(returns):
    """Eigenportfolio (first principal component of returns).

    Built from the leading eigenvector of the return (correlation) structure;
    related to statistical-factor / PCA approaches to covariance modeling
    discussed in Palomar (2025), Section 3.5 (factor models). The first PC
    sign is arbitrary, so it is oriented to a positive majority loading.
    """
    R = as_frame(returns)
    comp = PCA(n_components=1).fit(R).components_[0]
    # The first PC sign is arbitrary; orient it so the majority loading is positive.
    if np.sum(comp) < 0:
        comp = -comp
    return _normalize(comp, R.columns)


# Least-correlated family -------------------------------------------------
def min_avg_correlation_portfolio(returns):
    """Minimum average pairwise correlation portfolio.

    NOT a named portfolio in Palomar (2025); included as an intuitive
    heuristic. It is closely related to the Maximum Decorrelation Portfolio
    (see below) but minimizes the weighted-average off-diagonal correlation
    directly rather than the correlation-matrix quadratic form. Ignores
    volatility entirely (correlation matrix only).
    """
    R = as_frame(returns)
    n = R.shape[1]
    if n == 1:
        return pd.Series([1.0], index=R.columns)
    C = R.corr().values
    off = C - np.eye(n)  # zero the diagonal so self-correlation drops out

    def obj(w):
        # weighted average pairwise correlation; denom excludes the diagonal
        num = w @ off @ w
        denom = 1.0 - np.sum(w ** 2)
        return num / denom if denom > 1e-12 else num

    res = minimize(obj, _equal_start(n), method="SLSQP",
                   bounds=_long_only_bounds(n), constraints=SUM_TO_ONE)
    return pd.Series(res.x, index=R.columns)


def max_decorrelation_portfolio(returns):
    """Maximum Decorrelation Portfolio (MDecP).

    Palomar (2025), Section 6.5 (Risk-Based Portfolios), eq. (6.18). Defined
    like the GMVP but using the correlation matrix C in place of the
    covariance, so only co-movement structure (not volatility) drives the
    weights: minimize wᵀ C w  s.t.  1ᵀw = 1, w ≥ 0.
    """
    R = as_frame(returns)
    n = R.shape[1]
    if n == 1:
        return pd.Series([1.0], index=R.columns)
    C = R.corr().values

    def obj(w):
        return w @ C @ w  # "variance" under unit volatilities

    res = minimize(obj, _equal_start(n), method="SLSQP",
                   bounds=_long_only_bounds(n), constraints=SUM_TO_ONE)
    return pd.Series(res.x, index=R.columns)


def max_diversification_portfolio(returns):
    """Most Diversified Portfolio (MDivP), Choueifaty & Coignard (2008).

    Palomar (2025), Section 6.5 (Risk-Based Portfolios), eq. (6.17).
    Maximizes the diversification ratio DR = (wᵀσ) / √(wᵀΣw), which uses both
    per-asset volatilities and correlations. Equivalently, dividing the MDecP
    weights by their asset volatilities and renormalizing recovers the MDivP.
    """
    R = as_frame(returns)
    n = R.shape[1]
    if n == 1:
        return pd.Series([1.0], index=R.columns)
    Sigma = R.cov().values * TRADING_DAYS
    sigma = np.sqrt(np.diag(Sigma))  # per-asset annualized vol

    def neg_div_ratio(w):
        port_vol = np.sqrt(w @ Sigma @ w)
        if port_vol <= 1e-12:
            return 1e6
        return -(w @ sigma) / port_vol

    res = minimize(neg_div_ratio, _equal_start(n), method="SLSQP",
                   bounds=_long_only_bounds(n), constraints=SUM_TO_ONE)
    return pd.Series(res.x, index=R.columns)


# Hierarchical methods ----------------------------------------------------
def _ivp_weights(cov_slice):
    """Inverse-variance weights for a sub-covariance matrix (the allocation
    rule used within each leaf/cluster of HRP's bisection)."""
    ivp = 1.0 / np.diag(cov_slice)
    return ivp / ivp.sum()


def _cluster_variance(cov, idx):
    """Variance of the inverse-variance portfolio over the assets in `idx`.
    Used to split weight between two sub-clusters inversely to their risk."""
    sub = cov[np.ix_(idx, idx)]
    w = _ivp_weights(sub)
    return float(w @ sub @ w)


def _recursive_bisection(cov, sorted_idx):
    """Top-down recursive bisection over the quasi-diagonalized ordering.

    Walks the seriated list of assets, repeatedly splitting each contiguous
    block into two halves and dividing weight between them in inverse
    proportion to each half's cluster variance.
    """
    weights = pd.Series(1.0, index=sorted_idx)
    clusters = [sorted_idx]
    while clusters:
        # bisect every current cluster with more than one asset
        clusters = [
            half
            for cl in clusters
            for half in (cl[: len(cl) // 2], cl[len(cl) // 2:])
            if len(cl) > 1
        ]
        # process pairs (left half, right half)
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            var_left = _cluster_variance(cov, left)
            var_right = _cluster_variance(cov, right)
            alpha = 1.0 - var_left / (var_left + var_right)  # less risk -> more weight
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
    return weights


def hierarchical_risk_parity_portfolio(returns, linkage_method="single"):
    """Hierarchical Risk Parity (HRP).

    López de Prado (2016), "Building Diversified Portfolios that Outperform
    Out of Sample," Journal of Portfolio Management; Palomar (2025),
    Section 12.3 (Hierarchical Clustering-Based Portfolios), Algorithm 12.1.

    Three stages, none of which invert the covariance matrix (the source of
    instability in Markowitz-style optimizers at high dimension):
      1. Tree clustering. Correlation distance d_ij = sqrt((1 - rho_ij) / 2),
         then the "distance of distances" D~_ij = ||d_i - d_j||_2 between the
         columns of d (so assets are grouped by how similarly they relate to
         the rest of the universe), clustered with single linkage.
      2. Quasi-diagonalization: reorder assets so similar ones are adjacent.
      3. Recursive bisection: split weight top-down between sub-clusters in
         inverse proportion to their (inverse-variance) cluster variance.
    """
    R = as_frame(returns)
    n = R.shape[1]
    if n == 1:
        return pd.Series([1.0], index=R.columns)
    if n == 2:  # clustering is degenerate for two assets; use inverse variance
        cov2 = R.cov().values
        return _normalize(_ivp_weights(cov2), R.columns)

    cov = R.cov().values
    corr = R.corr().values
    # Stage 1: correlation distance, then distance-of-distances (López de Prado).
    d = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
    np.fill_diagonal(d, 0.0)
    # Euclidean distance between the columns of d.
    diff = d[:, None, :] - d[None, :, :]
    d_tilde = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(d_tilde, 0.0)
    condensed = squareform(d_tilde, checks=False)  # SciPy wants the upper triangle

    link = linkage(condensed, method=linkage_method)
    sorted_idx = list(leaves_list(link))  # Stage 2: quasi-diagonal ordering

    raw = _recursive_bisection(cov, sorted_idx)  # Stage 3
    # Map positional weights back onto the original column order.
    w = pd.Series(raw.values, index=R.columns[sorted_idx])
    return _normalize(w.reindex(R.columns), R.columns)


# Registry ----------------------------------------------------------------
def build_all(returns, risk_free, market_caps, momentum_lookback=252):
    """Construct every strategy and return a name -> weight-Series dict,
    each renormalized and reindexed to the full asset universe."""
    cols = as_frame(returns).columns
    portfolios = {
        "Equal-Weight": equal_weight_portfolio(returns),
        "Minimum Variance": minimum_variance_portfolio(returns),
        "Maximum Sharpe Ratio": maximum_sharpe_ratio_portfolio(returns, risk_free),
        "Risk Parity": risk_parity_portfolio(returns),
        "Market-Cap Weighted": market_cap_weighted_portfolio(returns, market_caps),
        "Momentum": momentum_portfolio(returns, lookback=momentum_lookback),
        "Eigenportfolio": eigen_portfolio(returns),
        "Min Avg Correlation": min_avg_correlation_portfolio(returns),
        "Max Decorrelation": max_decorrelation_portfolio(returns),
        "Max Diversification": max_diversification_portfolio(returns),
        "Hierarchical Risk Parity": hierarchical_risk_parity_portfolio(returns),
    }
    return {
        k: (w / w.sum()).reindex(cols).fillna(0.0)
        for k, w in portfolios.items()
    }