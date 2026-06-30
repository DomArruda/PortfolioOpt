# -*- coding: utf-8 -*-
# main.py - entry point for the streamlit app...
"""Portfolio Optimization & Analysis — Streamlit app entry point."""
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from utils import (
    fetch_data,
    fetch_factor_data,
    get_risk_free_rate_scalar,
    get_market_cap,
)
from strategies import build_all
from metrics import (
    calculate_performance,
    factor_analysis,
    bootstrap_sharpe_ratio,
    stress_test,
)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Portfolio Optimization & Analysis",
    page_icon="📈",
    layout="wide",
)

st.title("Portfolio Optimization & Analysis")
st.caption(
    "Build several classic portfolios on a fitting window, then see how each "
    "would have performed out-of-sample on a separate backtest window."
)


# ----------------------------- Inputs -----------------------------
with st.sidebar:
    st.header("Portfolio setup")
    ticker_input = st.text_input(
        "Stock tickers (comma-separated)",
        value="AAPL, MSFT, GOOGL, AMZN",
        help="Example: AAPL, MSFT, GOOGL",
    )
    stock_tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    st.markdown("**Fitting window** — used to build the portfolios")
    analysis_start = st.date_input(
        "Fitting start", datetime(2021, 1, 1).date(), key="analysis_start"
    )
    analysis_end = st.date_input(
        "Fitting end", datetime(2023, 12, 31).date(), key="analysis_end"
    )

    st.markdown("**Backtest window** — used to evaluate out-of-sample")
    backtest_start = st.date_input(
        "Backtest start", datetime(2024, 1, 1).date(), key="backtest_start"
    )
    backtest_end = st.date_input(
        "Backtest end", datetime(2025, 12, 31).date(), key="backtest_end"
    )

    run = st.button("Run analysis", type="primary", width='stretch')

analysis_start = datetime.combine(analysis_start, datetime.min.time())
analysis_end = datetime.combine(analysis_end, datetime.min.time())
backtest_start = datetime.combine(backtest_start, datetime.min.time())
backtest_end = datetime.combine(backtest_end, datetime.min.time())


# ----------------------------- Run -----------------------------
if run:
    try:
        if not stock_tickers:
            st.warning("Enter at least one ticker to get started.")
            st.stop()
        if analysis_end <= analysis_start or backtest_end <= backtest_start:
            st.error("Each end date must come after its start date.")
            st.stop()

        with st.spinner("Downloading price data…"):
            data = fetch_data(stock_tickers, analysis_start, backtest_end).dropna(
                axis=1, how="all"
            )
        if data.empty:
            st.error(
                "No price data came back for those tickers and dates. "
                "Check the ticker symbols and try a different range."
            )
            st.stop()

        returns = data.pct_change().dropna(how="all")

        with st.expander("Price data", expanded=False):
            st.dataframe(data, width='stretch')

        analysis_returns = returns[
            (returns.index >= analysis_start) & (returns.index <= analysis_end)
        ].dropna(how="any", axis=1)
        backtest_returns = returns[
            (returns.index >= backtest_start) & (returns.index <= backtest_end)
        ].dropna(how="any", axis=1)

        common_cols = analysis_returns.columns.intersection(backtest_returns.columns)
        analysis_returns = analysis_returns[common_cols]
        backtest_returns = backtest_returns[common_cols]
        if len(common_cols) == 0:
            st.error(
                "No tickers had complete data in both windows. "
                "Try widening the date ranges or removing thinly-traded tickers."
            )
            st.stop()

        dropped = sorted(set(stock_tickers) - set(common_cols))
        if dropped:
            st.info("Skipped (missing data in one or both windows): " + ", ".join(dropped))

        # Risk-free series from Fama-French (preferred), with fallback scalar.
        factor_all = fetch_factor_data(analysis_start, backtest_end)
        if factor_all.empty or factor_all["RF"].dropna().empty:
            rf_series = None
            rf_scalar = get_risk_free_rate_scalar(analysis_start, backtest_end)
        else:
            rf_series = factor_all["RF"]
            rf_scalar = None
        rf_for_anal = rf_series if rf_scalar is None else rf_scalar
        rf_for_back = rf_series if rf_scalar is None else rf_scalar

        # Market caps (equal-weight fallback if unavailable).
        mkt_caps = pd.Series({t: get_market_cap(t) for t in common_cols}).astype("float")
        if mkt_caps.isna().all() or mkt_caps.le(0).all():
            mkt_caps = pd.Series(1.0 / len(common_cols), index=common_cols)
        else:
            mkt_caps = mkt_caps.fillna(mkt_caps.median()).clip(lower=1.0).loc[common_cols]
            mkt_caps = mkt_caps / mkt_caps.sum()

        # Build every strategy on the fitting window.
        with st.spinner("Optimizing portfolios…"):
            portfolios = build_all(
                analysis_returns,
                rf_for_anal,
                mkt_caps,
                momentum_lookback=min(252, len(analysis_returns)),
            )

        # Evaluate on the backtest window.
        results = {
            name: calculate_performance(weights, backtest_returns, rf_for_back)
            for name, weights in portfolios.items()
        }

        # --- Headline ---
        best_strategy = max(results, key=lambda x: results[x]["Total Return"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Best strategy (by total return)", best_strategy)
        c2.metric("Total return", f"{results[best_strategy]['Total Return']:.2%}")
        sharpe_val = results[best_strategy]["Sharpe Ratio"]
        c3.metric("Sharpe ratio", f"{sharpe_val:.2f}" if pd.notna(sharpe_val) else "n/a")

        # --- Cumulative performance ---
        st.header("Performance comparison")
        st.caption("Cumulative growth of $1 over the backtest window.")
        fig, ax = plt.subplots(figsize=(12, 6))
        for name, weights in portfolios.items():
            cum = (1 + (backtest_returns * weights).sum(axis=1)).cumprod()
            ax.plot(cum.index, cum, label=f'{name} ({results[name]["Total Return"]:.2%})')
        ax.set_xlabel("Date")
        ax.set_ylabel("Growth of $1")
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # --- Summary table ---
        st.header("Performance summary")
        summary_df = pd.DataFrame(results).T
        pct_cols = ["Total Return", "Annualized Return", "Volatility", "Max Drawdown"]
        ratio_cols = ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"]
        styled = summary_df.style.format(
            {**{c: "{:.2%}" for c in pct_cols}, **{c: "{:.2f}" for c in ratio_cols}}
        )
        st.dataframe(styled, width='stretch')

        # --- Best strategy weights ---
        st.subheader(f"{best_strategy} — weights")
        best_weights = portfolios[best_strategy].sort_values(ascending=False)
        weight_df = best_weights.reset_index()
        weight_df.columns = ["Stock", "Weight"]
        st.dataframe(
            weight_df.style.format({"Weight": "{:.2%}"}),
            width='stretch',
            hide_index=True,
        )

        # --- Factor analysis ---
        st.header("Factor analysis (Fama-French 3-factor)")
        ff_back = (
            factor_all[
                (factor_all.index >= backtest_start) & (factor_all.index <= backtest_end)
            ]
            if not factor_all.empty
            else pd.DataFrame()
        )

        if ff_back.empty:
            st.info("Fama-French factor data was unavailable for the backtest window.")
        else:
            tabs = st.tabs(list(portfolios.keys()))
            for tab, (name, weights) in zip(tabs, portfolios.items()):
                with tab:
                    port_ret = (backtest_returns * weights).sum(axis=1)
                    model = factor_analysis(port_ret, ff_back)
                    out = pd.DataFrame(
                        {"coef": model.params, "t": model.tvalues, "p-value": model.pvalues}
                    )
                    out.loc["R-squared", ["coef", "t", "p-value"]] = [
                        model.rsquared,
                        np.nan,
                        np.nan,
                    ]
                    st.dataframe(out.round(3), width='stretch')

        # --- Bootstrap ---
        st.header("Bootstrap Sharpe ratio (95% CI)")
        st.caption(
            "Resampled with replacement to estimate uncertainty around each "
            "strategy's Sharpe ratio."
        )
        bootstrap_rows = []
        for name, weights in portfolios.items():
            mean_s, ci = bootstrap_sharpe_ratio(backtest_returns, weights, rf_for_back)
            bootstrap_rows.append(
                {"Strategy": name, "Mean Sharpe": mean_s, "CI Lower": ci[0], "CI Upper": ci[1]}
            )
        st.dataframe(
            pd.DataFrame(bootstrap_rows).style.format(
                {"Mean Sharpe": "{:.2f}", "CI Lower": "{:.2f}", "CI Upper": "{:.2f}"}
            ),
            width='stretch',
            hide_index=True,
        )

        # --- Stress test ---
        st.header("Stress tests")
        st.caption(
            "Hypothetical total return under three scenarios applied to the "
            "backtest window."
        )
        scenarios = {
            "One-day −10% shock": "oneday_-10",
            "Returns scaled ×1.3": 1.3,
            "Returns unchanged (baseline)": 1.0,
        }
        stress_results = {
            name: stress_test(weights, backtest_returns, scenarios)
            for name, weights in portfolios.items()
        }
        st.dataframe(
            pd.DataFrame(stress_results).T.style.format("{:.2%}"),
            width='stretch',
        )

        # --- Help ---
        with st.expander("How to read the factor analysis"):
            st.markdown(
                """
Factor analysis estimates how much of your portfolio's excess return is
explained by well-known market factors.

- **Coefficients** — sensitivity to each factor. Positive means the portfolio
  tends to move with the factor; negative means against it.
- **t / p-value** — statistical significance. A common rule of thumb is
  |t| > 2 or p < 0.05.
- **R-squared** — share of return variance the model explains.
- **Factors** — Mkt-RF (market excess return), SMB (small minus big, a size
  tilt), HML (high minus low, a value tilt).

Past performance does not guarantee future results.
                """
            )

        with st.expander("What the least-correlated strategies do"):
            st.markdown(
                """
- **Min Avg Correlation** — minimizes the weighted-average pairwise
  correlation. Spreads weight toward names least correlated with the rest,
  ignoring volatility. (Heuristic; not a named portfolio in Palomar.)
- **Max Decorrelation** — the Maximum Decorrelation Portfolio (MDecP):
  minimum-variance on the correlation matrix, so only co-movement structure
  drives the weights. Palomar §6.5, eq. (6.18).
- **Max Diversification** — the Most Diversified Portfolio (MDivP) of
  Choueifaty & Coignard, maximizing the diversification ratio
  (weighted-average asset vol) / (portfolio vol). Palomar §6.5, eq. (6.17).
  Dividing MDecP weights by asset volatilities and renormalizing recovers
  this portfolio.
                """
            )

        with st.expander("References — where these algorithms come from"):
            st.markdown(
                """
Primary reference: **Daniel P. Palomar (2025), _Portfolio Optimization:
Theory and Application_, Cambridge University Press.** Free online edition:
[bookdown.org/palomar/portfoliooptimizationbook](https://bookdown.org/palomar/portfoliooptimizationbook/)
· [PDF](https://portfoliooptimizationbook.com/portfolio-optimization-book.pdf)

| Strategy | Location in the book |
|---|---|
| Equal-Weight (1/N) | §6.4 Heuristic Portfolios |
| Momentum | §6.4 (heuristics); Jegadeesh & Titman (1993) |
| Market-Cap / Black–Litterman prior | §3.6 Black–Litterman fusion |
| Minimum Variance (GMVP) | §6.5 Risk-Based Portfolios, eq. (6.16) |
| Max Decorrelation (MDecP) | §6.5, eq. (6.18) |
| Max Diversification (MDivP) | §6.5, eq. (6.17); Choueifaty & Coignard (2008) |
| Maximum Sharpe Ratio | §7.2 Maximum Sharpe Ratio Portfolio |
| Risk Parity (ERC) | Ch. 11; Maillard, Roncalli & Teiletche (2010) |
| Eigenportfolio | §3.5 factor models (PCA covariance structure) |
| Min Avg Correlation | not in the book — heuristic related to MDecP |

Factor analysis uses the Fama–French 3-factor model (Fama & French, 1993).
                """
            )

    except Exception as e:
        st.error(f"Something went wrong: {e}")
        st.error("Double-check your tickers and date ranges, then try again.")
else:
    st.info("Set your tickers and date ranges in the sidebar, then click **Run analysis**.")
