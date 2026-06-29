# Portfolio Optimization & Analysis

An interactive Streamlit app that builds eleven classic portfolio-construction
strategies on a fitting window, then evaluates each one out-of-sample on a
separate backtest window. It pulls price data from Yahoo Finance and factor
data from the Fama–French library, and reports performance metrics, a
Fama–French 3-factor regression, bootstrapped Sharpe-ratio confidence
intervals, and simple stress tests.

> [!IMPORTANT]
> **This project is for educational and portfolio-demonstration purposes only.
> It is not financial advice, not an investment recommendation, and not a
> solicitation to buy or sell any security.** Backtested results are historical,
> hypothetical, and do not predict future performance. Nothing here should be
> used to make real investment decisions. See [Disclaimer](#disclaimer) below.

---

## What it does

Given a list of tickers and two date ranges, the app:

1. Downloads adjusted close prices and computes daily returns.
2. Splits the data into a **fitting window** (used to construct the portfolios)
   and a separate **backtest window** (used only to evaluate them).
3. Builds eleven portfolios on the fitting window.
4. Scores each on the backtest window: cumulative growth of \$1, total and
   annualized return, volatility, Sharpe, Sortino, max drawdown, and Calmar.
5. Runs a Fama–French 3-factor regression, a bootstrap of each strategy's
   Sharpe ratio (95% CI), and three illustrative stress scenarios.

## Strategies

| Strategy | Idea | Reference (see below) |
|---|---|---|
| Equal-Weight (1/N) | Equal split across all names | Palomar §6.4 |
| Momentum | Weight by trailing return | §6.4; Jegadeesh & Titman (1993) |
| Market-Cap Weighted | Value weights / no-views Black–Litterman prior | §3.6 |
| Minimum Variance (GMVP) | Lowest-variance long-only portfolio | §6.5, eq. (6.16) |
| Maximum Sharpe Ratio | Tangency portfolio | §7.2 |
| Risk Parity (ERC) | Equal risk contribution from each asset | Ch. 11; Maillard et al. (2010) |
| Max Decorrelation (MDecP) | Min-variance on the correlation matrix | §6.5, eq. (6.18) |
| Max Diversification (MDivP) | Maximize the diversification ratio | §6.5, eq. (6.17); Choueifaty & Coignard (2008) |
| Min Avg Correlation | Minimize average pairwise correlation (heuristic) | not in the book |
| Eigenportfolio | First principal component of returns | §3.5 (factor models) |
| Hierarchical Risk Parity (HRP) | Cluster, seriate, recursively bisect — no matrix inversion | §12.3, Alg. 12.1; López de Prado (2016) |

## Project structure

```
.
├── main.py          # Streamlit UI and the fit -> evaluate flow
├── strategies.py    # All eleven portfolio constructors + build_all() registry
├── metrics.py       # Performance, factor regression, bootstrap, stress tests
├── utils.py         # Data fetching (yfinance, Fama–French) and shared helpers
├── requirements.txt
└── README.md
```

## Getting started

Requires Python 3.9+.

```bash
# (optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run main.py
```

The app opens in your browser. Enter tickers in the sidebar (e.g.
`AAPL, MSFT, GOOGL, AMZN`), set the fitting and backtest windows, and click
**Run analysis**. All four `.py` files must sit in the same directory so the
imports resolve.

> Data is fetched live from Yahoo Finance and the Fama–French data library, so
> an internet connection is required. Symbols or windows with insufficient data
> are skipped with a notice.

## Methodology notes and honest limitations

The point of this project is to demonstrate sound construction *and* honesty
about what a small backtest can and cannot show. Known limitations:

- **Single static split, not walk-forward.** Weights are fit once on the
  fitting window and held fixed across the entire backtest window. Professional
  backtests typically use rolling/expanding windows with periodic rebalancing;
  a single split is one data point and is not evidence of a durable edge.
- **No transaction costs, slippage, or rebalancing.** Real-world frictions can
  materially change which strategies are viable, especially higher-turnover
  ones.
- **Momentum uses a 12-0 window** (trailing return through the most recent day)
  rather than the textbook 12-1 convention that skips the latest month.
- **Estimation risk.** Mean–variance-style optimizers are sensitive to
  estimation error; this is precisely why robust methods like risk parity and
  HRP are included for comparison.
- **No look-ahead in the fit.** Portfolios are constructed only from
  fitting-window data and evaluated only on the held-out backtest window. The
  factor regression is explanatory (contemporaneous) by design, not predictive.

In short: this is a learning and demonstration tool, not a trading system.

## Primary reference

The algorithms follow:

> Palomar, D. P. (2025). *Portfolio Optimization: Theory and Application.*
> Cambridge University Press.

Free online edition (used for the section and equation references throughout the
code): <https://bookdown.org/palomar/portfoliooptimizationbook/>

Per-strategy section/equation citations appear in the docstrings in
`strategies.py`. Additional sources cited there include López de Prado (2016) for
HRP, Choueifaty & Coignard (2008) for maximum diversification, Maillard,
Roncalli & Teiletche (2010) for risk parity, and Fama & French (1993) for the
factor model.

These references are cited for attribution. No text, code, or figures from the
book are reproduced here; the implementations are independent.

## Disclaimer

This software is provided for educational and informational purposes only. It
does **not** constitute financial, investment, legal, or tax advice, and must
not be relied upon for any investment decision. The author is not a registered
investment adviser or broker-dealer. Past and backtested performance is not
indicative of future results, and all investing involves risk, including the
possible loss of principal. The data may be inaccurate, delayed, or incomplete.
Use at your own risk; the author accepts no liability for any loss arising from
use of this software.
