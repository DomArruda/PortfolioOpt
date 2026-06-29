# -*- coding: utf-8 -*-
"""Data fetching and small shared helpers."""
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader as pdr
import streamlit as st

TRADING_DAYS = 252.0


def make_naive(idx):
    """Return a tz-naive DatetimeIndex regardless of input tz state."""
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        return idx.tz_convert(None)
    return idx


def as_frame(returns):
    """Coerce a Series of returns into a single-column DataFrame."""
    return returns if isinstance(returns, pd.DataFrame) else returns.to_frame()


def align_rf(rf, idx):
    """Broadcast a risk-free input (scalar or Series) onto a target index."""
    if isinstance(rf, pd.Series):
        return rf.reindex(idx).ffill().bfill().fillna(0.0)
    return pd.Series(float(rf), index=idx)


@st.cache_data
def fetch_factor_data(start_date, end_date):
    """Daily Fama-French 3-factor data (Mkt-RF, SMB, HML, RF) as decimals."""
    try:
        ff = pdr.get_data_famafrench(
            "F-F_Research_Data_Factors_daily", start=start_date, end=end_date
        )[0]
    except Exception:
        return pd.DataFrame()
    # get_data_famafrench returns a daily PeriodIndex; convert to timestamps
    # before anything tries to treat it as a DatetimeIndex.
    if isinstance(ff.index, pd.PeriodIndex):
        ff.index = ff.index.to_timestamp()
    ff.index = make_naive(ff.index)
    return ff / 100.0


@st.cache_data
def fetch_data(tickers, start, end):
    """Return a Close-price DataFrame with one column per ticker.

    yfinance returns a flat frame for a single ticker and a MultiIndex frame
    for multiple tickers, so normalize both cases to a tidy DataFrame.
    """
    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()
        close.columns = [tickers[0] if isinstance(tickers, (list, tuple)) else tickers]

    close.index = make_naive(close.index)
    return close


@st.cache_data
def get_risk_free_rate_scalar(start, end):
    """Daily risk-free fallback if Fama-French is unavailable in the window."""
    try:
        tnx = yf.Ticker("^TNX").history(start=start, end=end)
        tnx.index = make_naive(tnx.index)
        # ^TNX quotes 10x percent: 45.00 => 4.5% => 0.045 annual
        return (tnx["Close"].iloc[-1] / 1000.0) / TRADING_DAYS
    except Exception:
        return 0.02 / TRADING_DAYS


@st.cache_data
def get_market_cap(ticker):
    """Best-effort market cap for a single ticker, or None."""
    try:
        t = yf.Ticker(ticker)
        fi = getattr(t, "fast_info", None)
        if fi is not None:
            try:
                val = fi["market_cap"]
            except (KeyError, TypeError):
                val = None
            if val and val > 0:
                return float(val)
        val = t.info.get("marketCap", None)
        return float(val) if val and val > 0 else None
    except Exception:
        return None