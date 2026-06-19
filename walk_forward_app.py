"""
Walk-Forward Trading App — Momentum Pullback Strategy
=====================================================
Streamlit + Plotly + yfinance

Strategy:  Trade pullbacks in a confirmed uptrend, entering on momentum
           resumption rather than on a (lagging) MA crossover.
  Regime : close > SMA200 AND SMA50 > SMA200
  Arm    : while flat & in regime, price dips to the pullback EMA (low <= EMA)
  Entry  : armed AND close > previous bar's high   (trend resuming)
  Stop   : initial = entry - (atr_stop * ATR);  ratchets up via a
           chandelier trail = highest_close_since_entry - (atr_trail * ATR)
  Exit   : close < effective stop  OR  close < SMA200  OR  end of window
Optimizer: Rolling train / blind test, maximize CAGR s.t. Sharpe >= 1.2
Costs:     0.10% exchange fee + 0.05% slippage per trade (entry & exit)

Design note: the entry TRIGGER (close above prior high after a pullback)
replaces the EMA cross, and the EXIT is an ATR/chandelier trail rather than
a reverse cross — the two changes that most often fix poor crossover OOS
results (late signals + giving back winners).
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import product
from datetime import datetime
from dateutil.relativedelta import relativedelta
import json

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

EXCHANGE_FEE = 0.0010   # 0.10%
SLIPPAGE     = 0.0005   # 0.05%
COST_PER_TRADE = EXCHANGE_FEE + SLIPPAGE  # applied at entry AND exit

WARM_UP_BARS = 50
MIN_TEST_WEEKS = 6

PARAM_GRID = {
    "ema":       [10, 15, 20, 25],        # pullback EMA the price dips to
    "atr_stop":  [1.5, 2.0, 2.5, 3.0],    # initial stop  = entry - mult*ATR
    "atr_trail": [2.5, 3.0, 3.5, 4.0],    # chandelier    = hi_close - mult*ATR
}   # 4*4*4 = 64 combos/fold

# ──────────────────────────────────────────────
# INDICATOR FUNCTIONS
# ──────────────────────────────────────────────

def compute_ema(series: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average."""
    return pd.Series(series).ewm(span=span, adjust=False).mean().values


def compute_sma(series: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average."""
    return pd.Series(series).rolling(window=period, min_periods=1).mean().values


def compute_wma(series: np.ndarray, period: int) -> np.ndarray:
    """Weighted moving average (linearly weighted)."""
    weights = np.arange(1, period + 1, dtype=float)
    return pd.Series(series).rolling(window=period, min_periods=1).apply(
        lambda x: np.dot(x[-len(weights[:len(x)]):], weights[:len(x)]) / weights[:len(x)].sum(),
        raw=True
    ).values


def compute_hma(series: np.ndarray, period: int) -> np.ndarray:
    """
    Hull Moving Average = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Direction: positive when HMA[i] > HMA[i-1], negative when HMA[i] < HMA[i-1].
    """
    half  = max(int(period / 2), 1)
    sqrtn = max(int(np.sqrt(period)), 1)
    wma_half = compute_wma(series, half)
    wma_full = compute_wma(series, period)
    diff     = 2.0 * wma_half - wma_full
    return compute_wma(diff, sqrtn)


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int = 14) -> np.ndarray:
    """Wilder's Average True Range (uses Wilder smoothing = ewm alpha=1/period)."""
    h = pd.Series(high); l = pd.Series(low); c = pd.Series(close)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean().values


def pullback_signals(high: np.ndarray, low: np.ndarray, close: np.ndarray, ema_pull: int):
    """
    All indicators needed for the momentum-pullback strategy.
      ema_pull_arr : the EMA that price pulls back to (the entry zone)
      sma50, sma200: trend filter  (uptrend = close > sma200 AND sma50 > sma200)
      atr          : Wilder ATR(14) for stop and chandelier-trail sizing
    Returns (ema_pull_arr, sma50, sma200, atr)
    """
    ema_pull_arr = compute_ema(close, ema_pull)
    sma50        = compute_sma(close, 50)
    sma200       = compute_sma(close, 200)
    atr          = compute_atr(high, low, close, 14)
    return ema_pull_arr, sma50, sma200, atr

# ──────────────────────────────────────────────
# BACKTEST ENGINE
# ──────────────────────────────────────────────

def backtest(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             ema_pull: np.ndarray, sma50: np.ndarray, sma200: np.ndarray,
             atr: np.ndarray, atr_stop: float, atr_trail: float,
             start_idx: int = 0) -> tuple:
    """
    Momentum-Pullback strategy (long-only). All signals & fills at today's close.

    Regime : close > SMA200 AND SMA50 > SMA200  (confirmed uptrend)
    Arm    : while flat & in regime, a bar whose low tags the pullback EMA
             (low <= ema_pull) arms a setup.
    Entry  : armed AND close > previous bar's high  (momentum resuming).
    Stop   : initial   = entry - atr_stop  * ATR[entry]
             chandelier = highest_close_since_entry - atr_trail * ATR[i]
             effective  = max(initial, chandelier)   (ratchets up only)
    Exit   : close < effective stop  OR  close < SMA200  OR  end of window.
    Force-close at last bar of every window.
    Returns (daily_ret, trade_count, trade_records)
    """
    n = len(close)
    daily_ret     = np.zeros(n)
    in_position   = False
    armed         = False
    trades        = 0
    trade_records = []
    entry_idx = entry_price = None
    init_stop = hi_close = None

    i0 = max(start_idx, 1)
    for i in range(i0, n):
        uptrend = (close[i] > sma200[i]) and (sma50[i] > sma200[i])

        if not in_position:
            if not uptrend:
                armed = False
            else:
                if low[i] <= ema_pull[i]:        # price tagged the pullback zone
                    armed = True
                if armed and close[i] > high[i - 1]:   # trend resuming → enter
                    in_position = True
                    entry_idx   = i
                    entry_price = close[i]
                    init_stop   = close[i] - atr_stop * atr[i]
                    hi_close    = close[i]
                    daily_ret[i] -= COST_PER_TRADE
                    trades += 1
                    armed = False
        else:
            daily_ret[i] = (close[i] / close[i - 1]) - 1.0
            hi_close   = max(hi_close, close[i])
            trail_stop = hi_close - atr_trail * atr[i]
            eff_stop   = max(init_stop, trail_stop)
            if (close[i] < eff_stop) or (close[i] < sma200[i]):
                in_position = False
                daily_ret[i] -= COST_PER_TRADE
                trade_records.append((entry_idx, i, entry_price, close[i]))
                trades += 1
                entry_idx = entry_price = init_stop = hi_close = None

    # Force-close at end of window
    if in_position:
        daily_ret[-1] -= COST_PER_TRADE
        trade_records.append((entry_idx, n - 1, entry_price, close[n - 1]))
        trades += 1

    return daily_ret, trades, trade_records

# ──────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────

def sharpe_ratio(returns: np.ndarray) -> float:
    """Annualized Sharpe, rf = 0."""
    if len(returns) < 2:
        return 0.0
    s = np.std(returns, ddof=1)
    if s < 1e-12:
        return 0.0
    return (np.mean(returns) / s) * np.sqrt(252)


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Max drawdown as a negative fraction."""
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / np.where(peak > 0, peak, 1e-10)
    return float(np.min(dd))


def cagr(equity_curve: np.ndarray, trading_days: int) -> float:
    """Compound annual growth rate."""
    if trading_days < 1 or equity_curve[0] <= 0:
        return 0.0
    total = equity_curve[-1] / equity_curve[0]
    years = trading_days / 252
    if years < 0.01:
        return 0.0
    return total ** (1 / years) - 1


def win_rate(returns: np.ndarray) -> float:
    """Fraction of positive-return days among days with nonzero returns."""
    active = returns[returns != 0]
    if len(active) == 0:
        return 0.0
    return float(np.sum(active > 0) / len(active))


def regime_filtered_bah(df: pd.DataFrame, rf_annual_pct: float = 4.0,
                        sma_period: int = 200) -> dict:
    """
    Benchmark: hold the instrument while it closed above its SMA(200) yesterday,
    otherwise sit in cash earning the risk-free rate. Signal is lagged one day
    (decide on yesterday's close, act today) to avoid look-ahead. A round-trip
    cost is charged on each regime switch.
    Returns metrics + the daily equity curve for plotting.
    """
    close = df["Close"].values
    n = len(close)
    sma = compute_sma(close, sma_period)
    rf_daily = (1.0 + rf_annual_pct / 100.0) ** (1.0 / 252.0) - 1.0

    daily = np.zeros(n)
    in_mkt_prev = False
    switches = 0
    days_in = 0
    for i in range(1, n):
        in_mkt = close[i - 1] > sma[i - 1]      # lagged signal (no look-ahead)
        if in_mkt:
            daily[i] = close[i] / close[i - 1] - 1.0
            days_in += 1
        else:
            daily[i] = rf_daily
        if in_mkt != in_mkt_prev:
            daily[i] -= COST_PER_TRADE
            switches += 1
        in_mkt_prev = in_mkt

    eq = (1.0 + daily).cumprod()
    return {
        "_equity":   eq,
        "_returns":  daily,
        "total_pct": round((eq[-1] - 1) * 100, 1),
        "cagr_pct":  round(cagr(eq, n) * 100, 2),
        "maxdd_pct": round(max_drawdown(eq) * 100, 2),
        "sharpe":    round(sharpe_ratio(daily), 2),
        "pct_time_in_market": round(100 * days_in / max(n - 1, 1), 1),
        "num_switches": switches,
        "cash_yield_pct": rf_annual_pct,
    }


def full_metrics(daily_returns: np.ndarray, total_trades: int) -> dict:
    """
    Returns dict with both raw float values and formatted display strings.
    NOTE: Total Return is raw and window-length-dependent.
          CAGR is annualized and is the fair apples-to-apples comparison
          between IS (12-month) and OOS (3-month) windows.
    """
    eq = (1.0 + daily_returns).cumprod()
    total_ret  = (eq[-1] - 1) * 100
    cagr_val   = cagr(eq, len(eq)) * 100
    sharpe_val = sharpe_ratio(daily_returns)
    mdd_val    = max_drawdown(eq) * 100
    wr_val     = win_rate(daily_returns) * 100

    return {
        # Raw floats for math
        "_total_return": total_ret,
        "_sharpe":       sharpe_val,
        "_cagr":         cagr_val,
        "_max_dd":       mdd_val,
        "_win_rate":     wr_val,
        "_trades":       total_trades,
        # Formatted strings for display
        "Total Return":  f"{total_ret:+.1f}% *",   # asterisk = not normalized
        "CAGR":          f"{cagr_val:+.1f}%",
        "Sharpe Ratio":  f"{sharpe_val:.2f}",
        "Max Drawdown":  f"{mdd_val:.1f}%",
        "Win Rate":      f"{wr_val:.0f}%",
        "Trades":        str(total_trades),
    }

# ──────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────

WARMUP_DAYS = 220   # trading days needed before start date (200 for SMA + buffer)

@st.cache_data(show_spinner=False)
def fetch_data(ticker: str, start: str, end: str) -> tuple:
    """
    Returns (df, df_full) where:
      df      — user's requested date range (all analysis/Gantt/equity curves)
      df_full — warmup prepended to df so SMA200 & EMAs are pre-settled from day 1

    auto_adjust=True: yfinance retroactively restates all historical Close prices
    to account for stock splits and dividends, so a 4:1 split doesn't create a
    false 75% price drop in the backtest. This is the correct price series for
    any technical indicator or return calculation.
    """
    from dateutil.relativedelta import relativedelta as rd
    import pandas as pd

    start_dt     = pd.Timestamp(start)
    warmup_start = start_dt - rd(days=int(WARMUP_DAYS * 2.0))

    raw = yf.download(ticker, start=warmup_start.strftime("%Y-%m-%d"),
                      end=end, progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()

    df_full = raw.copy()
    df      = raw[raw.index >= start_dt].copy()

    return df, df_full

# ──────────────────────────────────────────────
# WALK-FORWARD ENGINE
# ──────────────────────────────────────────────

def _generate_folds(df: pd.DataFrame, train_m: int, test_m: int):
    """Yield (fold_num, train_slice, test_slice, train_start, train_end, test_end).
    
    Operates only on df (user's date range). SMA200/EMA warm-up is satisfied
    by df_full which contains pre-start-date history — no bars consumed here.
    """
    MIN_TRAIN_BARS = 20   # minimum scoreable bars per fold after warmup is external
    start  = df.index[0]
    end    = df.index[-1]
    cursor = start
    fold   = 0

    while True:
        train_end = cursor + relativedelta(months=train_m)
        test_end  = train_end + relativedelta(months=test_m)

        if train_end >= end:
            break

        if test_end > end:
            test_end = end
        test_days = (test_end - train_end).days
        if test_days < MIN_TEST_WEEKS * 7:
            break

        train_df = df[(df.index >= cursor) & (df.index < train_end)]
        test_df  = df[(df.index >= train_end) & (df.index < test_end)]

        if len(train_df) < MIN_TRAIN_BARS or len(test_df) < 5:
            cursor += relativedelta(months=test_m)
            continue

        yield fold, train_df, test_df, cursor, train_end, test_end
        fold   += 1
        cursor += relativedelta(months=test_m)


def _optimize_on_train(train_df, df_full):
    """
    Grid-search over PARAM_GRID.
    Objective: maximize CAGR subject to Sharpe >= 1.2.
    If no combo meets Sharpe >= 1.2, falls back to best Sharpe overall.
    """
    SHARPE_MIN = 1.2

    best_cagr        = -np.inf
    best_params_main = None   # meets Sharpe >= 1.2

    best_sharpe_fallback = -np.inf
    best_params_fallback = None   # best Sharpe if constraint never met

    keys   = list(PARAM_GRID.keys())
    combos = list(product(*PARAM_GRID.values()))

    train_start  = train_df.index[0]
    full_idx     = df_full.index.get_loc(train_start)
    warmup_start = max(0, full_idx - WARMUP_DAYS)
    extended     = df_full.iloc[warmup_start : full_idx + len(train_df)]
    high         = extended["High"].values
    low          = extended["Low"].values
    close        = extended["Close"].values
    warmup       = full_idx - warmup_start

    for combo in combos:
        p = dict(zip(keys, combo))

        ema_pull, sma50, sma200, atr = pullback_signals(high, low, close, p["ema"])
        ret, trades, _ = backtest(high, low, close, ema_pull, sma50, sma200, atr,
                                  p["atr_stop"], p["atr_trail"], start_idx=warmup)
        if trades < 2:
            continue

        active_ret = ret[warmup:]
        sr = sharpe_ratio(active_ret)
        eq = (1 + active_ret).cumprod()
        cg = cagr(eq, len(active_ret))

        # Primary: max CAGR where Sharpe >= 1.2
        if sr >= SHARPE_MIN and cg > best_cagr:
            best_cagr        = cg
            best_params_main = p

        # Fallback: best Sharpe regardless
        if sr > best_sharpe_fallback:
            best_sharpe_fallback = sr
            best_params_fallback = p

    if best_params_main is not None:
        return best_params_main, best_sharpe_fallback
    else:
        # No combo met Sharpe >= 1.2 — use best Sharpe available
        return best_params_fallback, best_sharpe_fallback


def _evaluate_with_lookback(df_full, window_df, params):
    """
    Evaluate a window with indicators pre-calculated from before start date.
    Returns (window_ret, trades, window_dates, trade_records_dated)
    trade_records_dated: list of dicts with open/close dates and prices.
    """
    window_start = window_df.index[0]
    full_idx     = df_full.index.get_loc(window_start)
    warmup_start = max(0, full_idx - WARMUP_DAYS)

    extended = df_full.iloc[warmup_start : full_idx + len(window_df)]
    high     = extended["High"].values
    low      = extended["Low"].values
    close    = extended["Close"].values
    dates    = extended.index
    warmup   = full_idx - warmup_start

    ema_pull, sma50, sma200, atr = pullback_signals(high, low, close, params["ema"])
    ret, trades, raw_records = backtest(high, low, close, ema_pull, sma50, sma200, atr,
                                        params["atr_stop"], params["atr_trail"], start_idx=warmup)

    # Convert index-based records to dated records
    trade_records_dated = []
    for (ei, xi, ep, xp) in raw_records:
        open_date  = dates[ei]
        close_date = dates[xi]
        days_open  = (close_date - open_date).days
        pnl_pct    = (xp / ep - 1 - 2 * COST_PER_TRADE) * 100
        trade_records_dated.append({
            "open_date":  open_date,
            "close_date": close_date,
            "days_open":  days_open,
            "buy_price":  round(ep, 2),
            "sell_price": round(xp, 2),
            "pnl_pct":    round(pnl_pct, 2),
        })

    window_ret   = ret[warmup:]
    window_dates = window_df.index[:len(window_ret)]

    return window_ret, trades, window_dates, trade_records_dated


def run_walk_forward(df, df_full, train_m=12, test_m=3, progress_bar=None):
    """Full walk-forward loop. Returns everything the dashboard needs."""
    folds = list(_generate_folds(df, train_m, test_m))
    total = len(folds)
    if total == 0:
        return None

    gantt_rows    = []
    fold_results  = []
    is_ret_list   = []
    is_date_list  = []
    oos_ret_list  = []
    oos_date_list = []
    all_trades    = []   # unified trade log with IS/OOS tag
    regime_folds  = []   # (regime_label, oos_ret_array, oos_dates) for per-regime stats

    # Regime classification series (on the instrument's own price)
    _sma50_full  = pd.Series(compute_sma(df_full["Close"].values, 50),  index=df_full.index)
    _sma200_full = pd.Series(compute_sma(df_full["Close"].values, 200), index=df_full.index)

    def _regime_at(date):
        c   = df_full["Close"].asof(date)
        s50 = _sma50_full.asof(date)
        s200 = _sma200_full.asof(date)
        if np.isnan(s200) or np.isnan(s50):
            return "unknown"
        if c > s200 and s50 > s200:
            return "bull"
        if c < s200:
            return "bear"
        return "chop"   # above 200 but 50<200, or mixed

    for i, (fnum, train_df, test_df, t_start, t_end, te_end) in enumerate(folds):
        best_params, is_sharpe = _optimize_on_train(train_df, df_full)
        if best_params is None:
            if progress_bar:
                progress_bar.progress((i + 1) / total)
            continue

        is_ret, is_trades, is_dates, is_recs   = _evaluate_with_lookback(df_full, train_df, best_params)
        oos_ret, oos_trades, oos_dates, oos_recs = _evaluate_with_lookback(df_full, test_df, best_params)

        for rec in is_recs:
            all_trades.append({**rec, "window": "IS", "fold": fnum + 1})
        for rec in oos_recs:
            all_trades.append({**rec, "window": "OOS", "fold": fnum + 1})

        is_ret_list.append(is_ret)
        is_date_list.append(is_dates)
        oos_ret_list.append(oos_ret)
        oos_date_list.append(oos_dates)

        # Per-fold metrics (evaluated, for the analysis export / stability check)
        def _fold_stats(ret):
            eq = (1 + ret).cumprod() if len(ret) else np.array([1.0])
            return {
                "cagr":   round(cagr(eq, len(ret)) * 100, 2),
                "sharpe": round(sharpe_ratio(ret), 2),
                "maxdd":  round(max_drawdown(eq) * 100, 2),
            }
        is_fs  = _fold_stats(is_ret)
        oos_fs = _fold_stats(oos_ret)

        fold_regime = _regime_at(test_df.index[0])
        regime_folds.append((fold_regime, np.asarray(oos_ret), oos_dates))

        fold_results.append({
            "fold": fnum + 1,
            "params": best_params,
            "regime": fold_regime,
            "is_sharpe": is_sharpe,
            "is_trades": is_trades,
            "oos_trades": oos_trades,
            "train_start": t_start.strftime("%Y-%m-%d"),
            "train_end":   t_end.strftime("%Y-%m-%d"),
            "test_end":    te_end.strftime("%Y-%m-%d"),
            "is_cagr":     is_fs["cagr"],
            "is_sharpe_eval": is_fs["sharpe"],
            "is_maxdd":    is_fs["maxdd"],
            "oos_cagr":    oos_fs["cagr"],
            "oos_sharpe":  oos_fs["sharpe"],
            "oos_maxdd":   oos_fs["maxdd"],
        })

        ema_p  = best_params["ema"]
        a_stop = best_params["atr_stop"]
        a_trl  = best_params["atr_trail"]
        label  = f"Fold {fnum+1}  EMA{ema_p}  stop{a_stop:g}×/trail{a_trl:g}×"
        gantt_rows.append(dict(
            Task=label,
            Type="Training (IS)",
            Start=t_start, Finish=t_end
        ))
        gantt_rows.append(dict(
            Task=label,
            Type="Testing (OOS)",
            Start=t_end, Finish=te_end
        ))

        if progress_bar:
            progress_bar.progress((i + 1) / total)

    if not fold_results:
        return None

    # ---- Stitch equity curves ----
    def stitch(ret_list, date_list):
        all_ret = np.concatenate(ret_list)
        all_dates = np.concatenate(date_list)
        # Sort by date to handle any overlap
        order = np.argsort(all_dates)
        return all_ret[order], pd.DatetimeIndex(all_dates[order])

    is_stitched, is_dates  = stitch(is_ret_list, is_date_list)
    oos_stitched, oos_dates = stitch(oos_ret_list, oos_date_list)

    is_equity  = (1 + is_stitched).cumprod()
    oos_equity = (1 + oos_stitched).cumprod()

    is_total_trades  = sum(f["is_trades"]  for f in fold_results)
    oos_total_trades = sum(f["oos_trades"] for f in fold_results)

    return {
        "folds":        fold_results,
        "gantt":        pd.DataFrame(gantt_rows),
        "is_equity":    is_equity,
        "is_dates":     is_dates,
        "is_returns":   is_stitched,
        "is_trades":    is_total_trades,
        "oos_equity":   oos_equity,
        "oos_dates":    oos_dates,
        "oos_returns":  oos_stitched,
        "oos_trades":   oos_total_trades,
        "all_trades":   sorted(all_trades, key=lambda x: x["open_date"]),
        "_regime_folds": regime_folds,   # internal: per-regime OOS stats (not serialized raw)
    }

# ──────────────────────────────────────────────
# PLOTLY CHARTS
# ──────────────────────────────────────────────

COLORS = {
    "is":  "#636EFA",   # blue
    "oos": "#EF553B",   # orange-red
    "bg":  "#0E1117",
    "grid":"#1E2530",
    "text":"#FAFAFA",
}

def _chart_layout(fig, title=""):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        title=dict(text=title, font=dict(size=20)),
        font=dict(color=COLORS["text"]),
        xaxis=dict(gridcolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=30, t=80, b=40),
    )
    return fig


def gantt_chart(gantt_df):
    import plotly.express as px

    color_map = {
        "Training (IS)": COLORS["is"],
        "Testing (OOS)": "#FF8C00",
    }

    # px.timeline needs string or datetime columns
    df = gantt_df.copy()
    df["Start"]  = pd.to_datetime(df["Start"])
    df["Finish"] = pd.to_datetime(df["Finish"])

    fig = px.timeline(
        df,
        x_start="Start",
        x_end="Finish",
        y="Task",
        color="Type",
        color_discrete_map=color_map,
        hover_data={"Start": "|%Y-%m-%d", "Finish": "|%Y-%m-%d", "Type": True, "Task": False},
    )

    fig.update_yaxes(autorange="reversed", title="")
    fig.update_xaxes(title="Date")
    _chart_layout(fig, "Walk-Forward Windows")
    fig.update_layout(
        height=max(300, len(gantt_df) // 2 * 38 + 120),
        legend=dict(title="", orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def equity_chart(res, bah_dates, bah_equity):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=res["is_dates"], y=res["is_equity"],
        mode="lines", name="Retail In-Sample (Illusion)",
        line=dict(color=COLORS["is"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=res["oos_dates"], y=res["oos_equity"],
        mode="lines", name="Walk-Forward OOS (Reality)",
        line=dict(color="#FF8C00", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=bah_dates, y=bah_equity,
        mode="lines", name="Buy & Hold SPY",
        line=dict(color="#00CC96", width=1.5, dash="dot"),
    ))
    fig.add_hline(y=1.0, line_dash="dot", line_color="#555", line_width=1)
    _chart_layout(fig, "Illusion vs. Reality — Equity Curves")
    fig.update_layout(
        yaxis_title="Growth of $1",
        xaxis_title="Date",
        height=500,
    )
    return fig

# ──────────────────────────────────────────────
# STREAMLIT UI
# ──────────────────────────────────────────────

def build_analysis_export(result, *, ticker, start, end, train_m, test_m,
                          is_metrics, oos_metrics, bah_total, bah_cagr, bah_maxdd,
                          regime_bench=None):
    """
    Build a compact, self-describing JSON blob of a walk-forward run that can be
    pasted into a chat for diagnosis. Captures config, IS/OOS aggregates,
    degradation, benchmarks (raw B&H + regime-filtered B&H), per-fold params +
    per-fold OOS metrics, per-regime OOS breakdown, and round-trip trade summaries.
    """
    import collections

    folds = result["folds"]

    # Parameter stability: how often each value was chosen across folds
    stability = {}
    for k in PARAM_GRID.keys():
        cnt = collections.Counter(f["params"][k] for f in folds)
        stability[k] = {str(v): c for v, c in sorted(cnt.items())}

    # Per-regime OOS breakdown — stitch each regime's OOS daily returns and
    # compute real (not per-fold-annualized) metrics on the subset.
    per_regime = {}
    buckets = collections.defaultdict(list)
    for label, ret, _dates in result.get("_regime_folds", []):
        buckets[label].append(np.asarray(ret))
    for label, arrs in buckets.items():
        r = np.concatenate(arrs) if arrs else np.array([])
        if len(r) == 0:
            continue
        eq = (1.0 + r).cumprod()
        per_regime[label] = {
            "folds": len(arrs),
            "trading_days": int(len(r)),
            "cagr_pct": round(cagr(eq, len(r)) * 100, 2),
            "sharpe": round(sharpe_ratio(r), 2),
            "maxdd_pct": round(max_drawdown(eq) * 100, 2),
            "total_return_pct": round((eq[-1] - 1) * 100, 2),
        }

    def trade_summary(window):
        ts = [t for t in result["all_trades"] if t["window"] == window]
        if not ts:
            return {"round_trips": 0}
        pnls  = [t["pnl_pct"] for t in ts]
        wins  = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        holds = [t["days_open"] for t in ts]
        gross_win  = sum(wins)
        gross_loss = abs(sum(losses))
        return {
            "round_trips":   len(ts),
            "win_rate_pct":  round(100 * len(wins) / len(ts), 1),
            "avg_win_pct":   round(float(np.mean(wins)), 2) if wins else 0.0,
            "avg_loss_pct":  round(float(np.mean(losses)), 2) if losses else 0.0,
            "avg_hold_days": round(float(np.mean(holds)), 1),
            "best_pct":      round(max(pnls), 2),
            "worst_pct":     round(min(pnls), 2),
            "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 1e-9 else None,
        }

    cagr_deg   = round(is_metrics["_cagr"] - oos_metrics["_cagr"], 2)
    sharpe_deg = round(is_metrics["_sharpe"] - oos_metrics["_sharpe"], 2)
    oos_is_ratio = (round(oos_metrics["_cagr"] / is_metrics["_cagr"] * 100, 1)
                    if abs(is_metrics["_cagr"]) > 0.01 else None)

    payload = {
        "schema": "walk_forward_run.v1",
        "strategy": "momentum_pullback (regime SMA50/200 + EMA-pullback arm + "
                    "close>prior-high trigger + ATR/chandelier trail)",
        "notes_for_analyst": (
            "OOS = blind out-of-sample (the honest number). The strategy must beat "
            "benchmark_regime_filtered_bah (hold-when-trending, else T-bills) to justify "
            "its complexity, not just raw buy_hold. per_regime_oos shows where any edge "
            "lives (bull/chop/bear) computed on stitched daily returns, not per-fold "
            "annualized. Check per_fold params for stability: values jumping around each "
            "fold => fitting noise. oos_is_cagr_ratio near 100% => robust; near 0% => "
            "fragile. 'trades' aggregate counts fills (entry+exit); trade_summary counts "
            "round trips. IS aggregate metrics are distorted by overlapping train windows; "
            "trust per_fold IS and all OOS figures."
        ),
        "config": {
            "ticker": ticker,
            "date_range": [str(start), str(end)],
            "train_months": train_m,
            "test_months": test_m,
            "cost_per_trade_pct": round(COST_PER_TRADE * 100, 3),
            "param_grid": {k: list(v) for k, v in PARAM_GRID.items()},
            "optimizer_objective": "max CAGR s.t. Sharpe>=1.2 (fallback: max Sharpe)",
            "num_folds": len(folds),
        },
        "benchmark_buy_hold": {
            "total_return_pct": round(bah_total, 1),
            "cagr_pct": round(bah_cagr, 1),
            "max_drawdown_pct": round(bah_maxdd, 1),
        },
        "in_sample": {
            "cagr_pct": round(is_metrics["_cagr"], 2),
            "sharpe": round(is_metrics["_sharpe"], 2),
            "max_drawdown_pct": round(is_metrics["_max_dd"], 2),
            "win_rate_daily_pct": round(is_metrics["_win_rate"], 1),
            "trades_fills": is_metrics["_trades"],
        },
        "out_of_sample": {
            "cagr_pct": round(oos_metrics["_cagr"], 2),
            "sharpe": round(oos_metrics["_sharpe"], 2),
            "max_drawdown_pct": round(oos_metrics["_max_dd"], 2),
            "win_rate_daily_pct": round(oos_metrics["_win_rate"], 1),
            "trades_fills": oos_metrics["_trades"],
        },
        "degradation": {
            "cagr_pp_lost": cagr_deg,
            "sharpe_lost": sharpe_deg,
            "oos_is_cagr_ratio_pct": oos_is_ratio,
            "oos_beats_buy_hold": bool(oos_metrics["_cagr"] > bah_cagr),
            "oos_vs_buy_hold_cagr_gap_pp": round(oos_metrics["_cagr"] - bah_cagr, 2),
        },
        "benchmark_regime_filtered_bah": (
            {
                "description": "Hold instrument when close>SMA200 (lagged 1 day), "
                               "else cash at risk-free rate; cost on each switch.",
                "cagr_pct": regime_bench["cagr_pct"],
                "max_drawdown_pct": regime_bench["maxdd_pct"],
                "sharpe": regime_bench["sharpe"],
                "total_return_pct": regime_bench["total_pct"],
                "pct_time_in_market": regime_bench["pct_time_in_market"],
                "num_switches": regime_bench["num_switches"],
                "cash_yield_pct": regime_bench["cash_yield_pct"],
                "oos_strategy_beats_this": bool(oos_metrics["_cagr"] > regime_bench["cagr_pct"]),
            } if regime_bench else None
        ),
        "per_regime_oos": per_regime,
        "parameter_stability": stability,
        "trade_summary": {
            "in_sample":     trade_summary("IS"),
            "out_of_sample": trade_summary("OOS"),
        },
        "per_fold": [
            {
                "fold": f["fold"],
                "regime": f.get("regime", "unknown"),
                "train": [f["train_start"], f["train_end"]],
                "test_end": f["test_end"],
                "params": f["params"],
                "is_cagr": f["is_cagr"], "is_sharpe": f["is_sharpe_eval"], "is_maxdd": f["is_maxdd"],
                "oos_cagr": f["oos_cagr"], "oos_sharpe": f["oos_sharpe"], "oos_maxdd": f["oos_maxdd"],
                "is_trades": f["is_trades"], "oos_trades": f["oos_trades"],
            }
            for f in folds
        ],
    }
    return json.dumps(payload, indent=2)


def main():
    st.set_page_config(page_title="Walk-Forward Analyzer", layout="wide",
                       initial_sidebar_state="expanded",
                       menu_items={})

    # Force dark mode regardless of system/browser setting
    st.markdown("""
        <script>
        document.documentElement.setAttribute('data-theme', 'dark');
        </script>
    """, unsafe_allow_html=True)
    st.markdown("""
    <style>
    /* Force dark mode */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
        background-color: #0E1117 !important;
        color: #E8ECF0 !important;
    }
    [data-testid="stToolbar"] { background-color: #0E1117 !important; }
    /* Better contrast for all text elements */
    p, span, div, label, .stMarkdown { color: #E8ECF0 !important; }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label { color: #C8D0D8 !important; }
    /* Alert/info boxes — force dark text since Streamlit renders these light */
    .stAlert, .stAlert p, .stAlert span, .stAlert div,
    div[data-testid="stNotification"], div[data-testid="stNotification"] p,
    div[data-testid="stNotification"] span { color: #111827 !important; }
    .stInfo  { background: #dbeafe !important; border-color: #3b82f6 !important; }
    .stSuccess { background: #dcfce7 !important; border-color: #22c55e !important; }
    .stWarning { background: #fef9c3 !important; border-color: #eab308 !important; }
    .stError   { background: #fee2e2 !important; border-color: #ef4444 !important; }
    .metric-card {
        background: #1a2035;
        border-radius: 10px;
        padding: 18px 14px;
        text-align: center;
        border: 1px solid #2e3a50;
    }
    .metric-label {
        color: #8899AA;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #E8ECF0;
    }
    .metric-green  { color: #00EE88 !important; }
    .metric-red    { color: #FF5555 !important; }
    .metric-blue   { color: #7090FF !important; }
    .metric-orange { color: #FFAA00 !important; }
    .metric-white  { color: #E8ECF0 !important; }
    .degradation-card {
        background: #141820;
        border-radius: 10px;
        padding: 16px 14px;
        text-align: center;
        border: 1px solid #2e3a50;
    }
    .deg-arrow-down { color: #FF5555; font-size: 0.85rem; }
    .deg-arrow-up   { color: #00EE88; font-size: 0.85rem; }
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        color: #E8ECF0 !important;
    }
    /* Trade table */
    .trade-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    .trade-table th {
        background: #1e2a3a; color: #A0B4C8;
        padding: 8px 12px; text-align: left;
        border-bottom: 2px solid #2e3a50;
        font-weight: 600; letter-spacing: 0.5px;
        text-transform: uppercase; font-size: 0.72rem;
    }
    .trade-table td { padding: 7px 12px; border-bottom: 1px solid #1e2a3a; color: #D0D8E0; }
    .trade-row-is  { background: #12183a; }
    .trade-row-oos { background: #2a1800; }
    .trade-row-is:hover  { background: #1a2450; }
    .trade-row-oos:hover { background: #3a2200; }
    .tag-is  { background:#1e2e6a; color:#7090FF; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:700; }
    .tag-oos { background:#4a2800; color:#FFAA00; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:700; }
    .pnl-pos { color: #00EE88; font-weight: 600; }
    .pnl-neg { color: #FF5555; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ──
    st.sidebar.title("⚙️ Walk-Forward Analyzer")
    ticker = st.sidebar.text_input("Ticker", value="SPY")
    col_s1, col_s2 = st.sidebar.columns(2)
    start_date = col_s1.date_input("Start", value=datetime(2018, 1, 1))
    end_date   = col_s2.date_input("End",   value=datetime.today())

    st.sidebar.markdown("---")
    train_months = st.sidebar.slider("Training window (months)", 6, 24, 12)
    test_months  = st.sidebar.slider("Blind test window (months)", 1, 24, 3)
    cash_yield   = st.sidebar.number_input(
        "Cash yield for regime benchmark (annual %)",
        min_value=0.0, max_value=10.0, value=3.7, step=0.1,
        help="Return earned while the regime-filtered benchmark sits in cash "
             "(approx. current 3-month T-bill ≈ 3.7%).",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Strategy Rules")
    st.sidebar.markdown(
        '<pre style="background:#1a2035;color:#C8E0FF;border:1px solid #2e3a50;'
        'border-radius:6px;padding:10px;font-size:0.75rem;line-height:1.6;overflow-x:auto;">'
        'REGIME: close &gt; SMA(200)\n'
        '        AND SMA(50) &gt; SMA(200)\n'
        'ARM   : flat &amp; in-regime, a bar dips to\n'
        '        the pullback EMA (low &le; EMA)\n'
        'ENTRY : armed AND close &gt; prior-bar high\n'
        '        (momentum resuming; same-day close)\n'
        'STOP  : entry - atr_stop &times; ATR(14)\n'
        'TRAIL : hi_close - atr_trail &times; ATR(14)\n'
        '        (chandelier; ratchets up only)\n'
        'EXIT  : close &lt; effective stop\n'
        '        OR  close &lt; SMA(200)\n'
        '        OR  end of window (force-close)\n'
        'OPT   : Max CAGR where Sharpe &ge; 1.2\n'
        '        (fallback: max Sharpe if none qualify)'
        '</pre>', unsafe_allow_html=True
    )

    st.sidebar.markdown("### Transaction Costs")
    st.sidebar.markdown(
        f'<pre style="background:#1a2035;color:#C8E0FF;border:1px solid #2e3a50;'
        f'border-radius:6px;padding:10px;font-size:0.75rem;line-height:1.6;overflow-x:auto;">'
        f'Exchange fee : {EXCHANGE_FEE*100:.2f} % per trade\n'
        f'Slippage     : {SLIPPAGE*100:.2f} % per trade\n'
        f'Total cost   : {COST_PER_TRADE*100:.2f} % per trade'
        f'</pre>', unsafe_allow_html=True
    )

    st.sidebar.markdown("### Parameter Search Grid")
    grid_size = int(np.prod([len(v) for v in PARAM_GRID.values()]))
    st.sidebar.markdown(
        f'<pre style="background:#1a2035;color:#C8E0FF;border:1px solid #2e3a50;'
        f'border-radius:6px;padding:10px;font-size:0.75rem;line-height:1.6;overflow-x:auto;">'
        f'Pullback EMA   : {PARAM_GRID["ema"][0]} \u2013 {PARAM_GRID["ema"][-1]} days\n'
        f'ATR stop mult  : {PARAM_GRID["atr_stop"][0]} \u2013 {PARAM_GRID["atr_stop"][-1]} \u00d7 ATR\n'
        f'ATR trail mult : {PARAM_GRID["atr_trail"][0]} \u2013 {PARAM_GRID["atr_trail"][-1]} \u00d7 ATR\n'
        f'\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n'
        f'Grid combos/fold : {grid_size}'
        f'</pre>', unsafe_allow_html=True
    )

    run_btn = st.sidebar.button("🚀 Run Walk-Forward Analysis", use_container_width=True,
                                 type="primary")

    # ── Main area ──
    st.title("Illusion vs. Reality")
    st.caption(
        "**Left:** what a retail trader sees after optimising on historical data (the illusion).  "
        "**Right:** what actually happens when those parameters meet unseen data (the reality)."
    )

    if not run_btn:
        st.info("Configure settings in the sidebar, then click **Run Walk-Forward Analysis**.")
        return

    # ── Fetch data ──
    with st.spinner(f"Downloading {ticker} data …"):
        df, df_full = fetch_data(ticker, str(start_date), str(end_date))
    if df.empty or len(df) < 20:
        st.error("Not enough data. Check ticker and date range.")
        return
    warmup_bars = len(df_full) - len(df)
    st.success(
        f"Loaded {len(df)} bars of {ticker}  "
        f"({df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')})  "
        f"· {warmup_bars} pre-start warm-up bars fetched for SMA/EMA pre-calculation"
    )

    # ── Walk-forward ──
    st.markdown('<div class="section-header">⏳ Running Walk-Forward Optimization …</div>',
                unsafe_allow_html=True)
    progress = st.progress(0, text="Processing folds …")
    result = run_walk_forward(df, df_full, train_months, test_months, progress_bar=progress)
    progress.progress(100, text="Complete ✓")

    if result is None:
        st.error(
            f"**No valid folds produced.**\n\n"
            f"With a {train_months}-month training window and {test_months}-month test window, "
            f"no folds fit within your date range. "
            f"Try a wider date range or reduce the training/test window sizes."
        )
        return

    num_folds = len(result["folds"])
    st.success(f"Completed {num_folds} folds.")

    # ── Section 1: Gantt chart ──
    st.markdown('<div class="section-header">📅 Walk-Forward Windows</div>',
                unsafe_allow_html=True)
    st.plotly_chart(gantt_chart(result["gantt"]), use_container_width=True)

    # ── Section 2: Metrics cards ──
    st.markdown('<div class="section-header">📊 Illusion vs. Reality</div>',
                unsafe_allow_html=True)
    st.info(
        "**Annualized comparison:** IS windows are 12 months; OOS windows are 3 months. "
        "**Total Return\\*** is window-length-dependent and not directly comparable. "
        "Use **CAGR** and **Sharpe Ratio** for apples-to-apples IS vs OOS comparison — "
        "both are annualized to a common 1-year basis regardless of window length."
    )

    is_metrics  = full_metrics(result["is_returns"],  result["is_trades"])
    oos_metrics = full_metrics(result["oos_returns"], result["oos_trades"])

    # Buy & Hold over the full date range (no costs)
    bah_total  = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
    bah_days   = len(df)
    bah_eq     = np.array([1.0, df["Close"].iloc[-1] / df["Close"].iloc[0]])
    bah_cagr   = cagr(bah_eq, bah_days) * 100
    bah_curve  = df["Close"].values / df["Close"].values[0]
    bah_maxdd  = max_drawdown(bah_curve) * 100
    bah_color  = "metric-green" if bah_cagr > 0 else "metric-red"

    # Regime-filtered Buy & Hold (hold when trending, else cash at risk-free)
    regime_bench = regime_filtered_bah(df, rf_annual_pct=cash_yield)

    def _val_color(val: float, invert: bool = False) -> str:
        if invert:
            return "metric-green" if val <= 0 else "metric-red"
        return "metric-green" if val > 0 else ("metric-red" if val < 0 else "metric-white")

    def single_card(label, value_str, color_class):
        return (
            f'<div class="metric-card">'
            f'  <div class="metric-label">{label}</div>'
            f'  <div class="metric-value {color_class}" style="font-size:1.8rem;">{value_str}</div>'
            f'</div>'
        )

    # ── Buy & Hold banner — full width horizontal strip ──
    st.markdown(
        f'<div style="background:#0f2a1a; border:1px solid #00CC96; border-radius:10px; '
        f'padding:14px 28px; display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">'
        f'  <div style="color:#00CC96; font-size:1rem; font-weight:600;">📈 Buy &amp; Hold {ticker} — No costs · Full period</div>'
        f'  <div style="display:flex; gap:40px; align-items:center;">'
        f'    <div style="text-align:center;">'
        f'      <div style="font-size:0.65rem; color:#888; text-transform:uppercase; letter-spacing:1px;">Total Return</div>'
        f'      <div class="metric-value {bah_color}" style="font-size:1.4rem;">{bah_total:+.1f}%</div>'
        f'    </div>'
        f'    <div style="text-align:center;">'
        f'      <div style="font-size:0.65rem; color:#888; text-transform:uppercase; letter-spacing:1px;">CAGR (annualized)</div>'
        f'      <div class="metric-value {bah_color}" style="font-size:1.4rem;">{bah_cagr:+.1f}%</div>'
        f'    </div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Header row: IS | OOS ──
    col_h1, col_h2 = st.columns(2)
    col_h1.markdown(
        '<div class="metric-card" style="background:#1e2a5a; border-color:#636EFA; padding:24px;">'
        '<div class="metric-value metric-blue" style="font-size:1.8rem;">Retail Backtest</div>'
        '<div class="metric-label" style="margin-top:6px;">In-Sample (Curve-Fit)</div></div>',
        unsafe_allow_html=True
    )
    col_h2.markdown(
        '<div class="metric-card" style="background:#3a2510; border-color:#FF8C00; padding:24px;">'
        '<div class="metric-value metric-orange" style="font-size:1.8rem;">Walk-Forward</div>'
        '<div class="metric-label" style="margin-top:6px;">Out-of-Sample (Blind)</div></div>',
        unsafe_allow_html=True
    )

    # ── Row 1: CAGR (annualized) | Sharpe  ||  CAGR | Sharpe ──
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(single_card("CAGR (annualized)", is_metrics["CAGR"],
                _val_color(is_metrics["_cagr"])), unsafe_allow_html=True)
    c2.markdown(single_card("SHARPE RATIO", is_metrics["Sharpe Ratio"],
                _val_color(is_metrics["_sharpe"])), unsafe_allow_html=True)
    c3.markdown(single_card("CAGR (annualized)", oos_metrics["CAGR"],
                _val_color(oos_metrics["_cagr"])), unsafe_allow_html=True)
    c4.markdown(single_card("SHARPE RATIO", oos_metrics["Sharpe Ratio"],
                _val_color(oos_metrics["_sharpe"])), unsafe_allow_html=True)

    # ── Row 2: Max Drawdown | Trades  ||  Max Drawdown | Trades ──
    c5, c6, c7, c8 = st.columns(4)
    c5.markdown(single_card("MAX DRAWDOWN", is_metrics["Max Drawdown"],
                _val_color(is_metrics["_max_dd"], invert=True)), unsafe_allow_html=True)
    c6.markdown(single_card("TRADES", is_metrics["Trades"], "metric-white"),
                unsafe_allow_html=True)
    c7.markdown(single_card("MAX DRAWDOWN", oos_metrics["Max Drawdown"],
                _val_color(oos_metrics["_max_dd"], invert=True)), unsafe_allow_html=True)
    c8.markdown(single_card("TRADES", oos_metrics["Trades"], "metric-white"),
                unsafe_allow_html=True)

    # Degradation row — CAGR-based for annualized apples-to-apples comparison
    st.markdown("")
    cagr_deg   = is_metrics["_cagr"] - oos_metrics["_cagr"]
    sr_deg     = is_metrics["_sharpe"] - oos_metrics["_sharpe"]
    oos_is_pct = (
        (oos_metrics["_cagr"] / is_metrics["_cagr"] * 100)
        if abs(is_metrics["_cagr"]) > 0.01 else 0
    )

    dc1, dc2, dc3 = st.columns(3)
    dc1.markdown(
        f'<div class="degradation-card">'
        f'  <div class="metric-label">CAGR Degradation (annualized)</div>'
        f'  <div class="metric-value metric-red">{cagr_deg:+.1f}%</div>'
        f'  <div class="deg-arrow-down">↓ IS CAGR lost {cagr_deg:.1f}pp on unseen data</div>'
        f'</div>', unsafe_allow_html=True
    )
    dc2.markdown(
        f'<div class="degradation-card">'
        f'  <div class="metric-label">Sharpe Degradation (annualized)</div>'
        f'  <div class="metric-value metric-red">{sr_deg:+.2f}</div>'
        f'  <div class="deg-arrow-down">↓ {sr_deg:.2f} Sharpe lost on unseen data</div>'
        f'</div>', unsafe_allow_html=True
    )
    oos_color = "metric-green" if oos_is_pct > 50 else "metric-red"
    dc3.markdown(
        f'<div class="degradation-card">'
        f'  <div class="metric-label">OOS / IS CAGR Ratio</div>'
        f'  <div class="metric-value {oos_color}">{oos_is_pct:.0f}%</div>'
        f'  <div class="deg-arrow-down">OOS annualized return as % of IS</div>'
        f'</div>', unsafe_allow_html=True
    )

    # ── Section 3: Equity curves ──
    st.markdown('<div class="section-header">📈 Equity Curves</div>',
                unsafe_allow_html=True)
    bah_close  = df["Close"].values
    bah_equity = bah_close / bah_close[0]
    st.plotly_chart(equity_chart(result, df.index, bah_equity), use_container_width=True)

    # ── Section 4: Trade Log ──
    st.markdown('<div class="section-header">🗒️ Trade Log</div>', unsafe_allow_html=True)
    trades_list = result.get("all_trades", [])
    if trades_list:
        # Sort: fold → IS before OOS → open date
        sorted_trades = sorted(
            trades_list,
            key=lambda t: (t["fold"], 0 if t["window"] == "IS" else 1, t["open_date"])
        )

        # Pre-compute cumulative $ P&L per (fold, window) group
        # Assumes $1000 per trade notional for cumulative display
        NOTIONAL = 1000.0
        cum_tracker = {}   # key=(fold, window) → running cumulative $
        cum_values  = []   # parallel list to sorted_trades
        for t in sorted_trades:
            key = (t["fold"], t["window"])
            prev = cum_tracker.get(key, 0.0)
            trade_dollar = NOTIONAL * t["pnl_pct"] / 100.0
            cum = prev + trade_dollar
            cum_tracker[key] = cum
            cum_values.append(cum)

        rows_html = ""
        for t, cum in zip(sorted_trades, cum_values):
            tag     = "IS" if t["window"] == "IS" else "OOS"
            tag_cls = "tag-is" if tag == "IS" else "tag-oos"
            row_cls = "trade-row-is" if tag == "IS" else "trade-row-oos"
            pnl_cls = "pnl-pos" if t["pnl_pct"] >= 0 else "pnl-neg"
            cum_cls = "pnl-pos" if cum >= 0 else "pnl-neg"
            pnl_str = f"{t['pnl_pct']:+.2f}%"
            cum_str = f"${cum:+.2f}"
            rows_html += (
                f'<tr class="{row_cls}">'
                f'  <td style="color:#8899AA;">Fold {t["fold"]}</td>'
                f'  <td><span class="{tag_cls}">{tag}</span></td>'
                f'  <td>{t["open_date"].strftime("%y%m%d")}</td>'
                f'  <td>{t["close_date"].strftime("%y%m%d")}</td>'
                f'  <td>{t["days_open"]}</td>'
                f'  <td>${t["buy_price"]:.2f}</td>'
                f'  <td>${t["sell_price"]:.2f}</td>'
                f'  <td class="{pnl_cls}">{pnl_str}</td>'
                f'  <td class="{cum_cls}">{cum_str}</td>'
                f'</tr>'
            )
        st.markdown(
            f'<table class="trade-table">'
            f'<thead><tr>'
            f'  <th>Fold</th><th>Window</th><th>Open Date</th><th>Close Date</th>'
            f'  <th>Days Open</th><th>Buy Price</th><th>Sell Price</th>'
            f'  <th>P&amp;L %</th><th>Cum $ (per $1k/trade)</th>'
            f'</tr></thead>'
            f'<tbody>{rows_html}</tbody>'
            f'</table>',
            unsafe_allow_html=True
        )
        st.caption(
            f"Total: {len(trades_list)} trades  ·  "
            f"IS: {sum(1 for t in trades_list if t['window']=='IS')}  ·  "
            f"OOS: {sum(1 for t in trades_list if t['window']=='OOS')}  ·  "
            f"Winners: {sum(1 for t in trades_list if t['pnl_pct'] > 0)}  ·  "
            f"Losers: {sum(1 for t in trades_list if t['pnl_pct'] <= 0)}"
        )
    else:
        st.info("No completed trades in this run.")

    # ── Section 5: Fold detail table ──
    with st.expander("🔍 Fold-by-Fold Parameter Detail"):
        rows = []
        for f in result["folds"]:
            p = f["params"]
            rows.append({
                "Fold":         f["fold"],
                "Pullback EMA": p["ema"],
                "ATR Stop ×":   p["atr_stop"],
                "ATR Trail ×":  p["atr_trail"],
                "IS Sharpe":    f"{f['is_sharpe']:.2f}",
                "IS Trades":    f["is_trades"],
                "OOS Trades":   f["oos_trades"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Section 5b: Regime-Filtered Benchmark & Per-Regime OOS ──
    st.markdown('<div class="section-header">🛡️ Regime-Filtered Buy &amp; Hold '
                '(the benchmark to beat)</div>', unsafe_allow_html=True)
    st.caption(
        f"Holds {ticker} when it closed above its 200-day yesterday, otherwise sits in "
        f"cash earning {cash_yield:.1f}% annually. Costs charged on each switch. "
        f"This is the honest, low-complexity bar the active strategy must clear — on "
        f"both return and drawdown."
    )
    rb = regime_bench
    beat = oos_metrics["_cagr"] > rb["cagr_pct"]
    cols = st.columns(4)
    cols[0].markdown(single_card("CAGR (annualized)", f'{rb["cagr_pct"]:+.1f}%',
                                 _val_color(rb["cagr_pct"])), unsafe_allow_html=True)
    cols[1].markdown(single_card("Max Drawdown", f'{rb["maxdd_pct"]:.1f}%',
                                 _val_color(rb["maxdd_pct"], invert=True)), unsafe_allow_html=True)
    cols[2].markdown(single_card("Sharpe", f'{rb["sharpe"]:.2f}',
                                 _val_color(rb["sharpe"])), unsafe_allow_html=True)
    cols[3].markdown(single_card("% Time In Market", f'{rb["pct_time_in_market"]:.0f}%',
                                 "metric-white"), unsafe_allow_html=True)
    verdict = ("✅ The OOS strategy beats this benchmark on CAGR."
               if beat else
               "❌ The OOS strategy does NOT beat this benchmark — the simpler regime "
               "overlay wins, so the active machinery is subtracting value.")
    st.markdown(
        f'<div style="background:{"#0f2a1a" if beat else "#2a0f12"};border:1px solid '
        f'{"#00CC96" if beat else "#EF553B"};border-radius:8px;padding:10px 16px;margin-top:6px;">'
        f'<span style="color:{"#00CC96" if beat else "#EF553B"};font-weight:600;">{verdict}</span> '
        f'<span style="color:#9aa4b2;">OOS {oos_metrics["_cagr"]:+.1f}% vs regime-B&amp;H '
        f'{rb["cagr_pct"]:+.1f}% · {rb["num_switches"]} switches</span></div>',
        unsafe_allow_html=True
    )

    # Per-regime OOS breakdown
    import collections as _collections
    _buckets = _collections.defaultdict(list)
    for label, ret, _d in result.get("_regime_folds", []):
        _buckets[label].append(np.asarray(ret))
    if _buckets:
        st.markdown("**Out-of-sample performance by market regime** "
                    "(where the edge actually lives):")
        reg_rows = []
        for label in ["bull", "chop", "bear", "unknown"]:
            if label not in _buckets:
                continue
            r = np.concatenate(_buckets[label])
            if len(r) == 0:
                continue
            eq = (1.0 + r).cumprod()
            reg_rows.append({
                "Regime":     label.upper(),
                "Folds":      len(_buckets[label]),
                "Trading Days": int(len(r)),
                "OOS CAGR":   f"{cagr(eq, len(r))*100:+.1f}%",
                "OOS Sharpe": f"{sharpe_ratio(r):.2f}",
                "OOS MaxDD":  f"{max_drawdown(eq)*100:.1f}%",
            })
        st.dataframe(pd.DataFrame(reg_rows), use_container_width=True, hide_index=True)

    # ── Section 6: Analysis Export (paste-to-Claude) ──
    st.markdown('<div class="section-header">🤖 Analysis Export</div>',
                unsafe_allow_html=True)
    st.caption(
        "Compact JSON summary of this run — config, IS/OOS aggregates, degradation, "
        "raw + regime-filtered benchmarks, per-regime OOS breakdown, per-fold params "
        "(with regime tags) and trade summaries. "
        "Copy it (hover the box → copy icon) or download, then paste it into Claude for analysis."
    )
    export_json = build_analysis_export(
        result,
        ticker=ticker, start=start_date, end=end_date,
        train_m=train_months, test_m=test_months,
        is_metrics=is_metrics, oos_metrics=oos_metrics,
        bah_total=bah_total, bah_cagr=bah_cagr, bah_maxdd=bah_maxdd,
        regime_bench=regime_bench,
    )
    import html as _html
    st.markdown(
        f'<pre style="background:#0d1320;color:#C8E0FF;border:1px solid #2e3a50;'
        f'border-radius:8px;padding:14px;font-size:0.72rem;line-height:1.45;'
        f'max-height:420px;overflow:auto;white-space:pre;">'
        f'{_html.escape(export_json)}</pre>',
        unsafe_allow_html=True
    )
    st.download_button(
        "⬇️ Download run summary (JSON)",
        data=export_json,
        file_name=f"wf_{ticker}_{datetime.today().strftime('%Y%m%d')}.json",
        mime="application/json",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
