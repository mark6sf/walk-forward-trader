"""
Drawdown Ladder (scale-out + re-entry) — strategy module
========================================================
Quantifies a "take some off the table as it drops, add back as it recovers"
overlay as a CONTINUOUS exposure rule instead of discrete in/out trades:

    drawdown   = (peak - price) / peak          # peak = running or trailing high
    exposure   = clamp(1 - k * drawdown, floor, 1)
    daily P&L  = exposure * asset_return + (1 - exposure) * cash_return
                 - cost * |change in exposure|     # turnover cost

Re-entry styles:
  symmetric : exposure is purely f(current drawdown); it rebuilds automatically
              as price recovers toward the peak.  (default)
  confirm   : trim freely on the way down, but only ADD back once price has
              reclaimed a short EMA (hysteresis — avoids buying a dead-cat bounce).

Anchor (peak):
  peak_lookback = 0   -> running all-time peak (re-risks only near new highs)
  peak_lookback = N   -> trailing N-day high   (re-risks faster)

The runner loops a basket and prints ONE table comparing, per name:
  Buy & Hold   vs   Binary 200-day (cash when below)   vs   Ladder
on CAGR, MaxDrawdown, and Calmar (CAGR/|MaxDD|) — the risk-adjusted yardstick.

Usage:
  python drawdown_ladder.py
  python drawdown_ladder.py --tickers NVDA,META,MSFT,SPY --k 3 --floor 0.25 \
                            --reentry confirm --peak-lookback 0 --target 7
Run from the same folder as walk_forward_app.py (imports its engine helpers).
NOTE: planning/eval tool — analysis to inform your own decisions, not advice.
"""

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

import walk_forward_app as wf   # cagr, sharpe_ratio, max_drawdown, regime_filtered_bah, etc.

DEFAULT_BASKET = ["NVDA", "AVGO", "MSFT", "META", "AAPL", "AMZN",
                  "XMMO", "VFMO", "SPY", "QQQ"]


# ── data ───────────────────────────────────────────────────────────────
def fetch(ticker: str, start: str, end: str):
    """Same warm-up-aware download as the app; returns df over [start, end]."""
    from dateutil.relativedelta import relativedelta as rd
    import yfinance as yf
    start_dt = pd.Timestamp(start)
    warmup   = start_dt - rd(days=int(wf.WARMUP_DAYS * 2.0))
    raw = yf.download(ticker, start=warmup.strftime("%Y-%m-%d"),
                      end=end, progress=False, auto_adjust=True)
    if raw is None or len(raw) == 0:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df = raw[raw.index >= start_dt].copy()
    return df if len(df) >= 60 else None


# ── ladder core ────────────────────────────────────────────────────────
def ladder_exposure(close: np.ndarray, k: float, floor: float,
                    peak_lookback: int = 0) -> np.ndarray:
    """Target exposure each day from drawdown vs a running or trailing peak.
    Causal (peak[t] depends only on close[0..t])."""
    close = np.asarray(close, dtype=float)
    if peak_lookback and peak_lookback > 0:
        peak = pd.Series(close).rolling(peak_lookback, min_periods=1).max().values
    else:
        peak = np.maximum.accumulate(close)
    dd = (peak - close) / np.where(peak > 0, peak, 1e-9)      # >= 0
    return np.clip(1.0 - k * dd, floor, 1.0)


def ladder_backtest(close, k, floor, peak_lookback, cash_yield,
                    reentry="symmetric", confirm_ema=20,
                    cost=None):
    """
    Run the continuous-exposure ladder. Exposure for day t's return is set from
    information through day t-1 (no look-ahead). Returns a metrics dict including
    the daily exposure path.
    """
    if cost is None:
        cost = wf.COST_PER_TRADE
    close = np.asarray(close, dtype=float)
    n = len(close)
    rf_daily = (1.0 + cash_yield / 100.0) ** (1.0 / 252.0) - 1.0

    target = ladder_exposure(close, k, floor, peak_lookback)
    ema = wf.compute_ema(close, confirm_ema) if reentry == "confirm" else None

    daily   = np.zeros(n)
    exp_arr = np.zeros(n)
    exp_prev = target[0]          # establish initial exposure (no cost, like B&H)
    exp_arr[0] = exp_prev
    turnover = 0.0

    for t in range(1, n):
        tgt = target[t - 1]                       # decided on yesterday's close
        if reentry == "confirm" and tgt > exp_prev:
            # only allow adding back once price has reclaimed the EMA
            if not (close[t - 1] > ema[t - 1]):
                tgt = exp_prev
        exp = tgt
        asset_ret = close[t] / close[t - 1] - 1.0
        d_turn = abs(exp - exp_prev)
        daily[t] = exp * asset_ret + (1.0 - exp) * rf_daily - cost * d_turn
        turnover += d_turn
        exp_arr[t] = exp
        exp_prev = exp

    eq = (1.0 + daily).cumprod()
    cg = wf.cagr(eq, n) * 100
    mdd = wf.max_drawdown(eq) * 100
    return {
        "cagr_pct": round(cg, 2),
        "maxdd_pct": round(mdd, 2),
        "sharpe": round(wf.sharpe_ratio(daily), 2),
        "calmar": round(cg / abs(mdd), 2) if abs(mdd) > 1e-6 else None,
        "avg_exposure_pct": round(float(np.mean(exp_arr)) * 100, 1),
        "min_exposure_pct": round(float(np.min(exp_arr)) * 100, 1),
        "annual_turnover": round(turnover / (n / 252.0), 2),
        "_equity": eq,
        "_exposure": exp_arr,
        "_returns": daily,
    }


# ── per-ticker comparison ──────────────────────────────────────────────
def _calmar(cagr_pct, mdd_pct):
    return round(cagr_pct / abs(mdd_pct), 2) if abs(mdd_pct) > 1e-6 else None


def _dd_per_point(bh_cagr, bh_mdd, s_cagr, s_mdd):
    """Drawdown points saved per CAGR point sacrificed (ladder vs buy&hold).
    'FREE' => ladder matched/beat B&H return with less drawdown."""
    dd_saved = abs(bh_mdd) - abs(s_mdd)
    ret_given = bh_cagr - s_cagr
    if ret_given <= 0.05:
        return "FREE" if dd_saved > 0 else "none"
    return round(dd_saved / ret_given, 2)


def run_one(ticker, start, end, cash_yield, k, floor, peak_lookback,
            reentry, confirm_ema, target):
    df = fetch(ticker, start, end)
    if df is None:
        return None, {"ticker": ticker, "error": "no_data"}

    close = df["Close"].values
    days  = len(df)

    # Buy & Hold
    bh_cagr = wf.cagr(np.array([1.0, close[-1] / close[0]]), days) * 100
    bh_mdd  = wf.max_drawdown(close / close[0]) * 100

    # Binary 200-day (cash below) — reuse the app's benchmark
    ma = wf.regime_filtered_bah(df, rf_annual_pct=cash_yield, sma_period=200)

    # Ladder
    lad = ladder_backtest(close, k, floor, peak_lookback, cash_yield,
                          reentry=reentry, confirm_ema=confirm_ema)

    calmars = {"BH": _calmar(bh_cagr, bh_mdd),
               "MA": _calmar(ma["cagr_pct"], ma["maxdd_pct"]),
               "Ladder": lad["calmar"]}
    best = max((c for c in calmars.values() if c is not None), default=None)
    best_name = next((k_ for k_, v in calmars.items() if v == best), "-")

    row = {
        "Ticker":   ticker,
        "BH_CAGR":  round(bh_cagr, 1),  "BH_MDD":  round(bh_mdd, 1),  "BH_Clmr": calmars["BH"],
        "MA_CAGR":  ma["cagr_pct"],     "MA_MDD":  ma["maxdd_pct"],   "MA_Clmr": calmars["MA"],
        "Lad_CAGR": lad["cagr_pct"],    "Lad_MDD": lad["maxdd_pct"],  "Lad_Clmr": lad["calmar"],
        "Lad_AvgEx": lad["avg_exposure_pct"],
        "DD/pt":    _dd_per_point(bh_cagr, bh_mdd, lad["cagr_pct"], lad["maxdd_pct"]),
        "Lad>=Tgt": "Y" if lad["cagr_pct"] >= target else "-",
        "Best":     best_name,
    }
    summary = {
        "ticker": ticker,
        "buy_hold":   {"cagr_pct": round(bh_cagr, 2), "maxdd_pct": round(bh_mdd, 2), "calmar": calmars["BH"]},
        "binary_200ma": {"cagr_pct": ma["cagr_pct"], "maxdd_pct": ma["maxdd_pct"], "calmar": calmars["MA"],
                         "pct_time_in_market": ma["pct_time_in_market"]},
        "ladder": {kk: lad[kk] for kk in ("cagr_pct", "maxdd_pct", "sharpe", "calmar",
                                          "avg_exposure_pct", "min_exposure_pct", "annual_turnover")},
        "best_by_calmar": best_name,
        "ladder_clears_target": bool(lad["cagr_pct"] >= target),
    }
    return row, summary


def main():
    ap = argparse.ArgumentParser(description="Drawdown-ladder scale-out vs B&H vs 200-day")
    ap.add_argument("--tickers", default=",".join(DEFAULT_BASKET))
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=datetime.today().strftime("%Y-%m-%d"))
    ap.add_argument("--cash-yield", type=float, default=3.7)
    ap.add_argument("--k", type=float, default=3.0, help="De-risk slope (exposure drop per unit DD)")
    ap.add_argument("--floor", type=float, default=0.25, help="Minimum exposure (0-1)")
    ap.add_argument("--peak-lookback", type=int, default=0,
                    help="0 = all-time running peak; N = trailing N-day high")
    ap.add_argument("--reentry", choices=["symmetric", "confirm"], default="symmetric")
    ap.add_argument("--confirm-ema", type=int, default=20)
    ap.add_argument("--target", type=float, default=7.0)
    ap.add_argument("--out", default="ladder_summary.json")
    ap.add_argument("--sort", default="Lad_Clmr")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    print(f"\nDrawdown Ladder · {len(tickers)} tickers · {args.start} → {args.end}")
    print(f"k={args.k}  floor={args.floor:.0%}  peak={'all-time' if args.peak_lookback==0 else str(args.peak_lookback)+'d'}"
          f"  reentry={args.reentry}  cash={args.cash_yield}%  target={args.target}%\n")

    rows, summaries = [], []
    for i, t in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {t} ...", end=" ", flush=True)
        try:
            row, summary = run_one(t, args.start, args.end, args.cash_yield, args.k,
                                   args.floor, args.peak_lookback, args.reentry,
                                   args.confirm_ema, args.target)
            summaries.append(summary)
            if row is None:
                print(f"skipped ({summary.get('error','?')})")
            else:
                rows.append(row)
                print(f"Ladder CAGR {row['Lad_CAGR']:+.1f}% MDD {row['Lad_MDD']:.1f}% "
                      f"Calmar {row['Lad_Clmr']} (best: {row['Best']})")
        except Exception as e:
            print(f"ERROR: {e}")
            summaries.append({"ticker": t, "error": str(e)})

    if not rows:
        print("\nNo successful runs."); return

    df_tab = pd.DataFrame(rows)
    if args.sort in df_tab.columns:
        df_tab = df_tab.sort_values(args.sort, ascending=False,
                                    key=lambda s: pd.to_numeric(s, errors="coerce"))

    print("\n" + "=" * 110)
    print("LADDER vs BUY&HOLD vs BINARY-200MA   (CAGR & MaxDD %, Clmr = CAGR/|MaxDD|, higher Clmr better)")
    print("=" * 110)
    print(df_tab.to_string(index=False))
    print("=" * 110)

    n = len(rows)
    lad_best  = sum(1 for r in rows if r["Best"] == "Ladder")
    lad_tgt   = sum(1 for r in rows if r["Lad>=Tgt"] == "Y")
    free      = sum(1 for r in rows if r["DD/pt"] == "FREE")
    print(f"\nLadder is best by Calmar : {lad_best}/{n}")
    print(f"Ladder clears {args.target:.0f}% target : {lad_tgt}/{n}")
    print(f"Ladder cut drawdown for free (no CAGR cost) : {free}/{n}")
    print("\nDD/pt = drawdown points saved per CAGR point given up vs B&H "
          "(higher = better trade; FREE = less DD with no return cost)\n")

    payload = {
        "schema": "drawdown_ladder.v1",
        "config": {"tickers": tickers, "date_range": [args.start, args.end],
                   "k": args.k, "floor": args.floor, "peak_lookback": args.peak_lookback,
                   "reentry": args.reentry, "confirm_ema": args.confirm_ema,
                   "cash_yield_pct": args.cash_yield, "target_cagr_pct": args.target},
        "aggregate": {"n": n, "ladder_best_by_calmar": lad_best,
                      "ladder_clears_target": lad_tgt, "ladder_free_dd_reduction": free},
        "per_ticker": summaries,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Combined JSON written to: {args.out}  (paste into Claude)\n")


if __name__ == "__main__":
    main()
