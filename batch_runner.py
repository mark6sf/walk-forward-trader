"""
Multi-Ticker Walk-Forward Batch Runner
======================================
Loops a basket of tickers through the SAME engine as walk_forward_app.py and
prints one comparison table:  active strategy (blind OOS)  vs  regime-filtered
buy & hold  vs  raw buy & hold — plus a beat/miss verdict and a target hurdle.

Usage:
    python batch_runner.py
    python batch_runner.py --tickers NVDA,AVGO,MSFT,META,XMMO,SPY,QQQ
    python batch_runner.py --tickers NVDA,AMD --start 2019-01-01 --target 7 \
                           --cash-yield 3.7 --train 12 --test 3 --out basket.json

The combined JSON (one entry per ticker) is written for pasting into Claude.
Run from the same folder as walk_forward_app.py (it imports the engine).
"""

import argparse
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import walk_forward_app as wf   # reuse engine: run_walk_forward, full_metrics, etc.

DEFAULT_BASKET = ["NVDA", "AVGO", "MSFT", "META", "AAPL", "AMZN",
                  "XMMO", "VFMO", "SPY", "QQQ"]


def fetch(ticker: str, start: str, end: str):
    """Mirror of walk_forward_app.fetch_data without the Streamlit cache decorator.
    Returns (df, df_full) with warm-up history prepended so SMA200 is pre-settled.
    Kept as a module-level function so tests can monkeypatch it."""
    from dateutil.relativedelta import relativedelta as rd
    import yfinance as yf

    start_dt     = pd.Timestamp(start)
    warmup_start = start_dt - rd(days=int(wf.WARMUP_DAYS * 2.0))
    raw = yf.download(ticker, start=warmup_start.strftime("%Y-%m-%d"),
                      end=end, progress=False, auto_adjust=True)
    if raw is None or len(raw) == 0:
        return None, None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df_full = raw.copy()
    df      = raw[raw.index >= start_dt].copy()
    if len(df) < 60:
        return None, None
    return df, df_full


def _per_regime(result):
    """Compact per-regime OOS metrics from stitched daily returns."""
    import collections
    buckets = collections.defaultdict(list)
    for label, ret, _d in result.get("_regime_folds", []):
        buckets[label].append(np.asarray(ret))
    out = {}
    for label, arrs in buckets.items():
        r = np.concatenate(arrs) if arrs else np.array([])
        if len(r) == 0:
            continue
        eq = (1.0 + r).cumprod()
        out[label] = {
            "folds": len(arrs),
            "cagr_pct": round(wf.cagr(eq, len(r)) * 100, 2),
            "sharpe": round(wf.sharpe_ratio(r), 2),
            "maxdd_pct": round(wf.max_drawdown(eq) * 100, 2),
        }
    return out


def run_one(ticker, start, end, train_m, test_m, cash_yield, target):
    """Run the full pipeline for a single ticker; return (row_dict, json_summary)."""
    df, df_full = fetch(ticker, start, end)
    if df is None:
        return None, {"ticker": ticker, "error": "no_data"}

    result = wf.run_walk_forward(df, df_full, train_m, test_m)
    if result is None:
        return None, {"ticker": ticker, "error": "no_folds"}

    is_m  = wf.full_metrics(result["is_returns"],  result["is_trades"])
    oos_m = wf.full_metrics(result["oos_returns"], result["oos_trades"])

    bah_total = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
    bah_cagr  = wf.cagr(np.array([1.0, df["Close"].iloc[-1] / df["Close"].iloc[0]]),
                        len(df)) * 100
    bah_maxdd = wf.max_drawdown(df["Close"].values / df["Close"].values[0]) * 100

    rb = wf.regime_filtered_bah(df, rf_annual_pct=cash_yield)

    oos_cagr = oos_m["_cagr"]
    oos_is_ratio = (round(oos_cagr / is_m["_cagr"] * 100, 0)
                    if abs(is_m["_cagr"]) > 0.01 else None)

    row = {
        "Ticker":       ticker,
        "BH_CAGR":      round(bah_cagr, 1),
        "BH_MDD":       round(bah_maxdd, 1),
        "RegBH_CAGR":   rb["cagr_pct"],
        "RegBH_MDD":    rb["maxdd_pct"],
        "RegBH_%In":    rb["pct_time_in_market"],
        "OOS_CAGR":     round(oos_cagr, 1),
        "OOS_Shrp":     round(oos_m["_sharpe"], 2),
        "OOS_MDD":      round(oos_m["_max_dd"], 1),
        "OOS>Reg":      "Y" if oos_cagr > rb["cagr_pct"] else "-",
        "Reg>=Tgt":     "Y" if rb["cagr_pct"] >= target else "-",
        "OOS>=Tgt":     "Y" if oos_cagr >= target else "-",
        "OOS/IS%":      oos_is_ratio,
    }

    summary = {
        "ticker": ticker,
        "buy_hold":            {"cagr_pct": round(bah_cagr, 2), "maxdd_pct": round(bah_maxdd, 2)},
        "regime_filtered_bah": {"cagr_pct": rb["cagr_pct"], "maxdd_pct": rb["maxdd_pct"],
                                 "sharpe": rb["sharpe"], "pct_time_in_market": rb["pct_time_in_market"],
                                 "num_switches": rb["num_switches"]},
        "strategy_oos":        {"cagr_pct": round(oos_cagr, 2), "sharpe": round(oos_m["_sharpe"], 2),
                                 "maxdd_pct": round(oos_m["_max_dd"], 2), "oos_is_ratio_pct": oos_is_ratio},
        "verdicts": {
            "oos_beats_regime_bah": bool(oos_cagr > rb["cagr_pct"]),
            "oos_beats_buy_hold":   bool(oos_cagr > bah_cagr),
            "regime_bah_clears_target": bool(rb["cagr_pct"] >= target),
            "oos_clears_target":        bool(oos_cagr >= target),
        },
        "per_regime_oos": _per_regime(result),
        "num_folds": len(result["folds"]),
    }
    return row, summary


def main():
    ap = argparse.ArgumentParser(description="Multi-ticker walk-forward batch runner")
    ap.add_argument("--tickers", type=str, default=",".join(DEFAULT_BASKET),
                    help="Comma-separated tickers")
    ap.add_argument("--start", type=str, default="2018-01-01")
    ap.add_argument("--end", type=str, default=datetime.today().strftime("%Y-%m-%d"))
    ap.add_argument("--train", type=int, default=12, help="Training window (months)")
    ap.add_argument("--test", type=int, default=3, help="Blind test window (months)")
    ap.add_argument("--cash-yield", type=float, default=3.7,
                    help="Annual %% earned in cash by the regime benchmark")
    ap.add_argument("--target", type=float, default=7.0, help="Target CAGR hurdle (%%)")
    ap.add_argument("--out", type=str, default="batch_summary.json")
    ap.add_argument("--sort", type=str, default="RegBH_CAGR",
                    help="Column to sort the table by (descending)")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    print(f"\nBatch walk-forward · {len(tickers)} tickers · {args.start} → {args.end}")
    print(f"train={args.train}m test={args.test}m · cash={args.cash_yield}% · "
          f"target={args.target}%\n")

    rows, summaries = [], []
    for i, t in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {t} ...", end=" ", flush=True)
        try:
            row, summary = run_one(t, args.start, args.end, args.train, args.test,
                                   args.cash_yield, args.target)
            summaries.append(summary)
            if row is None:
                print(f"skipped ({summary.get('error','?')})")
            else:
                rows.append(row)
                print(f"OOS {row['OOS_CAGR']:+.1f}% | RegBH {row['RegBH_CAGR']:+.1f}% | "
                      f"BH {row['BH_CAGR']:+.1f}%")
        except Exception as e:
            print(f"ERROR: {e}")
            summaries.append({"ticker": t, "error": str(e)})

    if not rows:
        print("\nNo successful runs.")
        sys.exit(1)

    df_tab = pd.DataFrame(rows)
    if args.sort in df_tab.columns:
        df_tab = df_tab.sort_values(args.sort, ascending=False)

    print("\n" + "=" * 100)
    print("COMPARISON TABLE  (CAGR & MaxDD in %, blind OOS = honest)")
    print("=" * 100)
    print(df_tab.to_string(index=False))
    print("=" * 100)

    # Aggregate verdicts
    n = len(rows)
    beat_reg = sum(1 for r in rows if r["OOS>Reg"] == "Y")
    reg_tgt  = sum(1 for r in rows if r["Reg>=Tgt"] == "Y")
    oos_tgt  = sum(1 for r in rows if r["OOS>=Tgt"] == "Y")
    print(f"\nActive strategy beats regime-filtered B&H : {beat_reg}/{n}")
    print(f"Regime-filtered B&H clears {args.target:.0f}% target : {reg_tgt}/{n}")
    print(f"Active strategy clears {args.target:.0f}% target      : {oos_tgt}/{n}")
    print("\nLegend: OOS=blind out-of-sample strategy · RegBH=hold-when-trending-else-cash · "
          "BH=raw buy&hold · %In=time in market\n")

    payload = {
        "schema": "batch_walk_forward.v1",
        "config": {"tickers": tickers, "date_range": [args.start, args.end],
                   "train_months": args.train, "test_months": args.test,
                   "cash_yield_pct": args.cash_yield, "target_cagr_pct": args.target},
        "aggregate": {"n": n, "oos_beats_regime_bah": beat_reg,
                      "regime_bah_clears_target": reg_tgt, "oos_clears_target": oos_tgt},
        "per_ticker": summaries,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Combined JSON written to: {args.out}  (paste this into Claude)\n")


if __name__ == "__main__":
    main()
