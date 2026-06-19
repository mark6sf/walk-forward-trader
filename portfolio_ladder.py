"""
Portfolio Ladder — basket-level drawdown test
==============================================
Applies each overlay PER NAME, combines into a portfolio, and reports the
basket-level numbers you'd actually experience. Compares, at portfolio level:

    Buy & Hold   vs   MA regime exit   vs   Drawdown Ladder

under two weightings: equal and inverse-trailing-volatility.

MA regime exit is TUNABLE and has a MODE switch:
  --ma-period N         moving-average length for the exit (try 150-250)
  --ma-mode individual  each name exits to cash when ITS OWN close < its MA
  --ma-mode basket      ONE signal: whole book to cash when the BASKET index < its MA
  --ma-sweep "150,200,250"   optional: also print a gate sweep over MA lengths

Default basket = 13 large-cap momentum leaders (MTUM-style, mid-2026) incl. AMAT.
Run from the folder with walk_forward_app.py and drawdown_ladder.py.
NOTE: today's momentum leaders backtested over history are survivorship-biased;
read RETURN as inflated, focus on DRAWDOWN reduction. Not investment advice.
"""

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

import walk_forward_app as wf
import drawdown_ladder as dl

MOMENTUM_BASKET = ["NVDA", "AVGO", "MU", "AMD", "AMAT", "MSFT", "META",
                   "GOOGL", "ORCL", "PLTR", "JPM", "GE", "CAT"]


# ── metrics ──────────────────────────────────────────────────────────
def _metrics(daily: pd.Series, name):
    d = daily.dropna()
    if len(d) < 30:
        return None
    eq = (1.0 + d.values).cumprod()
    cg = wf.cagr(eq, len(d)) * 100
    mdd = wf.max_drawdown(eq) * 100
    return {
        "approach": name,
        "cagr_pct": round(cg, 2),
        "maxdd_pct": round(mdd, 2),
        "sharpe": round(wf.sharpe_ratio(d.values), 2),
        "calmar": round(cg / abs(mdd), 2) if abs(mdd) > 1e-6 else None,
        "_start": str(d.index[0].date()),
    }


# ── per-name return streams ──────────────────────────────────────────
def per_name_streams(ticker, start, end, cash_yield, k, floor, peak_lookback,
                     reentry, confirm_ema, ma_period):
    df = dl.fetch(ticker, start, end)
    if df is None:
        return None
    close = df["Close"].values
    asset_ret = np.concatenate([[0.0], close[1:] / close[:-1] - 1.0])
    ma = wf.regime_filtered_bah(df, rf_annual_pct=cash_yield, sma_period=ma_period)
    lad = dl.ladder_backtest(close, k, floor, peak_lookback, cash_yield,
                             reentry=reentry, confirm_ema=confirm_ema)
    vol = pd.Series(asset_ret, index=df.index).rolling(60, min_periods=20).std().shift(1)
    return pd.DataFrame({
        "bh":     asset_ret,
        "ma":     ma["_returns"],
        "ladder": lad["_returns"],
        "vol":    vol.values,
        "close":  close,
    }, index=df.index)


# ── portfolio combiner ───────────────────────────────────────────────
def combine(ret_df: pd.DataFrame, weighting: str, vol_df: pd.DataFrame = None) -> pd.Series:
    if weighting == "equal" or vol_df is None:
        return ret_df.mean(axis=1, skipna=True)
    inv = (1.0 / vol_df.reindex(ret_df.index)).where(ret_df.notna())
    inv = inv.where(vol_df.reindex(ret_df.index) > 0)
    w = inv.div(inv.sum(axis=1), axis=0)
    port = (ret_df * w).sum(axis=1, skipna=True)
    return port.where(w.sum(axis=1) > 0, ret_df.mean(axis=1, skipna=True))


def basket_gate(bh_port: pd.Series, ma_period, cash_yield, cost=None) -> pd.Series:
    """One MA signal on the whole basket: in-market when basket index>MA, else cash."""
    if cost is None:
        cost = wf.COST_PER_TRADE
    r = bh_port.dropna()
    rf_daily = (1.0 + cash_yield / 100.0) ** (1.0 / 252.0) - 1.0
    price = (1.0 + r.values).cumprod()
    ma = wf.compute_sma(price, ma_period)
    out = np.zeros(len(r))
    prev = True
    for t in range(1, len(r)):
        in_mkt = price[t - 1] > ma[t - 1]
        out[t] = r.values[t] if in_mkt else rf_daily
        if in_mkt != prev:
            out[t] -= cost
        prev = in_mkt
    return pd.Series(out, index=r.index)


def individual_ma_portfolio(streams, ma_period, cash_yield, weighting):
    """Per-name MA gate recomputed at ma_period, then combined."""
    ret = {}
    for t, s in streams.items():
        dfm = pd.DataFrame({"Close": s["close"].values}, index=s.index)
        g = wf.regime_filtered_bah(dfm, rf_annual_pct=cash_yield, sma_period=ma_period)
        ret[t] = pd.Series(g["_returns"], index=s.index)
    ret_df = pd.DataFrame(ret)
    vol_df = pd.DataFrame({t: s["vol"] for t, s in streams.items()})
    return combine(ret_df, weighting, vol_df)


def ma_portfolio(streams, bh_port, ma_period, ma_mode, cash_yield, weighting):
    if ma_mode == "basket":
        return basket_gate(bh_port, ma_period, cash_yield)
    return individual_ma_portfolio(streams, ma_period, cash_yield, weighting)


def main():
    ap = argparse.ArgumentParser(description="Portfolio overlay comparison (tunable MA exit)")
    ap.add_argument("--tickers", default=",".join(MOMENTUM_BASKET))
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=datetime.today().strftime("%Y-%m-%d"))
    ap.add_argument("--cash-yield", type=float, default=3.7)
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--floor", type=float, default=0.4)
    ap.add_argument("--peak-lookback", type=int, default=0)
    ap.add_argument("--reentry", choices=["symmetric", "confirm"], default="confirm")
    ap.add_argument("--confirm-ema", type=int, default=20)
    ap.add_argument("--weighting", choices=["equal", "invvol", "both"], default="both")
    ap.add_argument("--ma-period", type=int, default=200, help="MA length for the regime exit (try 150-250)")
    ap.add_argument("--ma-mode", choices=["individual", "basket"], default="individual",
                    help="individual = per-name MA exit; basket = one MA signal on the whole book")
    ap.add_argument("--ma-sweep", default=None, help='comma list of MA periods, e.g. "150,175,200,225,250"')
    ap.add_argument("--target", type=float, default=7.0)
    ap.add_argument("--out", default="portfolio_summary.json")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    print(f"\nPortfolio Ladder · {len(tickers)} names · {args.start} → {args.end}")
    print(f"ladder: k={args.k} floor={args.floor:.0%} peak={'all-time' if args.peak_lookback==0 else str(args.peak_lookback)+'d'} reentry={args.reentry}")
    print(f"MA exit: {args.ma_period}-day, mode={args.ma_mode} · cash={args.cash_yield}% · target={args.target}%")
    print(f"basket: {', '.join(tickers)}\n")

    streams = {}
    for i, t in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {t} ...", end=" ", flush=True)
        try:
            s = per_name_streams(t, args.start, args.end, args.cash_yield, args.k,
                                 args.floor, args.peak_lookback, args.reentry,
                                 args.confirm_ema, args.ma_period)
            if s is None:
                print("skipped (no data)")
            else:
                streams[t] = s
                print(f"ok ({s.index[0].date()} → {s.index[-1].date()})")
        except Exception as e:
            print(f"ERROR: {e}")

    if len(streams) < 2:
        print("\nNeed at least 2 names with data."); return

    weightings = ["equal", "invvol"] if args.weighting == "both" else [args.weighting]
    ma_label = f"MA-{args.ma_period} ({args.ma_mode})"

    results, rows = {}, []
    for wname in weightings:
        vol_df = pd.DataFrame({t: s["vol"] for t, s in streams.items()})
        bh_df  = pd.DataFrame({t: s["bh"] for t, s in streams.items()})
        lad_df = pd.DataFrame({t: s["ladder"] for t, s in streams.items()})
        bh_port = combine(bh_df, wname, vol_df)

        ports = {
            "Buy & Hold":      bh_port,
            ma_label:          ma_portfolio(streams, bh_port, args.ma_period, args.ma_mode, args.cash_yield, wname),
            "Drawdown Ladder": combine(lad_df, wname, vol_df),
        }
        for label, port in ports.items():
            m = _metrics(port, label)
            if not m:
                continue
            results[(wname, label)] = m
            rows.append({"Weighting": wname, "Approach": label, "CAGR": m["cagr_pct"],
                         "MaxDD": m["maxdd_pct"], "Calmar": m["calmar"], "Sharpe": m["sharpe"],
                         ">=Tgt": "Y" if m["cagr_pct"] >= args.target else "-"})

    df_tab = pd.DataFrame(rows)
    print("\n" + "=" * 86)
    print("PORTFOLIO-LEVEL RESULTS  (what the whole basket would have experienced)")
    print("=" * 86)
    print(df_tab.to_string(index=False))
    print("=" * 86)

    for wname in weightings:
        bh = results.get((wname, "Buy & Hold")); lad = results.get((wname, "Drawdown Ladder")); ma = results.get((wname, ma_label))
        if bh and ma:
            dd = abs(bh["maxdd_pct"]) - abs(ma["maxdd_pct"]); rc = bh["cagr_pct"] - ma["cagr_pct"]
            ratio = "FREE" if rc <= 0.05 and dd > 0 else (round(dd / rc, 2) if rc > 0.05 else "none")
            print(f"[{wname}] {ma_label} vs B&H: DD {bh['maxdd_pct']:.1f}% → {ma['maxdd_pct']:.1f}% "
                  f"({dd:+.1f}pp), CAGR {bh['cagr_pct']:.1f}% → {ma['cagr_pct']:.1f}% · DD saved/pt: {ratio}")

    # Optional MA sweep
    sweep_payload = None
    if args.ma_sweep:
        periods = [int(x) for x in args.ma_sweep.split(",") if x.strip()]
        print("\n" + "-" * 86)
        print(f"MA SWEEP · mode={args.ma_mode} · gate only")
        print("-" * 86)
        sweep_payload = {}
        for wname in weightings:
            vol_df = pd.DataFrame({t: s["vol"] for t, s in streams.items()})
            bh_df  = pd.DataFrame({t: s["bh"] for t, s in streams.items()})
            bh_port = combine(bh_df, wname, vol_df)
            srows = []
            for mp in periods:
                g = ma_portfolio(streams, bh_port, mp, args.ma_mode, args.cash_yield, wname)
                m = _metrics(g, f"MA-{mp}")
                if m:
                    srows.append({"Weighting": wname, "MA": mp, "CAGR": m["cagr_pct"],
                                  "MaxDD": m["maxdd_pct"], "Calmar": m["calmar"]})
                    sweep_payload[f"{wname}|MA-{mp}"] = {kk: m[kk] for kk in ("cagr_pct", "maxdd_pct", "calmar")}
            print(pd.DataFrame(srows).to_string(index=False))
            print("-" * 86)

    payload = {
        "schema": "portfolio_ladder.v2",
        "config": {"tickers": list(streams.keys()), "date_range": [args.start, args.end],
                   "k": args.k, "floor": args.floor, "peak_lookback": args.peak_lookback,
                   "reentry": args.reentry, "ma_period": args.ma_period, "ma_mode": args.ma_mode,
                   "cash_yield_pct": args.cash_yield, "target_cagr_pct": args.target,
                   "note": "survivorship-biased basket; focus on relative drawdown reduction"},
        "results": {f"{w}|{a}": {kk: results[(w, a)][kk] for kk in
                                 ("cagr_pct", "maxdd_pct", "sharpe", "calmar", "_start")}
                    for (w, a) in results},
        "ma_sweep": sweep_payload,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nJSON written to: {args.out}  (paste into Claude)\n")


if __name__ == "__main__":
    main()
