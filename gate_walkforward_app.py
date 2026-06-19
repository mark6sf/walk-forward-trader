"""
Per-Name MA Gate — Walk-Forward Validation (web UI)
===================================================
Validates the winning overlay out-of-sample at the PORTFOLIO level.

Each fold: pick the MA length that maximizes the objective (Calmar or Sharpe)
on the training window, apply it BLIND to the next test window, stitch the
out-of-sample results. Tests whether choosing the MA on history generalizes.

Run:  streamlit run gate_walkforward_app.py
Needs walk_forward_app.py + portfolio_ladder.py in the same folder.
Analysis to inform your own decisions, not investment advice.
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

import walk_forward_app as wf
import portfolio_ladder as pl

MOMENTUM_BASKET = pl.MOMENTUM_BASKET


# ── data (module-level so tests can monkeypatch) ─────────────────────
def fetch(ticker, start, end):
    return wf.fetch_data(ticker, str(start), str(end))   # (df, df_full)


# ── gating ───────────────────────────────────────────────────────────
def gate_returns(close, dates, ma_period, cash_yield, cost):
    """Binary per-name MA gate: in-market when prior close > MA, else cash.
    Re-entry is the symmetric recross (same line, lagged one day)."""
    sma = wf.compute_sma(close, ma_period)
    rf = (1.0 + cash_yield / 100.0) ** (1.0 / 252.0) - 1.0
    n = len(close)
    out = np.zeros(n)
    prev = None
    for t in range(1, n):
        in_mkt = close[t - 1] > sma[t - 1]
        out[t] = (close[t] / close[t - 1] - 1.0) if in_mkt else rf
        if prev is not None and in_mkt != prev:
            out[t] -= cost
        prev = in_mkt
    return pd.Series(out, index=dates)


def _obj(returns: np.ndarray, kind: str):
    if len(returns) < 20:
        return -np.inf
    eq = (1 + returns).cumprod()
    if kind == "sharpe":
        return wf.sharpe_ratio(returns)
    cg = wf.cagr(eq, len(returns))
    mdd = wf.max_drawdown(eq)
    return cg / abs(mdd) if abs(mdd) > 1e-6 else -np.inf


def _metrics(ret: np.ndarray):
    eq = (1 + ret).cumprod()
    cg = wf.cagr(eq, len(ret)) * 100
    mdd = wf.max_drawdown(eq) * 100
    return {"cagr_pct": round(cg, 2), "maxdd_pct": round(mdd, 2),
            "sharpe": round(wf.sharpe_ratio(ret), 2),
            "calmar": round(cg / abs(mdd), 2) if abs(mdd) > 1e-6 else None}


def _gen_folds(index, train_m, test_m):
    start, end = index[0], index[-1]
    cursor = start
    folds = []
    while True:
        tr_end = cursor + relativedelta(months=train_m)
        te_end = tr_end + relativedelta(months=test_m)
        if tr_end >= end:
            break
        train = index[(index >= cursor) & (index < tr_end)]
        test = index[(index >= tr_end) & (index < min(te_end, end + pd.Timedelta(days=1)))]
        if len(train) > 60 and len(test) > 5:
            folds.append((cursor, tr_end, min(te_end, end), train, test))
        cursor = cursor + relativedelta(months=test_m)
        if te_end >= end:
            break
    return folds


def run_gate_walkforward(tickers, start, end, train_m, test_m, ma_grid,
                         weighting, cash_yield, objective="calmar",
                         progress=None):
    cost = wf.COST_PER_TRADE
    # Fetch + build per-name series on warm (df_full) data, restricted to df range
    gated = {ma: {} for ma in ma_grid}     # ma -> {ticker: Series}
    bh = {}                                # ticker -> Series (asset returns)
    vol = {}                               # ticker -> Series (trailing vol)
    used = []
    for i, t in enumerate(tickers):
        if progress:
            progress(i / len(tickers), f"Fetching {t}")
        res = fetch(t, start, end)
        if res is None:
            continue
        df, df_full = res
        if df is None or len(df) < 60:
            continue
        cf = df_full["Close"].values
        full_dates = df_full.index
        rng = df.index
        for ma in ma_grid:
            g = gate_returns(cf, full_dates, ma, cash_yield, cost).reindex(rng)
            gated[ma][t] = g
        ar = pd.Series(np.concatenate([[0.0], cf[1:] / cf[:-1] - 1.0]),
                       index=full_dates).reindex(rng)
        bh[t] = ar
        vol[t] = ar.rolling(60, min_periods=20).std().shift(1)
        used.append(t)

    if len(used) < 2:
        return None

    union = sorted(set().union(*[set(bh[t].index) for t in used]))
    union = pd.DatetimeIndex(union)
    vol_df = pd.DataFrame({t: vol[t] for t in used}).reindex(union)
    bh_df = pd.DataFrame({t: bh[t] for t in used}).reindex(union)
    ret_df_by_ma = {ma: pd.DataFrame({t: gated[ma][t] for t in used}).reindex(union)
                    for ma in ma_grid}

    folds = _gen_folds(union, train_m, test_m)
    if not folds:
        return None

    fold_rows = []
    oos_parts = []
    for fi, (c0, tr_end, te_end, tr_idx, te_idx) in enumerate(folds):
        if progress:
            progress(0.5 + 0.5 * fi / len(folds), f"Fold {fi+1}/{len(folds)}")
        # choose MA on train
        best_ma, best_obj = None, -np.inf
        for ma in ma_grid:
            port_tr = pl.combine(ret_df_by_ma[ma].loc[tr_idx], weighting,
                                 vol_df.loc[tr_idx]).dropna()
            o = _obj(port_tr.values, objective)
            if o > best_obj:
                best_obj, best_ma = o, ma
        # apply blind to test
        port_te = pl.combine(ret_df_by_ma[best_ma].loc[te_idx], weighting,
                             vol_df.loc[te_idx]).dropna()
        if len(port_te) < 5:
            continue
        oos_parts.append(port_te)
        om = _metrics(port_te.values)
        port_tr_best = pl.combine(ret_df_by_ma[best_ma].loc[tr_idx], weighting,
                                  vol_df.loc[tr_idx]).dropna()
        im = _metrics(port_tr_best.values)
        fold_rows.append({
            "fold": fi + 1,
            "train": [str(tr_idx[0].date()), str(tr_idx[-1].date())],
            "test_end": str(te_idx[-1].date()),
            "chosen_ma": best_ma,
            "is_calmar": im["calmar"], "is_sharpe": im["sharpe"],
            "oos_cagr": om["cagr_pct"], "oos_maxdd": om["maxdd_pct"],
            "oos_calmar": om["calmar"], "oos_sharpe": om["sharpe"],
            "names": int(ret_df_by_ma[best_ma].loc[te_idx].notna().any().sum()),
        })

    if not oos_parts:
        return None
    oos = pd.concat(oos_parts)
    oos = oos[~oos.index.duplicated(keep="first")].sort_index()
    oos_eq = (1 + oos.values).cumprod()
    oos_m = _metrics(oos.values)

    # buy & hold over the same OOS dates
    bh_port = pl.combine(bh_df, weighting, vol_df).reindex(oos.index).dropna()
    bh_eq = (1 + bh_port.values).cumprod()
    bh_m = _metrics(bh_port.values)

    import collections
    ma_stab = dict(collections.Counter(r["chosen_ma"] for r in fold_rows))
    med_is = float(np.median([r["is_calmar"] for r in fold_rows if r["is_calmar"] is not None]))
    med_oos = float(np.median([r["oos_calmar"] for r in fold_rows if r["oos_calmar"] is not None]))

    return {
        "config": {"tickers": used, "date_range": [str(start), str(end)],
                   "train_months": train_m, "test_months": test_m,
                   "ma_grid": list(ma_grid), "weighting": weighting,
                   "cash_yield_pct": cash_yield, "objective": objective,
                   "num_folds": len(fold_rows)},
        "oos": oos_m, "buy_hold_oos": bh_m,
        "degradation": {"median_is_calmar": round(med_is, 2),
                        "median_oos_calmar": round(med_oos, 2),
                        "calmar_retained_pct": round(100 * med_oos / med_is, 0) if med_is > 0 else None,
                        "oos_beats_buy_hold_calmar": bool((oos_m["calmar"] or 0) > (bh_m["calmar"] or 0))},
        "ma_stability": ma_stab,
        "folds": fold_rows,
        "_oos_dates": oos.index, "_oos_eq": oos_eq, "_bh_eq": bh_eq,
    }


# ──────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ──────────────────────────────────────────────────────────────────────
def main():
    import streamlit as st
    import plotly.graph_objects as go

    st.set_page_config(page_title="MA Gate — Walk-Forward Validation", layout="wide")
    st.markdown("""<style>
      .stApp { background:#0b0f1a; }
      .metric-card { background:#141a2e; border:1px solid #2e3a50; border-radius:10px;
                     padding:16px; text-align:center; }
      .metric-label { color:#8aa0c0; font-size:0.72rem; letter-spacing:0.04em; text-transform:uppercase; }
      .metric-val { font-size:1.7rem; font-weight:700; margin-top:4px; }
      .section-header { color:#C8E0FF; font-size:1.15rem; font-weight:700;
                        margin:18px 0 8px; border-left:3px solid #636EFA; padding-left:10px; }
      .green { color:#00CC96; } .red { color:#EF553B; } .white { color:#E8ECF0; }
    </style>""", unsafe_allow_html=True)

    st.title("🛡️ Per-Name MA Gate — Walk-Forward Validation")
    st.caption("Out-of-sample test at the portfolio level. Each fold picks the MA length on "
               "training data and applies it blind to the next test window. The OOS column is "
               "the honest number. Today's momentum basket is survivorship-biased — read the "
               "relative drawdown reduction, not the absolute return.")

    sb = st.sidebar
    sb.header("Settings")
    tickers_s = sb.text_area("Tickers (comma-separated)", ", ".join(MOMENTUM_BASKET), height=80)
    c1, c2 = sb.columns(2)
    start = c1.date_input("Start", value=datetime(2018, 1, 1))
    end = c2.date_input("End", value=datetime.today())
    train_m = sb.slider("Training window (months)", 12, 48, 24, step=6)
    test_m = sb.slider("Blind test window (months)", 3, 12, 6, step=3)
    ma_text = sb.text_input("MA grid (days, comma-separated)", "150,175,200,225,250")
    weighting = sb.selectbox("Weighting", ["equal", "invvol"], index=0)
    objective = sb.selectbox("Optimize MA for", ["calmar", "sharpe"], index=0)
    cash_yield = sb.number_input("Cash yield (annual %)", 0.0, 10.0, 3.7, 0.1)
    go_btn = sb.button("▶ Run Walk-Forward", use_container_width=True, type="primary")

    if not go_btn:
        st.info("Set your basket and parameters, then press **Run Walk-Forward**.")
        return

    tickers = [t.strip().upper() for t in tickers_s.replace("\n", ",").split(",") if t.strip()]
    ma_grid = sorted({int(x) for x in ma_text.split(",") if x.strip()})

    bar = st.progress(0.0, text="Starting…")
    res = run_gate_walkforward(tickers, start, end, train_m, test_m, ma_grid,
                               weighting, cash_yield, objective,
                               progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
    bar.empty()
    if res is None:
        st.error("Not enough data / folds. Check tickers and date range.")
        return

    oos, bh = res["oos"], res["buy_hold_oos"]
    deg = res["degradation"]

    def card(label, val, cls="white"):
        return (f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-val {cls}">{val}</div></div>')

    st.markdown('<div class="section-header">Out-of-Sample (blind) vs Buy & Hold</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].markdown(card("OOS CAGR", f'{oos["cagr_pct"]:+.1f}%', "green" if oos["cagr_pct"] > 0 else "red"), unsafe_allow_html=True)
    cols[1].markdown(card("OOS Max Drawdown", f'{oos["maxdd_pct"]:.1f}%', "green"), unsafe_allow_html=True)
    cols[2].markdown(card("OOS Calmar", f'{oos["calmar"]}', "green"), unsafe_allow_html=True)
    cols[3].markdown(card("OOS Sharpe", f'{oos["sharpe"]}', "white"), unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].markdown(card("B&H CAGR", f'{bh["cagr_pct"]:+.1f}%', "white"), unsafe_allow_html=True)
    cols[1].markdown(card("B&H Max Drawdown", f'{bh["maxdd_pct"]:.1f}%', "red"), unsafe_allow_html=True)
    cols[2].markdown(card("B&H Calmar", f'{bh["calmar"]}', "white"), unsafe_allow_html=True)
    verdict_cls = "green" if deg["oos_beats_buy_hold_calmar"] else "red"
    cols[3].markdown(card("OOS beats B&H?", "YES" if deg["oos_beats_buy_hold_calmar"] else "NO", verdict_cls), unsafe_allow_html=True)

    st.markdown('<div class="section-header">Generalization (IS → OOS)</div>', unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    g1.markdown(card("Median IS Calmar", f'{deg["median_is_calmar"]}'), unsafe_allow_html=True)
    g2.markdown(card("Median OOS Calmar", f'{deg["median_oos_calmar"]}',
                     "green" if deg["median_oos_calmar"] >= 0.8 * deg["median_is_calmar"] else "red"), unsafe_allow_html=True)
    g3.markdown(card("Calmar retained", f'{deg["calmar_retained_pct"]}%',
                     "green" if (deg["calmar_retained_pct"] or 0) >= 70 else "red"), unsafe_allow_html=True)
    st.caption(f"Chosen-MA stability across folds: {res['ma_stability']}  "
               f"(clustered = robust; scattered = fitting noise)")

    # Equity curve
    st.markdown('<div class="section-header">Stitched Out-of-Sample Equity</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res["_oos_dates"], y=res["_oos_eq"], name="MA Gate (OOS)",
                             line=dict(color="#00CC96", width=2)))
    fig.add_trace(go.Scatter(x=res["_oos_dates"], y=res["_bh_eq"], name="Buy & Hold",
                             line=dict(color="#636EFA", width=1.5, dash="dot")))
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0b0f1a", plot_bgcolor="#0b0f1a",
                      height=380, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    # Fold table
    st.markdown('<div class="section-header">Per-Fold Detail</div>', unsafe_allow_html=True)
    fr = pd.DataFrame(res["folds"])
    fr_display = fr.assign(train=fr["train"].apply(lambda x: f"{x[0]}→{x[1]}")).rename(columns={
        "fold": "Fold", "train": "Train", "test_end": "Test End", "chosen_ma": "MA",
        "is_calmar": "IS Calmar", "oos_cagr": "OOS CAGR%", "oos_maxdd": "OOS MaxDD%",
        "oos_calmar": "OOS Calmar", "names": "Names"})
    st.dataframe(fr_display[["Fold", "Train", "Test End", "MA", "IS Calmar",
                             "OOS CAGR%", "OOS MaxDD%", "OOS Calmar", "Names"]],
                 use_container_width=True, hide_index=True)

    # JSON export
    st.markdown('<div class="section-header">🤖 Analysis Export</div>', unsafe_allow_html=True)
    export = {k: v for k, v in res.items() if not k.startswith("_")}
    js = json.dumps(export, indent=2, default=str)
    import html as _html
    st.markdown(f'<pre style="background:#0d1320;color:#C8E0FF;border:1px solid #2e3a50;'
                f'border-radius:8px;padding:14px;font-size:0.72rem;max-height:380px;overflow:auto;">'
                f'{_html.escape(js)}</pre>', unsafe_allow_html=True)
    st.download_button("⬇️ Download (JSON)", js,
                       file_name=f"gate_wf_{datetime.today().strftime('%Y%m%d')}.json",
                       mime="application/json", use_container_width=True)


if __name__ == "__main__":
    main()
