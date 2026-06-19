# Walk-Forward Trader

A walk-forward backtesting and optimization tool built with Streamlit, Plotly, and yfinance. It pits the optimizer's **in-sample illusion** against **out-of-sample reality** so you can see how much of a strategy's apparent edge survives on unseen data.

The current strategy is a **momentum-pullback** system: it buys pullbacks inside a confirmed uptrend and rides them with an ATR/chandelier trailing stop, instead of relying on a lagging moving-average crossover.

## Strategy

| Stage   | Rule |
|---------|------|
| Regime  | `close > SMA200` **and** `SMA50 > SMA200` |
| Arm     | While flat and in regime, price dips to the pullback EMA (`low <= EMA`) |
| Entry   | Armed **and** `close > previous bar's high` (trend resuming) |
| Stop    | Initial = `entry - (atr_stop * ATR)`; ratchets up via a chandelier trail = `highest_close_since_entry - (atr_trail * ATR)` |
| Exit    | `close < effective stop` **or** `close < SMA200` **or** end of window |

Long only. A 0.10% exchange fee + 0.05% slippage is charged on both entry and exit.

## Walk-forward optimization

The optimizer grid-searches each rolling training window, then applies the frozen best parameters to the subsequent blind test window. Only out-of-sample returns are stitched together for the "Reality" equity curve.

| Parameter   | Meaning                                   | Search range |
|-------------|-------------------------------------------|--------------|
| `ema`       | Pullback EMA the price dips to            | 10 – 25 days |
| `atr_stop`  | Initial stop multiple                     | 1.5 – 3.0 × ATR |
| `atr_trail` | Chandelier trail multiple                 | 2.5 – 4.0 × ATR |

That's 64 parameter combos per fold. The objective is to **maximize CAGR subject to Sharpe ≥ 1.2**, falling back to best Sharpe if no combo clears the constraint.

Each fold starts flat with no inherited indicator state. The app pre-fetches 220+ warm-up bars before the start date so SMA200, the EMAs, and ATR are fully settled before the first traded bar of every fold.

## Running locally

```bash
pip install -r requirements.txt
streamlit run walk_forward_app.py
```

Streamlit opens at `http://localhost:8501`. Enter a ticker and date range in the sidebar; the dashboard shows a fold Gantt chart, in-sample vs out-of-sample degradation cards, an "Illusion vs. Reality" equity curve, and a color-coded trade log.

## Companion tools

These scripts share the same engine (`walk_forward_app.py`) and must be run from the same folder.

### `batch_runner.py`
Loops a basket of tickers through the walk-forward engine and prints a comparison table: active strategy (blind OOS) vs. regime-filtered buy & hold vs. raw buy & hold, with beat/miss verdicts against a target CAGR hurdle. Writes a combined JSON summary for pasting into a chat.

```bash
python batch_runner.py --tickers NVDA,AVGO,MSFT,META,SPY,QQQ --target 7
```

### `drawdown_ladder.py`
A continuous-exposure alternative to discrete in/out trading: exposure scales down smoothly with drawdown (`exposure = clamp(1 - k*drawdown, floor, 1)`) and rebuilds as price recovers — either automatically (`symmetric`) or only once price reclaims a short EMA (`confirm`, avoids buying a dead-cat bounce). Compares Buy & Hold vs. binary 200-day vs. the ladder on CAGR, max drawdown, and Calmar ratio across a basket.

```bash
python drawdown_ladder.py --tickers NVDA,META,MSFT,SPY --k 3 --floor 0.25 --reentry confirm
```

### `portfolio_ladder.py`
Applies the ladder overlay per-name, combines into a portfolio (equal-weight or inverse-volatility), and reports basket-level results — what you'd actually experience holding all the names together. Includes a tunable MA regime exit (`individual` per-name or one `basket`-level signal) and an optional MA-length sweep.

```bash
python portfolio_ladder.py --ma-period 200 --ma-mode individual --weighting both
```

### `gate_walkforward_app.py`
A Streamlit app that walk-forward validates the MA-gate overlay at the portfolio level: each fold picks the MA length that maximizes Calmar or Sharpe on the training window, applies it blind to the next test window, and stitches the OOS results — testing whether an MA length chosen on history actually generalizes.

```bash
streamlit run gate_walkforward_app.py
```

## Notes

This is a research and educational tool. Backtested results — especially over short windows — are not predictive of live performance and are not investment advice.
