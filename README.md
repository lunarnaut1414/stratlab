# StratLab

Algorithmic trading backtest gym built for AI agents. Design strategies in Python, test them against historical data, and train RL agents through a standard Gymnasium interface.

## Architecture

```
stratlab/
  data/          Market data fetching & caching (yfinance, ~850 tickers)
  engine/        Backtest engine + simulated broker
  strategies/    Strategy base class + 8 built-in templates
  indicators.py  Curated TA primitives (thin facade over `ta`)
  evaluation.py  walk_forward + compare_to_benchmark
  analytics/     Metrics, trade extraction, multi-panel tearsheet
  news/          NPR/BBC/AP/Kyodo scrapers + FinBERT sentiment
  gym/           Gymnasium-compatible RL environment
examples/        Quickstart scripts
tests/           64 deterministic tests (engine, trades, metrics, ...)
tmp/             Scratch directory for tearsheets and demo scripts
```

For deep usage detail and command reference, see [`CHEATSHEET.md`](./CHEATSHEET.md).

## Install

```bash
pip install -e .
```

## Quick Start

### Run a backtest

```python
from stratlab import Backtest, load_bars
from stratlab.strategies.sma_crossover import SMACrossover

data = load_bars("AAPL", start="2020-01-01", end="2024-01-01")
strategy = SMACrossover(fast=10, slow=30)

bt = Backtest(data={"AAPL": data}, strategy=strategy)
result = bt.run()

print(result.metrics)
# {'total_return': 0.12, 'sharpe': 0.85, 'max_drawdown': -0.15, ...}
```

### Write your own strategy

```python
from stratlab import Strategy, Order, BarContext
from stratlab.engine.broker import OrderSide

class MyStrategy(Strategy):
    def on_bar(self, ctx: BarContext):
        if ctx.idx < 20:
            return []
        # ctx.history() returns bars BEFORE today (today is hidden until fill).
        # iloc[-1] is yesterday's close — the most recent observable bar.
        closes = ctx.history()["close"]
        if closes.iloc[-1] > closes.iloc[-20:].mean():
            return [Order(side=OrderSide.BUY, size=100)]              # market order
            # — or, with a limit:
            # return [Order(side=OrderSide.BUY, size=100, limit_price=closes.iloc[-1] * 0.99)]
        return []
```

### Cross-sectional strategy across many symbols

```python
from stratlab import Backtest, Strategy, Order, BarContext, load_universe, sp500_tickers
from stratlab.engine.broker import OrderSide

class TopKMomentum(Strategy):
    """Each month, equal-weight the top K names by 12-1 month return."""
    def __init__(self, k=10, lookback=252, skip=21, rebalance=21):
        super().__init__()
        self.k, self.lookback, self.skip, self.rebalance = k, lookback, skip, rebalance

    def on_bar(self, ctx: BarContext):
        if ctx.idx < self.lookback or ctx.idx % self.rebalance:
            return []
        prices = ctx.closes_window(self.lookback)
        ret = prices.iloc[-self.skip] / prices.iloc[0] - 1.0
        winners = ret.dropna().sort_values().tail(self.k).index.tolist()

        orders = []
        for sym, pos in ctx.positions.items():
            if sym not in winners and pos.size > 0:
                orders.append(Order(side=OrderSide.SELL, size=pos.size, symbol=sym))

        budget = ctx.cash / max(len(winners), 1)
        for sym in winners:
            price = float(ctx.closes()[sym])
            size = budget // price
            if size > 0 and ctx.position(sym).size == 0:
                orders.append(Order(side=OrderSide.BUY, size=size, symbol=sym))
        return orders

data = load_universe(sp500_tickers(), start="2020-01-01")
bt = Backtest(data=data, strategy=TopKMomentum(k=20))
print(bt.run().metrics)
```

### Train an RL agent

```python
from stratlab import load_bars
from stratlab.gym.trading_env import TradingEnv

data = load_bars("SPY", start="2020-01-01")
env = TradingEnv(df=data, window_size=20, trade_size=50)

# Works with any Gymnasium-compatible RL library
# e.g. stable-baselines3, cleanrl, etc.
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action=1)  # buy
```

## Indicators

`stratlab.indicators` is a thin facade over `ta` exposing ~25 curated
primitives under one stable namespace:

```python
from stratlab.indicators import (
    sma, ema, wma, macd, macd_signal, macd_diff,
    rsi, roc, stoch, stoch_signal, atr, adx, cci,
    bb_upper, bb_lower, bb_middle, bb_pband,
    donchian_upper, donchian_lower,
    obv, mfi, cmf, vwap, aroon_up, aroon_down,
)
```

All take pandas `Series` and return a `Series` aligned to the input
index — safe to call inside `on_bar` on the sliced `ctx.history()`
frame. For exotic primitives (Williams %R, Ulcer, KAMA, Vortex, …)
import directly from `ta.momentum` / `ta.volatility` / `ta.trend` /
`ta.volume`.

## Out-of-sample evaluation

```python
from stratlab import Backtest, walk_forward, compare_to_benchmark, tearsheet

# Per-window metrics across rolling N-year slices.
wf = walk_forward(strategy, data, window_years=1.0)

# Strategy vs buy-and-hold of any ticker or price Series.
result = Backtest(data=..., strategy=...).run()
cmp = compare_to_benchmark(result, benchmark="SPY")

# 5-panel performance report (equity vs benchmark, drawdown, monthly heatmap,
# rolling Sharpe, trade scatter), saved as standalone interactive HTML.
fig = tearsheet(result, benchmark="SPY", title="My Strategy")
fig.write_html("tmp/strategy.html")
```

## Tests

64 deterministic tests on synthetic data — no yfinance, no FinBERT, no
HTTP. Whole suite runs in <1s. Locks in engine invariants
(no-look-ahead, cash conservation, fill semantics, limit-fill rules,
gap protection), trade extraction edge cases (flips, partial closes,
pyramiding), evaluation correctness, news-scraper resume guarantees,
and storage atomicity.

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Gym Environment

The `TradingEnv` follows the standard Gymnasium API:

| | |
|-|-|
| **Observation** | Rolling window of normalized OHLCV + position info |
| **Actions** | `0` = hold, `1` = buy, `2` = sell |
| **Reward** | Change in portfolio value (configurable: `pnl` or `log_return`) |

## Built-in Strategies

| Strategy | Module | Description |
|---|---|---|
| `SMACrossover` | `strategies.sma_crossover` | Fast/slow SMA crossover, single-asset |
| `Momentum` | `strategies.momentum` | RSI mean-reversion, single-asset |
| `MeanReversion` | `strategies.mean_reversion` | Bollinger-band reversion using **limit orders** |
| `DonchianBreakout` | `strategies.donchian_breakout` | Turtle-style 20/10 breakout, single-asset |
| `CrossSectionalFactor` | `strategies.cross_sectional` | Long top-K / short bottom-K by user-supplied factor (default 12-1 momentum) |
| `Pairs` | `strategies.pairs` | Z-score mean-reversion on a hand-picked symbol pair |
| `NewsOverlay` | `strategies.news_overlay` | Trend gated by daily aggregated news sentiment |
| `MomentumPlusInverse` | `strategies.momentum_plus_inverse` | Long top-K momentum + tactical SH hedge when SPY < 200d SMA |

## Performance Metrics

Every backtest returns:

**Return / risk**
- `total_return`, `cagr`, `sharpe`, `sortino`, `calmar`
- `max_drawdown`, `annual_volatility`
- `win_rate` — percentage of positive-return *bars* (not trades)

**Activity & costs**
- `n_trades` — individual fills
- `n_round_trips` — paired entries → exits (real "trades")
- `turnover_annualized` — total notional / avg equity, per year. A long/short
  monthly rebalance commonly runs 10x+; at 10 bps round-trip costs that's 100
  bps drag.
- `borrow_cost` — dollars charged on short notional (only nonzero when
  `borrow_rate_annual > 0`)
- `dropped_orders` — orders that couldn't fill (last bar, NaN open, etc.)

**Trade-level**
- `trade_win_rate` — fraction of round trips with positive PnL
- `profit_factor` — sum of winning PnL / |sum of losing PnL|. <1.0 means
  losses outweigh wins even before costs.
- `avg_winner_pnl`, `avg_loser_pnl`, `avg_holding_days`, `avg_trade_return`

`BacktestResult.trades` contains the full list of `Trade` records — symbol,
side (long/short), entry/exit time and price, size, gross PnL, return %.

## Broker Simulation

The simulated broker models:
- **Slippage** — configurable percentage (default 0.05%)
- **Commission** — configurable percentage (default 0.1%)
- **Long & short** — `pos.size` is signed. SELL beyond your long flips to short
  in one fill. Disable with `Backtest(allow_short=False)`.
- **Borrow cost** — annualized rate accrued daily on absolute short notional;
  set with `Backtest(borrow_rate_annual=0.005)` (50 bps/yr is a stylized
  general-collateral default; hard-to-borrow names are higher).
- **Position tracking** — size-weighted average entry, resets when a position
  crosses zero.

> **Margin is not enforced.** The broker doesn't check Reg-T or maintenance
> margin — strategies are responsible for sizing within reasonable leverage.

## Execution model

**Same-bar limit-intraday with structural look-ahead prevention.** Each bar `i`:

1. **`on_bar(ctx)` runs before bar `i` is observed.** `ctx.history()`,
   `ctx.closes()`, and `ctx.closes_window()` return data through bar `i-1`
   only. There is *no API* a strategy can use to read today's open, high,
   low, or close — look-ahead is prevented by construction, not by
   convention. `ctx.idx == i` still names the bar where any returned
   orders will execute, but its data is invisible until after fills.
2. **Each returned `Order` is checked against bar `i`'s OHLC range:**

   | Order | Fills when | Fill price |
   |---|---|---|
   | `Order(BUY, size)` (market) | always | `bar.open × (1 + slippage_pct)` |
   | `Order(SELL, size)` (market) | always | `bar.open × (1 − slippage_pct)` |
   | `Order(BUY, size, limit_price=L)` | `bar.low ≤ L` | `min(L, bar.open)` — gap-down gives the better gap-open price |
   | `Order(SELL, size, limit_price=L)` | `bar.high ≥ L` | `max(L, bar.open)` — gap-up gives the better gap-open price |

   Limit orders **don't get slippage applied** — you specified the price.
   Limits whose range condition isn't met are dropped (counted in
   `metrics["dropped_orders"]`).
3. **Equity[i] is marked to bar `i`'s close** *after* fills, so the
   position-value side reflects today's mark.

### Same-bar round-trips

A strategy that submits a paired `BUY` limit + `SELL` limit on the same
bar can produce a true intra-day round-trip when today's range crosses
both limits. `MeanReversion` is the showcase — submits a buy limit at
the lower Bollinger band and a sell limit at the upper, on every bar.

### What this model deliberately doesn't capture

| Real-world mechanic | Modeled? |
|---|---|
| Borrow / locate availability for shorts | ❌ assumed available |
| Reg-T / maintenance margin | ❌ not enforced |
| Hard-to-borrow rate per name | ❌ flat `borrow_rate_annual` |
| Recall / forced cover | ❌ |
| Partial fills | ❌ orders fill in full or not at all |
| Market impact | ❌ a 1-share order fills at the same price as a 1M-share order |
| Intraday tick-by-tick path | ❌ only OHLC range matters |
| Options | ❌ pure cash equities (and equity ETFs) |
| Borrow cost on short notional | ✅ accrued daily on absolute short notional |
| Slippage (market) | ✅ |
| Commission | ✅ |
| Cross-sectional alignment & late-listed names | ✅ |
| Borrow accrual over weekends/holidays | ✅ calendar-day basis |

For daily-bar Yahoo data this set of simplifications is industry-standard.
Going to intraday data or live trading would require extending the broker.

## Data

Market data is fetched via yfinance and cached locally as CSV files. By
default the cache lives at `<project_root>/data/market/` if you're inside a
project (detected via `pyproject.toml` or `.git`); otherwise it falls back to
`~/.stratlab/cache/`. Override with `STRATLAB_CACHE_DIR=...`.

### Layout

```
data/market/
  catalog.json                  # ticker → sector/category map (auto-built)
  indices/
    sp500.json, nasdaq100.json, dow30.json    # ticker lists (constituents)
    volatility/^VIX_1d.csv, ^VVIX_1d.csv, ^MOVE_1d.csv, ^SKEW_1d.csv, ...
    equity/^GSPC_1d.csv, ^DJI_1d.csv, ^NDX_1d.csv, ^IXIC_1d.csv, ...
    international/^FTSE_1d.csv, ^N225_1d.csv, ^HSI_1d.csv, ...
    rates/^IRX_1d.csv, ^FVX_1d.csv, ^TNX_1d.csv, ^TYX_1d.csv
    currency/DX-Y.NYB_1d.csv
  stocks/                       # 11 GICS sectors
    information_technology/AAPL_1d.csv
    financials/JPM_1d.csv
    energy/XOM_1d.csv
    ...
  etfs/                         # 15 categories
    broad_market/, factor/, sector/, industry/, thematic/,
    international_developed/, international_emerging/,
    bonds/, commodities/, real_estate/, currency/, volatility/,
    crypto/, leveraged/, inverse/
  futures/                      # continuous contracts (Yahoo `=F`)
    energy/CL=F_1d.csv, NG=F_1d.csv, BZ=F_1d.csv, ...
    metals/GC=F_1d.csv, SI=F_1d.csv, HG=F_1d.csv, ...
    grains/, softs/, meats/, lumber/
    equity_index/ES=F_1d.csv, NQ=F_1d.csv, ...
    rates/ZB=F_1d.csv, ZN=F_1d.csv, ...
    currency/6E=F_1d.csv, 6J=F_1d.csv, ...
    crypto/BTC=F_1d.csv, ETH=F_1d.csv
  uncategorized/                # tickers not in the catalog
```

`catalog.json` is the authoritative ticker → category map (~810 entries:
~503 stocks + ~270 ETFs + ~32 indices + ~46 futures). Inspect it to see what
we know about each symbol.

Volatility indices (`^VIX`, `^VVIX`, `^MOVE`, …) are *index levels*, not
ETP wrappers. They have no daily-rebalance decay so they're cleaner inputs
than VXX/UVXY for vol-aware strategies.

### Refresh the local cache

The daily incremental — market data + recent news from 4 sources, ~7-day
window, all in parallel:

```bash
python -m stratlab.refresh_all                   # everything, parallel (default)
python -m stratlab.refresh_all --news-only       # skip the market step
python -m stratlab.refresh_all --with-sentiment  # also score new articles with FinBERT
```

Or run a single pipeline:

```bash
python -m stratlab.refresh                       # market data only (~850 tickers)
python -m stratlab.refresh --tickers AAPL MSFT   # specific tickers

python -m stratlab.news.npr                      # NPR scraper (date-archive)
python -m stratlab.news.bbc                      # BBC scraper (RSS)
python -m stratlab.news.kyodo                    # Kyodo News English (per-year sitemaps)
```

AP and CNA were previously included as latest-only sources but were
removed because their public surfaces don't expose historical archives.
See `docs/archive/news_scrapers_ap_cna.md` if you want to revive
either as a daily-only feed.

For deep historical news, the dedicated backfill job is **separate** from
the daily refresh — three sources expose public archives:

| Source | Archive depth |
|---|---|
| NPR | back to ~2000 (date-archive walker) |
| BBC | back to ~2009-09 (XML sitemap, ~120 child files) |
| Kyodo | back to 2017 (per-year sitemaps, ~10K articles/year) |

```bash
python -m stratlab.news.backfill --since 2017-01-01 --workers 6   # all three, sequential
python -m stratlab.news.backfill --days 365 --sources npr         # NPR only
python -m stratlab.news.backfill --since 2017-01-01 --sources kyodo --parallel
```

Backfills are **resumable** (verified by tests). NPR skips at the day
level; BBC and Kyodo skip at the article-slug level by indexing every
on-disk slug at startup — re-runs only HTTP-fetch the gaps. Each
scraper flushes to disk every 20 articles or 30 seconds, so Ctrl+C is
safe.

To enable FinBERT sentiment (optional dep — pulls torch + transformers):

```bash
pip install -e ".[sentiment]"
python -m stratlab.news.sentiment                  # score every unscored article
python -m stratlab.news.sentiment --sources kyodo  # one source at a time
```

Auto-picks CUDA → MPS → CPU. ~10-100 articles/sec depending on hardware.
Scored articles get a `sentiment` payload with `pos/neg/neutral/net`
probabilities written back to the same JSON. Aggregated daily features
load via `daily_sentiment(start, end, sources, topics)`.

By default refresh uses yfinance's `period="max"` for cold fetches, so each
ticker gets its full available history per its own inception (AAPL → 1980,
NVDA → 1999, ABNB → 2020 IPO, ^GSPC → 1927, HPE → 2015 spinoff). Pass an
explicit `--start` to truncate.

The refresh module categorizes each ticker into one of four buckets and only
hits the network where needed:

- **cold** — no cache, fetch full range
- **backfill** — cache exists but doesn't cover `--start`, re-fetch full range
- **warm** — cache covers `--start` but stops before the most recent business
  day, fetch only the gap on the right
- **up to date** — cache already covers both edges, skipped without a network
  call

Orphan files from older cache layouts are swept on every run. Final summary
reports counts per bucket, new bars added, total cache size on disk, and any
failures.

## For AI Agents

StratLab is designed as an environment for AI agents to:

1. **LLM agents**: Write a `Strategy` subclass, call `Backtest.run()`, read the metrics dict
2. **RL agents**: Train directly via `TradingEnv` using any Gymnasium-compatible library
3. **Hybrid**: Use an LLM to generate strategy code, backtest it, iterate on results

The strategy interface is intentionally simple — `on_bar(idx, history) -> list[Order]` — so agents can generate and test ideas with minimal boilerplate.
