# StratLab Cheatsheet

The minimum you need to know to run, query, and backtest.

---

## Daily refresh — the one-command answer

```bash
python -m stratlab.refresh_all              # market + all news, parallel (default)
python -m stratlab.refresh_all --news-only  # all news in parallel, skip market
python -m stratlab.refresh_all --serial     # one at a time, clean output
python -m stratlab.refresh_all --quiet      # only summaries
```

Pulls market data (~850 tickers) and recent articles (~7-day window)
from NPR, BBC, AP, and CNA. Pipelines run in parallel by default
(different domains don't compete on rate-limits), so wall-clock is
`max(market, npr, bbc, ap, cna)` — typically 5-15 minutes warm. Output
interleaves; `--serial` for clean ordered output. Idempotent:
re-running the same day finishes in seconds.

`refresh_all` is for the *incremental* job — what's new since
yesterday. For historical backfill, see the dedicated `news.backfill`
command below.

---

## Deep news backfill

`stratlab.news.backfill` is the one-shot job for pulling historical
news from the only two sources whose archives expose it:

| Source | Archive depth | Mechanism |
|---|---|---|
| NPR | back to ~2000 | date-archive walker (`/sections/<topic>/archive?date=...`) |
| BBC | back to ~2009 | XML sitemap (`https-index-com-archive.xml`) |
| AP  | none | (run daily refresh on cron to accumulate) |
| CNA | none | (run daily refresh on cron to accumulate) |

```bash
# 1 year of NPR + BBC, run sequentially (default)
python -m stratlab.news.backfill --days 365

# Absolute date range
python -m stratlab.news.backfill --since 2020-01-01

# Just one source
python -m stratlab.news.backfill --since 2020-01-01 --sources npr

# Run NPR and BBC in parallel; crank per-source workers higher
python -m stratlab.news.backfill --since 2020-01-01 --parallel --workers 8
```

The backfill is idempotent (per-day file existence ⇒ skip) so a crashed
run can be re-invoked safely.

The first run is slow (~5-10 min) because it cold-fetches max history per
ticker. Every run after that just appends yesterday's bar.

---

## When to run it

- **End of trading day** (after 4:30 PM ET): yesterday's bar is final, today's
  is closing. Run any time after that.
- **First thing in the morning**: same-day data may be stale until evening,
  but everything older is current.
- **Skip weekends**: nothing to fetch (script no-ops politely).

---

## Or split it

Sometimes you want one without the others:

```bash
python -m stratlab.refresh                    # market only
python -m stratlab.news.npr                   # NPR only (date-archive walker)
python -m stratlab.news.bbc                   # BBC only (RSS-driven)
python -m stratlab.news.ap                    # AP only (topic-hub walker)
python -m stratlab.news.cna                   # CNA only (Singapore, sitemap-feed)
```

Each news source has its own topic vocabulary and own `--topics` choices.
NPR has a date archive so it supports historical backfill (`--full` or
`--start ...`). BBC and AP only expose recent articles, so they run in
"latest" mode and accumulate coverage when run on a daily cadence.

---

## Common variations

```bash
# Specific tickers (any subset of the universe + arbitrary Yahoo symbols)
python -m stratlab.refresh --tickers AAPL MSFT GOOGL
python -m stratlab.refresh --tickers '^VIX' '^VVIX' 'CL=F' 'GC=F'

# Truncate history (default is yfinance period="max" — each ticker's full
# inception-to-today range, varies per ticker)
python -m stratlab.refresh --start 2010-01-01

# News for a specific date range / topic
python -m stratlab.news.npr --topics economy technology --start 2024-01-01 --end 2024-12-31

# Backfill all of NPR back to 2000 (slow — many hours; use --workers)
python -m stratlab.news.npr --full --workers 4

# Parallelize across topics (1 session per worker; 4-8 is the sweet spot)
python -m stratlab.news.npr --workers 4

# Quiet mode (only summary, no per-day logging)
python -m stratlab.news.npr --quiet

# BBC historical backfill via sitemap (covers ~2009 → today; --years scopes depth)
python -m stratlab.news.bbc --from-sitemap --years 1 --workers 4
python -m stratlab.news.bbc --from-sitemap --since 2024-01-01 --workers 4
```

News storage is one JSON per `(source, topic, day)`:
`data/news/<source>/<topic>/<YYYY>/<source>-<topic>-<YYYY-MM-DD>.json`. The
filename is self-describing so a single JSON copied or shared out of context
is still unambiguous. Resume is by file existence — if the day file exists
(even empty), the scraper skips that day without an HTTP request. Legacy
year-based files have been backed up to `data/news/_legacy_yearly_backup/`.

---

## News sentiment (FinBERT)

Scrapers fetch articles; sentiment scoring is a separate step (opt-in,
needs `pip install -e ".[sentiment]"` for torch + transformers).

```bash
# Score every unscored article on disk (uses CUDA → MPS → CPU automatically)
python -m stratlab.news.sentiment

# Just one source
python -m stratlab.news.sentiment --sources cna

# Only recent files (resumable: already-scored articles are skipped)
python -m stratlab.news.sentiment --since 2024-01-01

# Or fold it into the daily refresh
python -m stratlab.refresh_all --with-sentiment
```

Each scored article gains a `sentiment` field with the FinBERT class
probabilities and a `net = pos - neg` summary, written back to the same
JSON. Aggregated daily features are loaded via `daily_sentiment()`:

```python
from stratlab import daily_sentiment

# Default: one column per (source, topic), values = mean net sentiment
sent = daily_sentiment(start="2024-01-01", end="2024-12-31",
                       sources=["npr", "ap", "cna"], topics=["business"])

# Full breakdown: pos/neg/neutral/article_count per (source, topic)
sent_full = daily_sentiment(start="2024-01-01", breakdown=True)
```

Throughput: ~10-20 articles/sec on Apple MPS, ~50-100/sec on a discrete
GPU, ~2-5/sec on CPU. The first run also downloads `ProsusAI/finbert`
(~440MB) into the HuggingFace cache.

---

## Set it and forget it (cron)

Add to your crontab (`crontab -e`):

```cron
# Daily refresh, 6:30 PM ET, log to /tmp/stratlab.log
30 18 * * 1-5  cd ~/stratlab/stratlab && ./.venv/bin/python -m stratlab.refresh_all >> /tmp/stratlab.log 2>&1
```

`1-5` = Mon–Fri only. Adjust the path if your project lives elsewhere.

On macOS, `launchd` is the more native option but cron works fine.

---

## Where the data lives

```
data/
├── market/                              # refreshed by stratlab.refresh
│   ├── catalog.json                     # ticker → category map (~810 entries)
│   ├── indices/
│   │   ├── sp500.json, nasdaq100.json, dow30.json    # ticker lists
│   │   ├── volatility/^VIX_1d.csv  ^VVIX_1d.csv  ^MOVE_1d.csv  ...
│   │   ├── equity/^GSPC_1d.csv  ^DJI_1d.csv  ...
│   │   ├── international/^FTSE_1d.csv  ^N225_1d.csv  ...
│   │   ├── rates/^TNX_1d.csv  ^FVX_1d.csv  ...
│   │   └── currency/DX-Y.NYB_1d.csv
│   ├── stocks/
│   │   ├── information_technology/AAPL_1d.csv  MSFT_1d.csv  ...
│   │   ├── financials/JPM_1d.csv  ...
│   │   └── ... (11 GICS sectors)
│   ├── etfs/
│   │   ├── broad_market/SPY_1d.csv  ...
│   │   ├── leveraged/TQQQ_1d.csv  UPRO_1d.csv  ...
│   │   ├── inverse/SH_1d.csv  SQQQ_1d.csv  ...
│   │   └── ... (15 categories)
│   ├── futures/
│   │   ├── energy/CL=F_1d.csv  NG=F_1d.csv  ...
│   │   ├── metals/GC=F_1d.csv  SI=F_1d.csv  ...
│   │   ├── equity_index/ES=F_1d.csv  NQ=F_1d.csv  ...
│   │   └── ... (10 categories)
│   └── uncategorized/                  # tickers not in the catalog
└── news/                                # refreshed by stratlab.news.npr
    └── npr/
        ├── economy/2024.json  2023.json  ...
        ├── technology/...
        └── ... (8 topics × N years)
```

---

## Loading data in Python

```python
from stratlab import (
    load_bars,                    # one ticker
    load_universe,                # many tickers at once
    sp500_tickers, nasdaq100_tickers, dow30_tickers,
    popular_etfs, inverse_etfs, leveraged_etfs,
    volatility_indices, equity_indices, international_indices, rate_indices,
    commodity_futures, equity_index_futures, rate_futures, currency_futures,
    default_universe,             # everything, deduped
)

# Single ticker
df = load_bars("AAPL", start="2020-01-01")

# Cross-sectional / portfolio data
data = load_universe(sp500_tickers(), start="2020-01-01")
# → {"AAPL": DataFrame, "MSFT": DataFrame, ...}

# A "macro overlay" basket
macro = load_universe(volatility_indices() + rate_indices() + equity_indices())
```

---

## Technical-analysis primitives

`stratlab.indicators` exposes ~25 curated indicators (thin facade over the
`ta` package, no reimplementation). All take pandas `Series` and return a
`Series` aligned to the input index — safe to call inside `on_bar` on the
sliced `ctx.history()` frame.

```python
from stratlab.indicators import (
    sma, ema, wma,
    macd, macd_signal, macd_diff,
    adx, aroon_up, aroon_down, cci,
    rsi, roc, stoch, stoch_signal,
    atr, bb_upper, bb_lower, bb_middle, bb_pband,
    donchian_upper, donchian_lower,
    obv, mfi, cmf, vwap,
)

df = load_bars("AAPL", start="2023-01-01")
df["rsi14"] = rsi(df["close"], window=14)
df["atr14"] = atr(df["high"], df["low"], df["close"], window=14)
df["macd"]  = macd(df["close"])
```

Need something exotic (Williams %R, Ulcer, KAMA, Vortex, ...)? Import
directly from `ta.momentum` / `ta.volatility` / `ta.trend` / `ta.volume` —
all functional, same calling shape.

---

## Running a backtest

### Single-asset (built-in strategy)

```python
from stratlab import Backtest, load_bars
from stratlab.strategies.sma_crossover import SMACrossover

data = load_bars("AAPL", start="2020-01-01", end="2024-01-01")
bt = Backtest(data={"AAPL": data}, strategy=SMACrossover(fast=10, slow=30))
result = bt.run()
print(result.metrics)        # sharpe, max_drawdown, turnover, win_rate, ...
print(result.trades[:3])      # round-trip trades
```

### Built-in strategy templates

```python
from stratlab.strategies.sma_crossover import SMACrossover       # single-asset trend
from stratlab.strategies.momentum import Momentum                 # RSI mean-reversion
from stratlab.strategies.mean_reversion import MeanReversion      # Bollinger reversion
from stratlab.strategies.donchian_breakout import DonchianBreakout # turtle-style breakout
from stratlab.strategies.cross_sectional import CrossSectionalFactor # long/short factor
from stratlab.strategies.pairs import Pairs                       # 2-asset stat-arb
from stratlab.strategies.news_overlay import NewsOverlay          # trend × sentiment

# Donchian breakout on a single name
DonchianBreakout(entry_window=20, exit_window=10)

# Cross-sectional 12-1 momentum on a basket (default factor); pass `factor_fn=...` to swap
CrossSectionalFactor(k=20, lookback=252, rebalance=21)

# Pairs trade on a hand-picked pair (skips formal cointegration test — pre-screen offline)
Pairs(sym_a="KO", sym_b="PEP", lookback=60, entry_z=2.0, exit_z=0.5)

# News-aware overlay: long only when both price momentum AND sentiment are bullish
from stratlab import daily_sentiment
sent = daily_sentiment(start="2020-01-01", topics=["business"]).mean(axis=1)
NewsOverlay(sentiment=sent, momentum_window=20, sentiment_threshold=0.05)
```

### Cross-sectional (write your own)

```python
from stratlab import Backtest, BarContext, Strategy, Order, load_universe, sp500_tickers
from stratlab.engine.broker import OrderSide

class TopMomentum(Strategy):
    def on_bar(self, ctx: BarContext):
        if ctx.idx < 252 or ctx.idx % 21 != 0:
            return []
        prices = ctx.closes_window(252)
        ret = prices.iloc[-21] / prices.iloc[0] - 1.0
        winners = set(ret.dropna().sort_values().tail(20).index)
        # ... return a list of Orders to rebalance into winners

data = load_universe(sp500_tickers(), start="2020-01-01")
bt = Backtest(data=data, strategy=TopMomentum(),
              initial_cash=1_000_000, borrow_rate_annual=0.005)
print(bt.run().metrics)
```

See `examples/cross_sectional_demo.py` for a complete long/short version.

---

## Tearsheet

`stratlab.tearsheet(result)` renders a 5-panel performance report:
equity curve vs benchmark, underwater drawdown, monthly-returns
heatmap, rolling 6mo Sharpe, and round-trip trade scatter. Headline
metrics (CAGR/Sharpe/MaxDD/Calmar) are baked into the figure title so
saved tearsheets are self-describing.

```python
from stratlab import Backtest, load_bars, tearsheet
from stratlab.strategies.sma_crossover import SMACrossover

data = load_bars("AAPL", start="2018-01-01", end="2024-01-01")
result = Backtest(data={"AAPL": data}, strategy=SMACrossover(fast=10, slow=30)).run()

fig = tearsheet(result, benchmark="SPY", title="SMA 10/30 on AAPL")
fig.write_html("/tmp/aapl.html")    # interactive — open in browser
fig.write_image("/tmp/aapl.png")    # static PNG (requires `pip install kaleido`)
fig.show()                          # in a notebook
```

`benchmark` accepts a ticker string (auto-loaded from cache), a price
`Series`, or `None` to suppress the overlay.

---

## Out-of-sample evaluation

`stratlab.evaluation` exposes two functions for sanity-checking a
strategy beyond the in-sample headline metrics.

```python
from stratlab import Backtest, load_bars, walk_forward, compare_to_benchmark
from stratlab.strategies.sma_crossover import SMACrossover

data = load_bars("AAPL", start="2018-01-01", end="2024-01-01")

# Per-window metrics across rolling 1-year windows.
# Catches strategies that work in one regime but blow up in another.
wf = walk_forward(SMACrossover(fast=10, slow=30), {"AAPL": data}, window_years=1.0)
print(wf)
#         start         end   cagr  sharpe  ...
# 0  2018-01-02  2019-01-02  0.000  -0.006  ...
# 1  2019-01-03  2020-01-02  0.029   2.788  ...
# ...

# Strategy vs buy-and-hold of a benchmark over the same date range.
result = Backtest(data={"AAPL": data}, strategy=SMACrossover(fast=10, slow=30)).run()
cmp = compare_to_benchmark(result, benchmark="SPY")
print(cmp)
#                    strategy  benchmark  alpha
# metric
# cagr                  0.016      0.118 -0.103
# sharpe                0.688      0.652  0.036
# max_drawdown         -0.042     -0.337  0.295
# ...
```

`compare_to_benchmark` accepts a ticker string (auto-loaded from cache)
or a price `Series` — handy for benchmarking against a custom basket.

---

## Inspecting the catalog

```python
import json
catalog = json.load(open("data/market/catalog.json"))

# What sector is JPM in?
catalog["stocks"]["JPM"]
# → {"sector": "financials"}

# What category is TQQQ?
catalog["etfs"]["TQQQ"]
# → {"category": "leveraged"}

# What futures do we know about?
list(catalog["futures"].keys())[:10]
# → ['CL=F', 'BZ=F', 'NG=F', ...]
```

---

## Loading news

```python
import json
articles = json.load(open("data/news/npr/economy/2024.json"))
# → {"2024-01-16-1197961116": {"title": ..., "authors": [...], "content": ..., ...}}

for article_id, art in list(articles.items())[:3]:
    print(f"{art['published_date']}  {art['title'][:80]}")
    print(f"  by: {', '.join(art['authors'])}")
```

---

## Engine knobs

```python
Backtest(
    data=...,
    strategy=...,
    initial_cash=100_000,
    commission_pct=0.001,           # 10 bps per trade
    slippage_pct=0.0005,             # 5 bps per side
    allow_short=True,                # signed-size positions
    borrow_rate_annual=0.005,        # 50 bps/yr on absolute short notional
).run()
```

Returns a `BacktestResult` with `equity_curve`, `returns`, `fills`, `trades`,
and a `metrics` dict containing total_return, cagr, sharpe, sortino, calmar,
max_drawdown, annual_volatility, n_trades, n_round_trips, trade_win_rate,
profit_factor, turnover_annualized, borrow_cost, dropped_orders.

---

## Tests

The engine has a deterministic test suite covering its core invariants —
no live data, no API calls, runs in <1s. If anything below fails, every
backtest result is suspect.

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

What's covered:

- **No-trade preserves cash** — strategy that returns `[]` ⇒ flat equity curve at `initial_cash`
- **Next-bar fill** — order placed on bar `i` fills at bar `i+1`'s open
- **Final-bar orders dropped** — orders submitted on the last bar can't teleport-fill
- **Round-trip PnL** — buy at price A, sell at price B ⇒ `gross_pnl == (B - A) * size`
- **Short-side accounting** — sell-first opens a short with correct PnL sign
- **Commission per fill** — `commission_pct` deducted from notional on every fill
- **Slippage direction** — buys fill above the open, sells below
- **Cross-sectional alignment** — assets with non-overlapping date ranges align on the union, no NaN crashes
- **Buy-and-hold tracks price** — final equity matches initial cash + price drift
- **Returns ≡ equity.pct_change()** — the two series stay consistent

When you change anything in `stratlab/engine/`, run pytest first.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|---------------------|
| `ModuleNotFoundError: No module named 'stratlab'` | `pip install -e .` from project root |
| `failed: ['XYZ', ...]` in refresh summary | Yahoo doesn't have data for those (delisted, typo'd, or recently IPO'd in a way yfinance can't yet pull) — usually safe to ignore |
| News scrape returns 0 articles | NPR archive page may be empty for that date / topic. Try a different day or `--start` further back. |
| `urllib.error.HTTPError: 403` from Wikipedia | The user-agent is being blocked. Re-run; `stratlab` already sends a polite UA. |
| First refresh after a long break re-downloads everything | Cache key changed in a release. The orphan-cleanup will catch the old files; new layout repopulates incrementally. |
| `ValueError: ... 'date'` reading a CSV | Pre-existing user CSV with capitalized headers; `_read_cache` is now tolerant. Re-run refresh and the migration will normalize it. |

---

## Useful one-liners

```bash
# How many bars do I have for AAPL?
python -c "from stratlab import load_bars; print(len(load_bars('AAPL')))"

# How big is the cache?
du -sh data/market data/news

# How many articles total?
find data/news -name "*.json" -exec python -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" {} \; | awk '{s+=$1} END {print s}'

# Count tickers per asset class
python -c "
import json
c = json.load(open('data/market/catalog.json'))
for k in ['stocks','etfs','indices','futures']:
    print(f'{k}: {len(c[k])}')
"
```

---

## Editing the universe

To add tickers to the curated lists:

- **Stocks**: edit nothing — comes from the live S&P 500 / NDX / Dow lists.
- **ETFs**: `stratlab/data/_etf_lists.py` → `ETF_CATEGORIES`
- **Indices**: `stratlab/data/_index_lists.py` → `INDEX_CATEGORIES`
- **Futures**: `stratlab/data/_futures_lists.py` → `FUTURES_CATEGORIES`

After editing, run `python -m stratlab.refresh` and the new tickers get
fetched and routed to the right category folder.
