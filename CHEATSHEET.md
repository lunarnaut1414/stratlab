# StratLab Cheatsheet

The minimum you need to know to run, query, and backtest.

---

## Daily refresh — the one-command answer

```bash
python -m stratlab.refresh_all
```

Pulls market data (~850 tickers) and last-7-days of NPR articles. Idempotent:
re-running the same day fetches nothing new and finishes in seconds.

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

Sometimes you want one without the other:

```bash
python -m stratlab.refresh                    # market only
python -m stratlab.news.npr                   # news only
```

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
```

News storage is one JSON per `(source, topic, day)`:
`data/news/<source>/<topic>/<YYYY>/<YYYY-MM-DD>.json`. Resume is by file
existence — if the day file exists (even empty), the scraper skips that day
without an HTTP request. Legacy year-based files have been backed up to
`data/news/_legacy_yearly_backup/`.

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
