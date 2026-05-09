# StratLab

Algorithmic trading backtest gym built for AI agents. Design strategies in Python, test them against historical data, and train RL agents through a standard Gymnasium interface.

## Architecture

```
stratlab/
  data/          Market data fetching & caching (yfinance)
  engine/        Backtest engine + simulated broker
  strategies/    Strategy base class + built-in examples
  gym/           Gymnasium-compatible RL environment
  analytics/     Performance metrics & equity curve plotting
examples/        Quickstart scripts
```

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
        closes = ctx.history()["close"]  # already sliced to [0, idx]
        if closes.iloc[-1] > closes.iloc[-20:].mean():
            return [Order(side=OrderSide.BUY, size=100)]
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

## Gym Environment

The `TradingEnv` follows the standard Gymnasium API:

| | |
|-|-|
| **Observation** | Rolling window of normalized OHLCV + position info |
| **Actions** | `0` = hold, `1` = buy, `2` = sell |
| **Reward** | Change in portfolio value (configurable: `pnl` or `log_return`) |

## Built-in Strategies

| Strategy | Description | Key Params |
|----------|-------------|------------|
| `SMACrossover` | Fast/slow SMA crossover | `fast`, `slow` |
| `MeanReversion` | Bollinger Band mean reversion | `window`, `num_std` |
| `Momentum` | RSI-based momentum | `period`, `oversold`, `overbought` |

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

## Execution timing

Orders submitted by `on_bar` on bar `i` are filled at bar `i+1`'s open with
slippage applied — you can't decide on a close and fill at that close. Orders
submitted on the final bar are dropped and surfaced as `dropped_orders` in the
metrics dict. Held positions are marked-to-market at each bar's close.

## Data

Market data is fetched via yfinance and cached as CSV files in `~/.stratlab/cache/`. Pass `use_cache=False` to `load_bars()` to force a fresh download.

## For AI Agents

StratLab is designed as an environment for AI agents to:

1. **LLM agents**: Write a `Strategy` subclass, call `Backtest.run()`, read the metrics dict
2. **RL agents**: Train directly via `TradingEnv` using any Gymnasium-compatible library
3. **Hybrid**: Use an LLM to generate strategy code, backtest it, iterate on results

The strategy interface is intentionally simple — `on_bar(idx, history) -> list[Order]` — so agents can generate and test ideas with minimal boilerplate.
