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
from stratlab import Strategy, Order
from stratlab.engine.broker import OrderSide

class MyStrategy(Strategy):
    def on_bar(self, idx, history):
        if idx < 20:
            return []
        closes = history["close"].iloc[:idx+1]
        if closes.iloc[-1] > closes.iloc[-20:].mean():
            return [Order(side=OrderSide.BUY, size=100)]
        return []
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

- **Total Return** — overall P&L
- **CAGR** — compound annual growth rate
- **Sharpe Ratio** — risk-adjusted return
- **Sortino Ratio** — downside risk-adjusted return
- **Max Drawdown** — worst peak-to-trough decline
- **Annual Volatility** — annualized standard deviation
- **Calmar Ratio** — CAGR / max drawdown
- **Win Rate** — percentage of positive-return bars

## Broker Simulation

The simulated broker models:
- **Slippage** — configurable percentage (default 0.05%)
- **Commission** — configurable percentage (default 0.1%)
- **Position tracking** — average entry price, size
- **Cash management** — rejects orders that exceed available cash

## Data

Market data is fetched via yfinance and cached as CSV files in `~/.stratlab/cache/`. Pass `use_cache=False` to `load_bars()` to force a fresh download.

## For AI Agents

StratLab is designed as an environment for AI agents to:

1. **LLM agents**: Write a `Strategy` subclass, call `Backtest.run()`, read the metrics dict
2. **RL agents**: Train directly via `TradingEnv` using any Gymnasium-compatible library
3. **Hybrid**: Use an LLM to generate strategy code, backtest it, iterate on results

The strategy interface is intentionally simple — `on_bar(idx, history) -> list[Order]` — so agents can generate and test ideas with minimal boilerplate.
