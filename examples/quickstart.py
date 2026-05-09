"""StratLab quickstart — run three strategies on AAPL and compare results."""

from stratlab import Backtest, load_bars
from stratlab.analytics.plot import plot_equity
from stratlab.strategies.sma_crossover import SMACrossover
from stratlab.strategies.mean_reversion import MeanReversion
from stratlab.strategies.momentum import Momentum


def main():
    # 1. Load data
    print("Fetching AAPL daily bars...")
    data = load_bars("AAPL", start="2020-01-01", end="2024-01-01")
    print(f"Loaded {len(data)} bars\n")

    # 2. Define strategies
    strategies = {
        "SMA Crossover (10/30)": SMACrossover(fast=10, slow=30),
        "Mean Reversion (BB 20/2)": MeanReversion(window=20, num_std=2.0),
        "RSI Momentum (14)": Momentum(period=14),
    }

    # 3. Run backtests
    for name, strategy in strategies.items():
        bt = Backtest(data={"AAPL": data}, strategy=strategy)
        result = bt.run()

        print(f"--- {name} ---")
        for k, v in result.metrics.items():
            print(f"  {k:20s}: {v}")
        print()

        plot_equity(result, title=name)


if __name__ == "__main__":
    main()
