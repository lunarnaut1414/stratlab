"""Cross-sectional momentum across the current S&P 500.

Each month: rank all currently-tradeable names by their 12-1 month total return
(11 months ending one month ago, the classic Jegadeesh-Titman lookback that
skips the most recent month to dodge short-term reversal). Equal-weight the
top-K names, sell anything that fell out of the top-K.

Long-only — the broker doesn't model shorting yet, so this is the simplest
form of the cross-sectional momentum factor.
"""
from __future__ import annotations

from stratlab import (
    Backtest,
    BarContext,
    Order,
    Strategy,
    load_universe,
    sp500_tickers,
)
from stratlab.engine.broker import OrderSide


class TopKMomentum(Strategy):
    def __init__(
        self,
        k: int = 20,
        lookback: int = 252,
        skip: int = 21,
        rebalance: int = 21,
    ) -> None:
        super().__init__(k=k, lookback=lookback, skip=skip, rebalance=rebalance)
        self.k = k
        self.lookback = lookback
        self.skip = skip
        self.rebalance = rebalance

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.idx < self.lookback:
            return []
        if ctx.idx % self.rebalance != 0:
            return []

        prices = ctx.closes_window(self.lookback)
        # 12-1 month return: end one month before today
        ret = prices.iloc[-self.skip] / prices.iloc[0] - 1.0
        ret = ret.dropna()
        if len(ret) < self.k:
            return []

        winners = set(ret.sort_values().tail(self.k).index.tolist())

        orders: list[Order] = []

        for sym, pos in list(ctx.positions.items()):
            if sym not in winners and pos.size > 0:
                orders.append(Order(side=OrderSide.SELL, size=pos.size, symbol=sym))

        live_closes = ctx.closes()
        budget_per_name = ctx.cash / max(len(winners), 1)
        for sym in winners:
            if sym not in live_closes:
                continue
            if ctx.position(sym).size > 0:
                continue
            price = float(live_closes[sym])
            size = budget_per_name // price
            if size > 0:
                orders.append(Order(side=OrderSide.BUY, size=size, symbol=sym))

        return orders


def main() -> None:
    print("Loading S&P 500 tickers...")
    tickers = sp500_tickers()
    print(f"  {len(tickers)} names")

    print("Loading 5y of daily bars (cached after first run)...")
    data = load_universe(tickers, start="2020-01-01")
    print(f"  {len(data)} tickers with data\n")

    strategy = TopKMomentum(k=20, lookback=252, skip=21, rebalance=21)
    bt = Backtest(data=data, strategy=strategy, initial_cash=1_000_000.0)

    print("Running backtest...")
    result = bt.run()

    print("\n--- Top-20 Momentum (12-1mo, monthly rebal) ---")
    for k, v in result.metrics.items():
        print(f"  {k:20s}: {v}")


if __name__ == "__main__":
    main()
