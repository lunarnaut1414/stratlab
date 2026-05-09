"""Cross-sectional momentum across the current S&P 500 — long/short.

Each month: rank all currently-tradeable names by their 12-1 month total return
(11 months ending one month ago, the classic Jegadeesh-Titman lookback that
skips the most recent month to dodge short-term reversal). Equal-weight long
the top-K names, equal-weight short the bottom-K, dollar-neutral.
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


class LongShortMomentum(Strategy):
    def __init__(
        self,
        k: int = 20,
        lookback: int = 252,
        skip: int = 21,
        rebalance: int = 21,
        gross_leverage: float = 1.0,
    ) -> None:
        super().__init__(
            k=k, lookback=lookback, skip=skip, rebalance=rebalance, gross=gross_leverage,
        )
        self.k = k
        self.lookback = lookback
        self.skip = skip
        self.rebalance = rebalance
        self.gross_leverage = gross_leverage

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.idx < self.lookback:
            return []
        if ctx.idx % self.rebalance != 0:
            return []

        prices = ctx.closes_window(self.lookback)
        ret = prices.iloc[-self.skip] / prices.iloc[0] - 1.0
        ret = ret.dropna()
        if len(ret) < 2 * self.k:
            return []

        ranked = ret.sort_values()
        longs = set(ranked.tail(self.k).index)
        shorts = set(ranked.head(self.k).index)

        target: dict[str, int] = {}
        live_closes = ctx.closes()
        equity = ctx.portfolio_value({s: float(p) for s, p in live_closes.items()})
        per_leg_dollars = equity * self.gross_leverage / 2.0
        per_name_dollars = per_leg_dollars / self.k

        for sym in longs:
            if sym in live_closes:
                target[sym] = int(per_name_dollars // float(live_closes[sym]))
        for sym in shorts:
            if sym in live_closes:
                target[sym] = -int(per_name_dollars // float(live_closes[sym]))

        orders: list[Order] = []
        # Close anything not in the new target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Move existing holdings to their new target size
        for sym, tgt in target.items():
            current = ctx.position(sym).size
            delta = tgt - current
            if delta == 0:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


def main() -> None:
    print("Loading S&P 500 tickers...")
    tickers = sp500_tickers()
    print(f"  {len(tickers)} names")

    print("Loading 5y of daily bars (cached after first run)...")
    data = load_universe(tickers, start="2020-01-01")
    print(f"  {len(data)} tickers with data\n")

    strategy = LongShortMomentum(k=20, lookback=252, skip=21, rebalance=21, gross_leverage=1.0)
    bt = Backtest(
        data=data,
        strategy=strategy,
        initial_cash=1_000_000.0,
        borrow_rate_annual=0.005,  # 50 bps/yr stylized borrow cost
    )

    print("Running backtest...")
    result = bt.run()

    print("\n--- Long/Short Top-20 Momentum (12-1mo, monthly rebal) ---")
    for k, v in result.metrics.items():
        print(f"  {k:20s}: {v}")


if __name__ == "__main__":
    main()
