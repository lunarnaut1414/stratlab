"""SP500 idiosyncratic momentum with per-stock trailing-stop EXIT — opus-1 gen_7

Mutation of gen7_sp500_idiosyncratic_momentum (parent IS Calmar 1.20).

Parent: rebalances every 10 bars and replaces the top-15 wholesale.
Mutation: keep idiosyncratic-alpha ranking for ENTRY but replace the fixed
biweekly turnover with per-stock trailing-stop exits. Each holding tracks its
own peak-since-entry close; when current close drops 8% from that peak, the
position is liquidated. New entries only happen on a slower 21-bar refresh
(monthly), and only if the slot is empty (existing winners keep running).

This is a fundamentally different EXIT mechanism: parent has fixed-horizon
turnover (every name potentially churned every 10 bars regardless of P&L),
this variant lets winners run and cuts losers individually. Should produce
distinct trade-level fingerprint and lower turnover.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

ENTRY_REFRESH = 21       # monthly review for new entries
MOMENTUM_WINDOW = 63
BETA_WINDOW = 126
TREND_WINDOW = 200
TOP_K = 15
EXPOSURE = 0.97
TRAIL_STOP = 0.08        # 8% trailing stop per stock
_SPY = "SPY"
_TLT = "TLT"


class SP500IdioTrailStop(Strategy):
    def __init__(
        self,
        entry_refresh: int = ENTRY_REFRESH,
        momentum_window: int = MOMENTUM_WINDOW,
        beta_window: int = BETA_WINDOW,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
        trail_stop: float = TRAIL_STOP,
    ) -> None:
        super().__init__(
            entry_refresh=entry_refresh,
            momentum_window=momentum_window,
            beta_window=beta_window,
            trend_window=trend_window,
            top_k=top_k,
            exposure=exposure,
            trail_stop=trail_stop,
        )
        self.entry_refresh = int(entry_refresh)
        self.momentum_window = int(momentum_window)
        self.beta_window = int(beta_window)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)
        self.trail_stop = float(trail_stop)
        # Track peak close per holding (since entry).
        self._peak: dict[str, float] = {}

    def on_start(self) -> None:
        self._peak = {}

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.beta_window + 10
        if ctx.idx < warmup:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Update peaks for current positions
        current_holdings = [s for s, p in ctx.positions.items() if p.size > 0]
        for sym in current_holdings:
            price = live.get(sym)
            if price and price > 0:
                prev_peak = self._peak.get(sym, price)
                if price > prev_peak:
                    self._peak[sym] = price
                else:
                    # Initialize peak from avg_entry if not seen yet
                    self._peak.setdefault(sym, max(price, ctx.position(sym).avg_entry))

        # Drop peak entries for symbols we no longer hold
        for sym in list(self._peak.keys()):
            if sym not in current_holdings:
                self._peak.pop(sym, None)

        orders: list[Order] = []

        # SPY trend gate — if bear, exit everything to TLT
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_now = float(spy_close.iloc[-1])
        bull = spy_now > spy_sma

        if not bull:
            target: dict[str, float] = {}
            if _TLT in live:
                target[_TLT] = self.exposure
            self._peak.clear()
            return self._build_orders(ctx, target, live, equity)

        # Trailing-stop exits per holding
        stops_hit: list[str] = []
        for sym in current_holdings:
            price = live.get(sym)
            peak = self._peak.get(sym)
            if price is None or peak is None or peak <= 0:
                continue
            if price <= peak * (1.0 - self.trail_stop):
                stops_hit.append(sym)

        for sym in stops_hit:
            pos = ctx.position(sym)
            if pos.size > 0:
                orders.append(Order(side=OrderSide.SELL, size=int(pos.size), symbol=sym))
            self._peak.pop(sym, None)

        # Compute remaining holdings after stops
        remaining = [s for s in current_holdings if s not in stops_hit]

        # Entry refresh: monthly look at top-K, fill empty slots with new winners
        is_entry_bar = ctx.idx % self.entry_refresh == 0

        if is_entry_bar and len(remaining) < self.top_k:
            need = max(self.beta_window, self.momentum_window) + 5
            prices = ctx.closes_window(need)
            if len(prices) >= self.momentum_window + 5 and _SPY in prices.columns:
                spy_prices = prices[_SPY].dropna()
                if len(spy_prices) >= self.beta_window:
                    spy_log_rets = np.log(spy_prices.values[1:] / spy_prices.values[:-1])
                    spy_mom_ret = float(spy_prices.iloc[-1] / spy_prices.iloc[-self.momentum_window] - 1.0)

                    scores: dict[str, float] = {}
                    skip = {_SPY, _TLT}
                    for sym in prices.columns:
                        if sym in skip or sym in remaining:
                            continue
                        col = prices[sym].dropna()
                        if len(col) < self.beta_window:
                            continue
                        stock_log_rets = np.log(col.values[1:] / col.values[:-1])
                        n = min(len(stock_log_rets), len(spy_log_rets))
                        if n < 30:
                            continue
                        stock_r = stock_log_rets[-n:]
                        spy_r = spy_log_rets[-n:]
                        if np.std(spy_r) < 1e-8:
                            continue
                        beta = float(np.cov(stock_r, spy_r)[0, 1] / np.var(spy_r))
                        if not np.isfinite(beta):
                            continue
                        if len(col) < self.momentum_window + 1:
                            continue
                        raw_ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                        if not np.isfinite(raw_ret):
                            continue
                        idio = raw_ret - beta * spy_mom_ret
                        if np.isfinite(idio):
                            scores[sym] = idio

                    slots_to_fill = self.top_k - len(remaining)
                    ranked_new = sorted(scores, key=scores.__getitem__, reverse=True)[: slots_to_fill * 2]
                    new_entries = [s for s in ranked_new if s in live][:slots_to_fill]
                else:
                    new_entries = []
            else:
                new_entries = []
        else:
            new_entries = []

        # Build target weights: equal-weight among (remaining + new_entries)
        all_holdings = remaining + new_entries
        target = {}
        if not all_holdings:
            # No equity exposure available -> sit in TLT defensively
            if _TLT in live:
                target[_TLT] = self.exposure
        else:
            per_weight = self.exposure / max(len(all_holdings), 1)
            for sym in all_holdings:
                target[sym] = per_weight

        # Sizing pass: only adjust new entries; let existing holdings drift
        # (we rebalance only on entry-refresh bars; otherwise leave shares as-is)
        if is_entry_bar:
            return self._build_orders_with_sells(ctx, target, live, equity, orders)
        else:
            # Off-rebalance: just process the trail-stop sells we already queued
            return orders

    def _build_orders_with_sells(
        self,
        ctx: BarContext,
        target: dict[str, float],
        live: dict[str, float],
        equity: float,
        existing_orders: list[Order],
    ) -> list[Order]:
        """Rebalance pass: existing_orders already contains trail-stop sells.

        Don't double-sell symbols already in the sell queue.
        """
        already_selling = {o.symbol for o in existing_orders if o.side == OrderSide.SELL}
        orders = list(existing_orders)

        # Liquidate positions not in target (other than ones already being sold)
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0 and sym not in already_selling:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(int(pos.size)), symbol=sym))

        # Adjust to target
        for sym, weight in target.items():
            price = live.get(sym)
            if not price or price <= 0:
                continue
            tgt_shares = int(equity * weight / price)
            cur = int(ctx.position(sym).size)
            if sym in already_selling:
                cur = 0  # we're flattening
            delta = tgt_shares - cur
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders

    def _build_orders(
        self,
        ctx: BarContext,
        target: dict[str, float],
        live: dict[str, float],
        equity: float,
    ) -> list[Order]:
        orders: list[Order] = []
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(int(pos.size)), symbol=sym))
        for sym, weight in target.items():
            price = live.get(sym)
            if not price or price <= 0:
                continue
            tgt_shares = int(equity * weight / price)
            cur = int(ctx.position(sym).size)
            delta = tgt_shares - cur
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))
        return orders


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + [_TLT, _SPY]


NAME = "opus1_sp500_idio_trailstop"
HYPOTHESIS = (
    "SP500 idiosyncratic momentum with PER-STOCK trailing-stop exit: rank by 63d "
    "beta-adjusted alpha (top-15), but each holding has independent 8% trailing stop "
    "from peak-since-entry; replaces fixed biweekly rebalance with adaptive name-level "
    "exits; SPY 200d gate; mutation of idiosyncratic_momentum exit rule"
)

UNIVERSE = _universe

STRATEGY = SP500IdioTrailStop()
