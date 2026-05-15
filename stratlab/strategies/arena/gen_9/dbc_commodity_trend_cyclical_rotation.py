"""DBC Commodity-Trend Gated QQQ/IWM/TLT Switcher — gen_9 sonnet-7

Hypothesis: DBC (broad commodity ETF) 20d vs 60d SMA crossover as inflation/
growth cycle signal. Commodity uptrend → IWM 97% (small-caps are cyclical,
benefit from broad domestic demand and commodity-price inflation). Commodity
downtrend + SPY bull → QQQ 97% (tech/growth tends to outperform when commodity
prices are falling, input-cost tailwind). SPY bear → TLT 97%. Weekly rebalance.

Rationale: DBC trend creates two distinct equity regimes:
- Uptrend (inflation/growth): IWM (small-cap, high commodity exposure via
  industrials, materials, energy producers in the Russell 2000) outperforms
  mega-cap technology during commodity booms.
- Downtrend (deflation/disinflation): QQQ (technology, consumer discretionary
  mega-caps) benefits from falling input costs and tends to lead in
  disinflationary growth environments.

This creates a counter-intuitive rotation: QQQ when DBC falls (deflation =
tech outperforms), IWM when DBC rises (inflation = cyclicals outperform).

Distinct from: No prior strategy uses DBC trend to switch between IWM and QQQ.
All existing IWM/QQQ switchers use VIX, breadth, relative momentum, or credit
signals — never commodity trend.
"""
from __future__ import annotations

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# ── Parameters ──────────────────────────────────────────────────────────────
DBC_FAST = 20        # DBC short-term SMA
DBC_SLOW = 60        # DBC long-term SMA
SPY_TREND = 200      # SPY outer bear gate
REBALANCE_DAYS = 5   # weekly rebalance
EXPOSURE = 0.97


class DbcCommodityIwmQqqSwitch(Strategy):
    """DBC commodity trend: IWM in uptrend, QQQ in downtrend, TLT in bear."""

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(DBC_SLOW, SPY_TREND) + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % REBALANCE_DAYS != 0:
            return []

        closes = ctx.closes()
        if closes.empty:
            return []
        live = {s: float(closes[s]) for s in closes.index if closes[s] > 0}

        # SPY outer bear gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < SPY_TREND + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        spy_sma200 = float(spy_close.iloc[-SPY_TREND:].mean())
        spy_price = float(spy_close.iloc[-1])
        spy_bull = spy_price > spy_sma200

        if not spy_bull:
            target_sym = "TLT"
        else:
            # DBC commodity trend signal
            try:
                dbc_hist = ctx.history("DBC")
            except KeyError:
                target_sym = "QQQ"
            else:
                if len(dbc_hist) < DBC_SLOW + 5:
                    return []
                dbc_close = dbc_hist["close"].dropna()
                if len(dbc_close) < DBC_SLOW:
                    return []
                dbc_fast_sma = float(dbc_close.iloc[-DBC_FAST:].mean())
                dbc_slow_sma = float(dbc_close.iloc[-DBC_SLOW:].mean())
                if dbc_fast_sma > dbc_slow_sma:
                    target_sym = "IWM"   # commodity uptrend -> small-cap cyclicals
                else:
                    target_sym = "QQQ"   # commodity downtrend -> tech/growth

        targets = {target_sym: EXPOSURE}

        # Compute portfolio equity
        equity = ctx.cash
        for sym, pos in ctx.positions.items():
            price = live.get(sym, 0.0)
            if price > 0:
                equity += pos.size * price
        if equity <= 0:
            return []

        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in targets and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Adjust to target weight
        price = live.get(target_sym)
        if price and price > 0:
            tgt_shares = int(equity * EXPOSURE / price)
            cur = int(ctx.position(target_sym).size)
            delta = tgt_shares - cur
            if abs(delta) >= 1:
                side = OrderSide.BUY if delta > 0 else OrderSide.SELL
                orders.append(Order(side=side, size=abs(delta), symbol=target_sym))

        return orders


def _universe() -> list[str]:
    return ["DBC", "SPY", "QQQ", "IWM", "TLT"]


NAME = "gen9_dbc_commodity_iwm_qqq_switch"
HYPOTHESIS = (
    "DBC 20d vs 60d SMA crossover as inflation/deflation regime gate: "
    "commodity uptrend -> IWM 97% (small-cap cyclicals benefit from inflation/demand); "
    "commodity downtrend + SPY bull -> QQQ 97% (tech outperforms in disinflationary growth); "
    "SPY bear -> TLT 97%. Weekly rebalance."
)

UNIVERSE = _universe

STRATEGY = DbcCommodityIwmQqqSwitch()
