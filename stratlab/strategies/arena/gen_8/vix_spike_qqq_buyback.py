"""VIX spike QQQ buy-the-dip in bull market (gen_8, sonnet-1).

Hypothesis:
    When VIX spikes >15% in a single day (intraday fear surge) while SPY
    is above its 200d SMA (bull market), buy QQQ at 97% the next bar
    and hold for up to 20 bars (aggressive mean-reversion on fear spikes).
    Return to SPY 97% base when the holding period expires.
    During bear markets (SPY below 200d SMA), hold TLT 60% + IEF 37%.

    The key distinction from gen5_vix_spike_buy_dip_spy:
    - That strategy: buy SPY when SPY drops >2% AND VIX closes above 25
    - This strategy: buy QQQ on VIX intraday spike >15% regardless of SPY price drop
    - Base position is SPY (not SHY/cash), making it more aggressive
    - QQQ (not SPY) on the risk-on recovery leg

    Weekly rebalance check (every 5 bars). Target IS Calmar >0.6 from
    high trade count (frequent VIX spikes in 2010-2018).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# --- Parameters ---
SPY_TREND = 200             # SPY 200d SMA bear gate
VIX_SPIKE_PCT = 0.15        # VIX single-day spike threshold (15%)
HOLD_BARS = 20              # bars to hold QQQ after VIX spike
REBALANCE_EVERY = 5         # weekly
EXPOSURE = 0.97

NAME = "vix_spike_qqq_buyback"
HYPOTHESIS = (
    "VIX single-day spike >15% in SPY bull market triggers QQQ 97% entry for "
    "20-bar hold; base position SPY 97% in bull; TLT 60%+IEF 37% in bear; "
    "weekly rebalance; VIX-spike QQQ (not SPY) recovery distinct from "
    "gen5_vix_spike_buy_dip_spy which uses SPY drop threshold"
)


class VIXSpikeQQQBuyback(Strategy):
    """Buy QQQ on VIX intraday spike in bull market; base SPY; bear TLT."""

    def __init__(
        self,
        spy_trend: int = SPY_TREND,
        vix_spike_pct: float = VIX_SPIKE_PCT,
        hold_bars: int = HOLD_BARS,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            spy_trend=spy_trend,
            vix_spike_pct=vix_spike_pct,
            hold_bars=hold_bars,
            rebalance_every=rebalance_every,
            exposure=exposure,
        )
        self.spy_trend = int(spy_trend)
        self.vix_spike_pct = float(vix_spike_pct)
        self.hold_bars = int(hold_bars)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)
        self._qqq_hold_remaining: int = 0

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.spy_trend + 5
        if ctx.idx < warmup:
            return []

        # --- SPY trend gate ---
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        # --- Check for VIX spike ---
        vix_spike = False
        try:
            vix_hist = ctx.history("^VIX")
            vix_close = vix_hist["close"].dropna()
            if len(vix_close) >= 2:
                vix_prev = float(vix_close.iloc[-2])
                vix_curr = float(vix_close.iloc[-1])
                if vix_prev > 0 and (vix_curr - vix_prev) / vix_prev >= self.vix_spike_pct:
                    vix_spike = True
        except KeyError:
            pass

        # Update hold counter
        if vix_spike and spy_bull:
            self._qqq_hold_remaining = self.hold_bars
        elif self._qqq_hold_remaining > 0:
            self._qqq_hold_remaining -= 1

        # Only rebalance on schedule (or when state changes)
        if ctx.idx % self.rebalance_every != 0 and not vix_spike:
            return []

        # --- Regime routing ---
        if not spy_bull:
            # Bear market -> defensive
            target = {"TLT": 0.60, "IEF": 0.37}
        elif self._qqq_hold_remaining > 0:
            # VIX spike recovery -> QQQ
            target = {"QQQ": self.exposure}
        else:
            # Base position -> SPY
            target = {"SPY": self.exposure}

        # --- Execute ---
        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity_val = ctx.portfolio_value(live)
        if equity_val <= 0:
            return []

        orders: list[Order] = []

        # Sell positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Buy/adjust target positions
        for sym, weight in target.items():
            price = live.get(sym)
            if not price or price <= 0:
                continue
            tgt_shares = int(equity_val * weight / price)
            cur = int(ctx.position(sym).size)
            delta = tgt_shares - cur
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


STRATEGY = VIXSpikeQQQBuyback()
UNIVERSE = ["SPY", "QQQ", "TLT", "IEF", "^VIX"]
