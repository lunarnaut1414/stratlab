"""Sector Multi-Timeframe Momentum with JNK Credit Gate — gen_8 sonnet-2

Hypothesis: Rank 11 SPDR sector ETFs by composite momentum score across three
timeframes (21d + 63d + 126d equal-weighted), hold top-3 sectors equal-weight
when JNK is above its 30d SMA (credit risk-on) AND SPY is above its 200d SMA;
rotate to TLT when credit weakens; hold SHY when SPY is in bear.

Rationale:
- Multi-timeframe composite momentum is more stable than single-window ranking
  (avoids chasing short-term noise while respecting medium/long trend)
- JNK credit gate filters out regime where sector momentum often reverses
- Different from existing leaderboard: pure sector ETF rotation with credit gate
  (not VIX-gated, not individual stock selection, not single timeframe)
- Distinct from gen6_sector strategies which used single 42d window or no credit gate

Signal: composite = mean(21d_ret, 63d_ret, 126d_ret) for each sector ETF
Gate: JNK > 30d SMA (credit) AND SPY > 200d SMA (trend)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

SECTOR_ETFS = [
    "XLK", "XLF", "XLV", "XLI", "XLP", "XLU", "XLE", "XLB", "XLRE", "XLY", "XLC",
]
REBALANCE_EVERY = 10      # biweekly
SHORT_WIN = 21            # 1 month
MED_WIN = 63              # 3 months
LONG_WIN = 126            # 6 months
TOP_K = 3
EXPOSURE = 0.97
SPY_TREND_WIN = 200
JNK_MA_WIN = 30
_SPY = "SPY"
_TLT = "TLT"
_SHY = "SHY"
_JNK = "JNK"


class SectorMultiTFCreditGate(Strategy):
    """Sector ETF composite multi-TF momentum with JNK credit gate."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        short_win: int = SHORT_WIN,
        med_win: int = MED_WIN,
        long_win: int = LONG_WIN,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
        spy_trend_win: int = SPY_TREND_WIN,
        jnk_ma_win: int = JNK_MA_WIN,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            short_win=short_win,
            med_win=med_win,
            long_win=long_win,
            top_k=top_k,
            exposure=exposure,
            spy_trend_win=spy_trend_win,
            jnk_ma_win=jnk_ma_win,
        )
        self.rebalance_every = int(rebalance_every)
        self.short_win = int(short_win)
        self.med_win = int(med_win)
        self.long_win = int(long_win)
        self.top_k = int(top_k)
        self.exposure = float(exposure)
        self.spy_trend_win = int(spy_trend_win)
        self.jnk_ma_win = int(jnk_ma_win)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.long_win + self.spy_trend_win + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY trend gate
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_win:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend_win:].mean())
        spy_now = float(spy_close.iloc[-1])
        bull = spy_now > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not bull:
            # Bear market: park in SHY (no duration risk)
            if _SHY in live:
                target[_SHY] = self.exposure
        else:
            # JNK credit gate
            try:
                jnk_hist = ctx.history(_JNK)
            except KeyError:
                jnk_hist = None
            credit_ok = False
            if jnk_hist is not None:
                jnk_close = jnk_hist["close"].dropna()
                if len(jnk_close) >= self.jnk_ma_win:
                    jnk_ma = float(jnk_close.iloc[-self.jnk_ma_win:].mean())
                    jnk_now = float(jnk_close.iloc[-1])
                    credit_ok = jnk_now > jnk_ma

            if not credit_ok:
                # Credit stressed: rotate to TLT
                if _TLT in live:
                    target[_TLT] = self.exposure
            else:
                # Compute composite multi-TF momentum for each sector ETF
                need = self.long_win + 5
                prices = ctx.closes_window(need)
                if len(prices) < self.long_win:
                    return []

                scores: dict[str, float] = {}
                for etf in SECTOR_ETFS:
                    if etf not in prices.columns:
                        continue
                    col = prices[etf].dropna()
                    if len(col) < self.long_win + 1:
                        continue
                    p_end = float(col.iloc[-1])

                    # Short window return
                    if len(col) >= self.short_win + 1:
                        r_short = p_end / float(col.iloc[-self.short_win]) - 1.0
                    else:
                        r_short = 0.0

                    # Medium window return
                    if len(col) >= self.med_win + 1:
                        r_med = p_end / float(col.iloc[-self.med_win]) - 1.0
                    else:
                        r_med = 0.0

                    # Long window return
                    r_long = p_end / float(col.iloc[-self.long_win]) - 1.0

                    composite = (r_short + r_med + r_long) / 3.0
                    if np.isfinite(composite):
                        scores[etf] = composite

                if len(scores) < self.top_k:
                    # Not enough sector data: fall back to TLT
                    if _TLT in live:
                        target[_TLT] = self.exposure
                else:
                    k = min(self.top_k, len(scores))
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                    per_wt = self.exposure / len(ranked)
                    for etf in ranked:
                        if etf in live:
                            target[etf] = per_wt

        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Adjust to target
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


UNIVERSE = SECTOR_ETFS + [_SPY, _TLT, _SHY, _JNK]

NAME = "sector_multitf_credit_gate"
HYPOTHESIS = (
    "Sector multi-TF momentum + JNK credit gate: rank 11 SPDR sector ETFs by "
    "composite 1m+3m+6m momentum score; hold top-3 equal-weight when JNK > 30d SMA "
    "AND SPY > 200d SMA; TLT when credit weak; SHY in bear; rebalance every 10 bars"
)

STRATEGY = SectorMultiTFCreditGate()
