"""SP500 Skip-Month Momentum with RSP/SPY Continuous Breadth Scaling — gen_9 sonnet-9

Hypothesis: Rank SP500 stocks by 126d-skip-21d momentum (Jegadeesh-Titman
skip-month, the most OOS-robust SP500 signal from gen_8). Scale total
portfolio exposure continuously between 0.60-0.97 based on the RSP/SPY 20d
return spread (equal-weight vs cap-weight relative performance):
  - RSP outperforming SPY significantly (broad participation) → full 0.97
  - RSP at parity with SPY → 0.80
  - SPY outperforming RSP (narrow mega-cap leadership) → 0.60
Stock-level 100d SMA trend gate. TLT defensive when SPY below 200d SMA.
Inverse-vol weighted. Biweekly rebalance.

Rationale:
- Skip-month momentum (126d-skip-21d) was the most OOS-robust SP500 signal
  in gen_8 (OOS Calmar 0.63, 79% IS retention).
- RSP/SPY breadth as CONTINUOUS exposure scalar (not binary) is novel — prior
  rounds used RSP/SPY as a binary regime gate. The continuous form is smoother,
  reduces whipsaw, and scales risk proportionally rather than abruptly.
- When broad market participates (RSP leads), momentum stocks are more robust —
  all sectors are performing. When only mega-caps lead (SPY > RSP), momentum
  is concentrated and more fragile.
- 100d SMA stock-level gate: intermediate between 63d (gen_8 skipmon uses 63d)
  and 200d (many others use 200d). Distinguishes from existing strategies.

Differentiators:
- Skip-month 126d-21d + continuous breadth exposure scaling (not tried before)
- 100d per-stock SMA gate (63d too fast, 200d too slow)
- RSP/SPY spread as continuous function, not binary
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
MOM_LONG = 126          # skip-month momentum lookback
MOM_SKIP = 21           # skip most recent month
STOCK_SMA = 100         # per-stock intermediate trend gate
TREND_WINDOW = 200      # SPY bear gate
VOL_WINDOW = 21         # inverse-vol sizing
BREADTH_WINDOW = 20     # RSP/SPY spread lookback
TOP_K = 15
EXPOSURE_HIGH = 0.97    # when RSP strongly outperforms SPY
EXPOSURE_MID = 0.80     # when RSP ≈ SPY
EXPOSURE_LOW = 0.60     # when SPY strongly outperforms RSP
# Breadth spread thresholds (RSP - SPY 20d return)
BREADTH_HIGH = 0.015    # RSP outperforms SPY by >1.5pp → full tilt
BREADTH_LOW = -0.010    # SPY outperforms RSP by >1.0pp → min tilt

_SPY = "SPY"
_TLT = "TLT"
_IEF = "IEF"
_RSP = "RSP"


class SkipmonRspBreadthScaled(Strategy):
    """Skip-month SP500 momentum with continuous RSP/SPY breadth exposure scaling."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        mom_long: int = MOM_LONG,
        mom_skip: int = MOM_SKIP,
        stock_sma: int = STOCK_SMA,
        trend_window: int = TREND_WINDOW,
        vol_window: int = VOL_WINDOW,
        breadth_window: int = BREADTH_WINDOW,
        top_k: int = TOP_K,
        exposure_high: float = EXPOSURE_HIGH,
        exposure_mid: float = EXPOSURE_MID,
        exposure_low: float = EXPOSURE_LOW,
        breadth_high: float = BREADTH_HIGH,
        breadth_low: float = BREADTH_LOW,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            mom_long=mom_long,
            mom_skip=mom_skip,
            stock_sma=stock_sma,
            trend_window=trend_window,
            vol_window=vol_window,
            breadth_window=breadth_window,
            top_k=top_k,
            exposure_high=exposure_high,
            exposure_mid=exposure_mid,
            exposure_low=exposure_low,
            breadth_high=breadth_high,
            breadth_low=breadth_low,
        )
        self.rebalance_every = int(rebalance_every)
        self.mom_long = int(mom_long)
        self.mom_skip = int(mom_skip)
        self.stock_sma = int(stock_sma)
        self.trend_window = int(trend_window)
        self.vol_window = int(vol_window)
        self.breadth_window = int(breadth_window)
        self.top_k = int(top_k)
        self.exposure_high = float(exposure_high)
        self.exposure_mid = float(exposure_mid)
        self.exposure_low = float(exposure_low)
        self.breadth_high = float(breadth_high)
        self.breadth_low = float(breadth_low)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.mom_long) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # --- SPY 200d bear gate → TLT ---
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
        spy_bull = spy_now > spy_sma

        target: dict[str, float] = {}

        if not spy_bull:
            if _TLT in live:
                target[_TLT] = self.exposure_high
        else:
            # --- RSP/SPY continuous breadth scaling ---
            exposure = self.exposure_mid  # default to mid if breadth unavailable

            try:
                rsp_hist = ctx.history(_RSP)
                if (len(rsp_hist) >= self.breadth_window + 2 and
                        len(spy_close) >= self.breadth_window + 2):
                    rsp_close = rsp_hist["close"].dropna()
                    if (len(rsp_close) >= self.breadth_window + 1 and
                            len(spy_close) >= self.breadth_window + 1):
                        rsp_ret = float(
                            rsp_close.iloc[-1] / rsp_close.iloc[-self.breadth_window] - 1.0
                        )
                        spy_ret_bw = float(
                            spy_close.iloc[-1] / spy_close.iloc[-self.breadth_window] - 1.0
                        )
                        breadth_spread = rsp_ret - spy_ret_bw

                        # Continuous linear interpolation
                        if breadth_spread >= self.breadth_high:
                            exposure = self.exposure_high
                        elif breadth_spread <= self.breadth_low:
                            exposure = self.exposure_low
                        else:
                            # Map breadth_spread from [breadth_low, breadth_high]
                            # to [exposure_low, exposure_high]
                            frac = (breadth_spread - self.breadth_low) / (
                                self.breadth_high - self.breadth_low
                            )
                            exposure = self.exposure_low + frac * (
                                self.exposure_high - self.exposure_low
                            )
            except Exception:
                pass

            # --- Skip-month momentum scoring ---
            need = self.mom_long + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.mom_long + 2:
                if _IEF in live:
                    target[_IEF] = exposure
            else:
                scores: dict[str, float] = {}
                vols: dict[str, float] = {}

                for sym in prices.columns:
                    if sym in (_SPY, _TLT, _IEF, _RSP):
                        continue
                    col = prices[sym].dropna()
                    if len(col) < self.mom_long + 1:
                        continue

                    # Skip-month: return from T-126 to T-21
                    price_at_long = float(col.iloc[-self.mom_long])
                    price_at_skip = float(col.iloc[-self.mom_skip])
                    if price_at_long <= 0 or price_at_skip <= 0:
                        continue
                    skipmon_ret = float(price_at_skip / price_at_long - 1.0)
                    if not np.isfinite(skipmon_ret):
                        continue

                    # Per-stock 100d SMA trend gate
                    if len(col) < self.stock_sma + 1:
                        continue
                    stock_sma_val = float(col.iloc[-self.stock_sma:].mean())
                    current_price = float(col.iloc[-1])
                    if current_price <= stock_sma_val:
                        continue  # below 100d SMA — skip

                    scores[sym] = skipmon_ret

                    daily_rets = col.pct_change().dropna()
                    if len(daily_rets) >= self.vol_window:
                        rv = float(daily_rets.iloc[-self.vol_window:].std())
                        vols[sym] = max(rv, 1e-6)

                if len(scores) < 5:
                    if _IEF in live:
                        target[_IEF] = exposure
                else:
                    k = min(self.top_k, len(scores))
                    ranked = sorted(
                        scores, key=scores.__getitem__, reverse=True
                    )[:k]

                    inv_vols = {sym: 1.0 / vols.get(sym, 0.02) for sym in ranked}
                    total_inv = sum(inv_vols.values())
                    if total_inv <= 0:
                        per_w = exposure / len(ranked)
                        for sym in ranked:
                            if sym in live:
                                target[sym] = per_w
                    else:
                        for sym in ranked:
                            if sym in live:
                                target[sym] = exposure * (
                                    inv_vols[sym] / total_inv
                                )

        orders: list[Order] = []
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))
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
    return sp500_tickers() + [_TLT, _IEF, _SPY, _RSP]


NAME = "skipmon_rsp_breadth_scaled"
HYPOTHESIS = (
    "SP500 skip-month momentum (126d-skip-21d) with continuous RSP/SPY 20d return spread "
    "as breadth-based exposure scalar (0.60 when SPY leads RSP by >1pp, 0.97 when RSP "
    "leads SPY by >1.5pp, linearly interpolated); per-stock 100d SMA trend gate; inverse-vol "
    "weighted; SPY 200d bear gate to TLT; IEF fallback when insufficient candidates; "
    "biweekly rebalance."
)

UNIVERSE = _universe

STRATEGY = SkipmonRspBreadthScaled()
