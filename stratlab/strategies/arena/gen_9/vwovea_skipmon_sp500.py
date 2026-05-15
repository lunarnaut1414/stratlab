"""gen_9 sonnet-1 — VWO/VEA Global Risk Signal + Skip-Month SP500 Momentum

Hypothesis: Use EM vs DM relative performance (VWO 42d return > VEA 42d return)
as a global-risk-appetite signal. When EM leads DM (risk-on globally), hold
top-15 SP500 stocks by 126d skip-21d (skip-month) momentum that are above their
own 63d SMA, inverse-vol weighted. When DM leads EM (dollar strength / risk-off),
route to SPY 60% + TLT 37%. SPY 200d SMA outer bear gate → TLT 97%.

Key differentiations from gen_7's opus2_vea_vwo_signal_us_only (IS 0.98, OOS 0.47):
- Skip-month momentum (126d-21d) instead of 63d raw — avoids 1-month reversal
  contamination (Jegadeesh-Titman insight), shown to be more OOS-robust in gen_8
- Per-stock 63d SMA filter (intra-stock trend gate) — gen7's version had none
- 42d ratio window instead of 60d (faster, more responsive global signal)
- DM-leads → SPY+TLT blend (moderate risk-off) instead of TLT-only (very defensive)
- Inverse-vol weighted instead of equal-weight

Both VWO and VEA are signal-only (never ordered). All exposure via SP500 stocks/SPY/TLT.

Coverage check (IS 2010-2018):
  VWO (2005), VEA (2007), SPY (1993), TLT (2002), SP500 stocks (full IS coverage)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

RATIO_WINDOW = 42       # VWO vs VEA 42d return spread
MOM_LONG = 126          # skip-month momentum long window
MOM_SKIP = 21           # skip-month momentum skip window
STOCK_SMA = 63          # per-stock 63d SMA trend filter
SPY_TREND = 200         # 200d SMA outer bear gate
VOL_WINDOW = 21         # 21d realized vol for inverse-vol weighting
TOP_K = 15
EXPOSURE = 0.97
REBALANCE_EVERY = 10    # biweekly

_VWO = "VWO"
_VEA = "VEA"
_SPY = "SPY"
_TLT = "TLT"


class VwoVeaSkipMonSp500(Strategy):
    """VWO/VEA EM-vs-DM signal gating skip-month SP500 momentum."""

    def __init__(
        self,
        ratio_window: int = RATIO_WINDOW,
        mom_long: int = MOM_LONG,
        mom_skip: int = MOM_SKIP,
        stock_sma: int = STOCK_SMA,
        spy_trend: int = SPY_TREND,
        vol_window: int = VOL_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
        rebalance_every: int = REBALANCE_EVERY,
    ) -> None:
        super().__init__(
            ratio_window=ratio_window,
            mom_long=mom_long,
            mom_skip=mom_skip,
            stock_sma=stock_sma,
            spy_trend=spy_trend,
            vol_window=vol_window,
            top_k=top_k,
            exposure=exposure,
            rebalance_every=rebalance_every,
        )
        self.ratio_window = int(ratio_window)
        self.mom_long = int(mom_long)
        self.mom_skip = int(mom_skip)
        self.stock_sma = int(stock_sma)
        self.spy_trend = int(spy_trend)
        self.vol_window = int(vol_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)
        self.rebalance_every = int(rebalance_every)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.mom_long + 5, self.spy_trend + 5) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- SPY 200d SMA outer bear gate ---
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        if len(spy_hist) < self.spy_trend + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend:].mean())
        spy_now = float(spy_close.iloc[-1])
        spy_bull = spy_now > spy_sma

        # --- VWO vs VEA global risk signal ---
        em_leads: bool | None = None
        try:
            vwo_hist = ctx.history(_VWO)
            vea_hist = ctx.history(_VEA)
            if (vwo_hist is not None and vea_hist is not None
                    and len(vwo_hist) >= self.ratio_window + 2
                    and len(vea_hist) >= self.ratio_window + 2):
                vwo_c = vwo_hist["close"].dropna()
                vea_c = vea_hist["close"].dropna()
                if len(vwo_c) >= self.ratio_window and len(vea_c) >= self.ratio_window:
                    vwo_ret = float(vwo_c.iloc[-1] / vwo_c.iloc[-self.ratio_window] - 1.0)
                    vea_ret = float(vea_c.iloc[-1] / vea_c.iloc[-self.ratio_window] - 1.0)
                    if np.isfinite(vwo_ret) and np.isfinite(vea_ret):
                        em_leads = vwo_ret > vea_ret
        except Exception:
            pass

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            # Bear → TLT
            if _TLT in live:
                target[_TLT] = self.exposure
        elif em_leads is False:
            # DM leads EM → moderate risk-off blend
            if _SPY in live:
                target[_SPY] = self.exposure * 0.618
            if _TLT in live:
                target[_TLT] = self.exposure * 0.382
        else:
            # EM leads DM (or signal unavailable) → skip-month SP500 momentum
            need = self.mom_long + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.mom_long:
                if _SPY in live:
                    target[_SPY] = self.exposure
            else:
                scores: dict[str, float] = {}
                vols: dict[str, float] = {}

                for sym in prices.columns:
                    if sym in (_SPY, _TLT, _VWO, _VEA):
                        continue
                    col = prices[sym].dropna()
                    if len(col) < self.mom_long:
                        continue

                    # Skip-month momentum: 126d total return skip most recent 21d
                    p_now = float(col.iloc[-self.mom_skip])    # price 21d ago
                    p_start = float(col.iloc[-self.mom_long])  # price 126d ago
                    if p_start <= 0:
                        continue
                    skip_ret = float(p_now / p_start - 1.0)
                    if not np.isfinite(skip_ret):
                        continue

                    # Per-stock 63d SMA filter
                    if len(col) < self.stock_sma:
                        continue
                    sma_val = float(col.iloc[-self.stock_sma:].mean())
                    current_price = float(col.iloc[-1])
                    if current_price <= sma_val:
                        continue  # below 63d SMA — skip

                    scores[sym] = skip_ret

                    # Realized vol
                    log_rets = np.log(col.values[1:] / col.values[:-1])
                    rv_slice = log_rets[-min(self.vol_window, len(log_rets)):]
                    rv = float(np.std(rv_slice)) * np.sqrt(252)
                    vols[sym] = rv if rv > 1e-6 else 1e-6

                if len(scores) < 5:
                    # Not enough qualifying stocks
                    if _SPY in live:
                        target[_SPY] = self.exposure
                else:
                    k = min(self.top_k, len(scores))
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                    inv_vols = {sym: 1.0 / vols.get(sym, 1.0) for sym in ranked}
                    total_iv = sum(inv_vols.values())
                    for sym in ranked:
                        if sym in live:
                            target[sym] = self.exposure * inv_vols[sym] / total_iv

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
    return sp500_tickers() + [_TLT, _SPY, _VWO, _VEA]


NAME = "vwovea_skipmon_sp500"
HYPOTHESIS = (
    "VWO/VEA EM-vs-DM 42d return as global risk signal gating skip-month SP500 momentum: "
    "EM leads DM → top-15 SP500 by 126d-skip-21d momentum above 63d SMA, inverse-vol weighted; "
    "DM leads EM → SPY 60%+TLT 37%; SPY 200d bear → TLT; biweekly rebalance; "
    "combines gen7's novel EM signal with gen8's OOS-robust skip-month momentum"
)

UNIVERSE = _universe

STRATEGY = VwoVeaSkipMonSp500()
