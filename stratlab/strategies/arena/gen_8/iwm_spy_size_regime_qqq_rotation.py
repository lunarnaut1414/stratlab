"""IWM vs SPY Size-Regime QQQ/SPY/TLT Rotation — gen_8 sonnet-5

Hypothesis: Use IWM vs SPY 63-day relative momentum as a "risk appetite / size
regime" signal to rotate between QQQ (tech/growth), SPY (broad market), and
TLT (defensive bonds).

Signal: IWM_63d_return - SPY_63d_return (small-cap premium)

Regime logic:
  - IWM leads SPY by >2% (small-caps outperforming = broad risk-on, expansion):
    Hold QQQ 97%. Small-cap leadership historically precedes tech/growth extension.
  - IWM leads or ties SPY (spread -2% to +2%): Hold SPY 60% + IEF 37%.
    Moderate risk-on but narrow leadership — stay broad without full tech tilt.
  - SPY leads IWM (spread < -2%): Hold SPY 60% + TLT 37%.
    Large-cap defensive within equity — flight to mega-cap quality.
  - SPY below 200d SMA (outer bear gate): Hold TLT 60% + SHY 37%.

Additional JNK credit check for the QQQ tier:
  - When IWM leads by >2% BUT JNK 20d MA < JNK 60d MA (credit weak), downgrade
    to SPY 60% + IEF 37% instead of QQQ. Prevents false-positive QQQ entry
    when size momentum is driven by risk-off small-cap bounce (e.g., value recovery).

Universe: popular_etfs (not SP500 stocks) — ensures low correlation with
SP500 single-stock momentum strategies dominating leaderboard.

Rationale:
  - No strategy on the leaderboard uses IWM/SPY 63d return spread as primary
    regime signal routed to QQQ (gen_6 used IWM/SPY 20d spread for QQQ+IWM vs
    SPY+TLT — different window AND different destination).
  - The 63d window (vs 20d in gen_6 `smallcap_leadership_rotation`) is less
    noisy and captures multi-month regime shifts in size factor premium.
  - Adding the JNK qualification layer prevents the quality degradation that
    killed prior "pure breadth" strategies (RSP/SPY breadth ideas repeatedly
    failed at IS Calmar <0.5 because breadth alone is too noisy).

Rebalance: weekly (5 bars) for sufficient trade count.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5
SPREAD_WINDOW = 63         # IWM vs SPY 63d return spread
TREND_WINDOW = 200         # SPY 200d SMA
JNK_FAST_MA = 20
JNK_SLOW_MA = 60
SPREAD_HIGH = 0.02         # IWM leads by >2%: QQQ tier
SPREAD_LOW = -0.02         # SPY leads by >2%: SPY+TLT tier
EXPOSURE = 0.97
_SPY = "SPY"
_QQQ = "QQQ"
_IWM = "IWM"
_TLT = "TLT"
_SHY = "SHY"
_IEF = "IEF"
_JNK = "JNK"

UNIVERSE = "popular_etfs"


class IwmSpySizeRegimeQqqRotation(Strategy):
    """IWM/SPY 63d spread as size regime signal: QQQ vs SPY vs TLT rotation."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        spread_window: int = SPREAD_WINDOW,
        trend_window: int = TREND_WINDOW,
        jnk_fast_ma: int = JNK_FAST_MA,
        jnk_slow_ma: int = JNK_SLOW_MA,
        spread_high: float = SPREAD_HIGH,
        spread_low: float = SPREAD_LOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            spread_window=spread_window,
            trend_window=trend_window,
            jnk_fast_ma=jnk_fast_ma,
            jnk_slow_ma=jnk_slow_ma,
            spread_high=spread_high,
            spread_low=spread_low,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.spread_window = int(spread_window)
        self.trend_window = int(trend_window)
        self.jnk_fast_ma = int(jnk_fast_ma)
        self.jnk_slow_ma = int(jnk_slow_ma)
        self.spread_high = float(spread_high)
        self.spread_low = float(spread_low)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.jnk_slow_ma, self.spread_window) + 10
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

        # --- SPY 200d SMA outer bear gate ---
        spy_hist = ctx.history(_SPY)
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_cl = spy_hist["close"].dropna()
        if len(spy_cl) < self.trend_window:
            return []
        spy_bull = float(spy_cl.iloc[-1]) > float(spy_cl.iloc[-self.trend_window:].mean())

        target: dict[str, float] = {}

        if not spy_bull:
            # Bear: TLT 60% + SHY 37%
            for sym, w in [(_TLT, 0.60), (_SHY, 0.37)]:
                if sym in live:
                    target[sym] = w * self.exposure
        else:
            # --- IWM vs SPY 63d return spread ---
            iwm_hist = ctx.history(_IWM)
            size_spread = 0.0  # default: neutral
            if len(iwm_hist) >= self.spread_window + 5:
                iwm_cl = iwm_hist["close"].dropna()
                if (len(iwm_cl) >= self.spread_window + 1 and
                        len(spy_cl) >= self.spread_window + 1):
                    iwm_ret = float(iwm_cl.iloc[-1] / iwm_cl.iloc[-self.spread_window - 1] - 1.0)
                    spy_ret = float(spy_cl.iloc[-1] / spy_cl.iloc[-self.spread_window - 1] - 1.0)
                    size_spread = iwm_ret - spy_ret

            # --- JNK credit qualification ---
            credit_ok = True  # default: assume credit ok
            try:
                jnk_hist = ctx.history(_JNK)
                if len(jnk_hist) >= self.jnk_slow_ma + 5:
                    jnk_cl = jnk_hist["close"].dropna()
                    if len(jnk_cl) >= self.jnk_slow_ma:
                        jnk_fast = float(jnk_cl.iloc[-self.jnk_fast_ma:].mean())
                        jnk_slow = float(jnk_cl.iloc[-self.jnk_slow_ma:].mean())
                        credit_ok = jnk_fast >= jnk_slow
            except Exception:
                pass

            if size_spread > self.spread_high and credit_ok:
                # Small-caps leading by >2% AND credit healthy: QQQ 97%
                if _QQQ in live:
                    target[_QQQ] = self.exposure
            elif size_spread > self.spread_high and not credit_ok:
                # Small-caps leading but credit weak: downgrade to SPY+IEF
                for sym, w in [(_SPY, 0.60), (_IEF, 0.37)]:
                    if sym in live:
                        target[sym] = w * self.exposure
            elif size_spread >= self.spread_low:
                # Neutral spread (-2% to +2%): SPY 60% + IEF 37%
                for sym, w in [(_SPY, 0.60), (_IEF, 0.37)]:
                    if sym in live:
                        target[sym] = w * self.exposure
            else:
                # Large-caps leading by >2% (defensive quality rotation): SPY 60% + TLT 37%
                for sym, w in [(_SPY, 0.60), (_TLT, 0.37)]:
                    if sym in live:
                        target[sym] = w * self.exposure

        # --- Execute ---
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


NAME = "iwm_spy_size_regime_qqq_rotation"
HYPOTHESIS = (
    "IWM vs SPY 63d relative momentum as size-regime gate for QQQ vs SPY rotation: "
    "IWM leads SPY by >2% AND credit ok (JNK 20d>60d MA) hold QQQ 97%; "
    "IWM leads but credit weak hold SPY 60%+IEF 37%; "
    "neutral spread hold SPY 60%+IEF 37%; "
    "SPY leads IWM by >2% hold SPY 60%+TLT 37%; "
    "SPY 200d bear gate hold TLT 60%+SHY 37%; weekly rebalance; popular_etfs universe"
)

STRATEGY = IwmSpySizeRegimeQqqRotation()
