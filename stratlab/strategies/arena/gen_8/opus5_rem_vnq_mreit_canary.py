"""REM mortgage-REIT vs VNQ broad-REIT duration+credit canary — gen_8 opus-5 (wildcard).

Hypothesis: The REM/VNQ spread is a unique, anti-consensus regime signal that
captures duration+credit stress dynamics absent from any leaderboard strategy.

Why mREITs are special:
  - REM holds mortgage REITs (AGNC, NLY, STWD, etc.) — leveraged-yield vehicles
    that hold MBS/CMBS financed with short-term repo. They are uniquely
    sensitive to BOTH (a) duration risk (long MBS) AND (b) short-term credit /
    repo-funding spreads — neither of which is cleanly captured by JNK/LQD
    (pure credit), TLT (pure duration), or VIX (pure equity vol).
  - VNQ holds broad equity REITs (industrial, residential, retail) — these
    are rate-sensitive but not leveraged-yield instruments. Their stress
    profile is dominated by real-estate cycle dynamics, not financing.
  - When REM materially underperforms VNQ (spread < -5% on 42d), it signals
    a stealth duration+credit shock: rate vol up, financing widening,
    book-value impairment — which historically PRECEDES broader equity
    weakness by weeks (mREITs price daily, the broader market lags).
  - When REM leads VNQ by >2% on 42d, the carry environment is benign:
    rates stable, credit tight, financing cheap — a duration+credit "all
    clear" that historically supports growth/QQQ extension.

Why anti-consensus:
  - No leaderboard strategy uses REM or MORT (mortgage REITs).
  - VNQ appears only in `vnq_tlt_yield_regime` as a broad-REIT-vs-bond
    signal — this strategy uses VNQ as the COMPARISON not the primary;
    REM is the lead signal.
  - All explored "duration" signals (TNX, yield curve, TLT/IEF) measure
    bond-market state directly; this measures it indirectly through a
    leveraged carry trade's relative weakness — a fundamentally different
    information source.
  - Prohibited list: not in any prohibited cluster (not SP500-xsect, not
    VIX-gate, not JNK-credit, not yield-curve, not Halloween, not sector
    rotation, not commodity, not gold-vs-equity, not PFF, not credit
    divergence).

Regime logic (popular_etfs universe; weekly rebalance):
  1. Outer bear gate: SPY < 200d SMA → TLT 60% + SHY 37% (4-bar defensive).
  2. Compute spread = REM_42d_return - VNQ_42d_return.
     - Spread > +2% AND SPY bull: REM leading = benign duration+credit
       → hold QQQ 97% (growth extension).
     - Spread between -5% and +2% AND SPY bull: neutral
       → hold SPY 60% + IEF 37% (balanced).
     - Spread < -5% AND SPY bull: stealth duration+credit stress
       → hold TLT 60% + IEF 37% (preemptive defensive, even though
         SPY itself looks OK).

The asymmetric threshold (-5% vs +2%) reflects the asymmetry of mREIT
stress: mREITs underperform broadly during equity tantrums AND duration
shocks, so a -5% spread is a stronger signal than a +2% one. Tuned to
generate ~weekly transitions over 2010-2018 IS window without overfitting.

Universe is popular_etfs (not SP500) — keeps n_trades small and avoids
correlation with the dominant SP500-xsect cluster.

Data availability check:
  - VNQ inception 2004 (full IS coverage).
  - REM inception 2007 (full IS coverage — 2010-2018).
  - SPY, QQQ, TLT, IEF, SHY all full coverage.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5            # weekly
SPREAD_WINDOW = 42             # ~2 months — long enough to be meaningful, short enough to act
TREND_WINDOW = 200             # SPY 200d SMA outer gate
SPREAD_HIGH = 0.02             # REM leads VNQ by >2% → QQQ tier
SPREAD_LOW = -0.05             # REM lags VNQ by >5% → defensive tier (asymmetric)
EXPOSURE = 0.97

_SPY = "SPY"
_QQQ = "QQQ"
_REM = "REM"
_VNQ = "VNQ"
_TLT = "TLT"
_IEF = "IEF"
_SHY = "SHY"

UNIVERSE = "popular_etfs"


class RemVnqMreitCanary(Strategy):
    """REM vs VNQ spread as duration+credit canary routing QQQ / SPY+IEF / TLT+IEF / TLT+SHY."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        spread_window: int = SPREAD_WINDOW,
        trend_window: int = TREND_WINDOW,
        spread_high: float = SPREAD_HIGH,
        spread_low: float = SPREAD_LOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            spread_window=spread_window,
            trend_window=trend_window,
            spread_high=spread_high,
            spread_low=spread_low,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.spread_window = int(spread_window)
        self.trend_window = int(trend_window)
        self.spread_high = float(spread_high)
        self.spread_low = float(spread_low)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.spread_window) + 10
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
        try:
            spy_hist = ctx.history(_SPY)
        except Exception:
            return []
        if spy_hist is None or len(spy_hist) < self.trend_window + 5:
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
            # --- Compute REM vs VNQ 42d return spread ---
            rem_ret = None
            vnq_ret = None
            try:
                rem_hist = ctx.history(_REM)
                if rem_hist is not None and len(rem_hist) >= self.spread_window + 5:
                    rem_cl = rem_hist["close"].dropna()
                    if len(rem_cl) >= self.spread_window + 1:
                        rem_ret = float(
                            rem_cl.iloc[-1] / rem_cl.iloc[-self.spread_window - 1] - 1.0
                        )
            except Exception:
                pass
            try:
                vnq_hist = ctx.history(_VNQ)
                if vnq_hist is not None and len(vnq_hist) >= self.spread_window + 5:
                    vnq_cl = vnq_hist["close"].dropna()
                    if len(vnq_cl) >= self.spread_window + 1:
                        vnq_ret = float(
                            vnq_cl.iloc[-1] / vnq_cl.iloc[-self.spread_window - 1] - 1.0
                        )
            except Exception:
                pass

            if rem_ret is None or vnq_ret is None or not (
                np.isfinite(rem_ret) and np.isfinite(vnq_ret)
            ):
                # Signal unavailable — default to neutral SPY+IEF
                for sym, w in [(_SPY, 0.60), (_IEF, 0.37)]:
                    if sym in live:
                        target[sym] = w * self.exposure
            else:
                spread = rem_ret - vnq_ret

                if spread > self.spread_high:
                    # REM leading by >2%: benign duration+credit → QQQ
                    if _QQQ in live:
                        target[_QQQ] = self.exposure
                    else:
                        for sym, w in [(_SPY, 0.60), (_IEF, 0.37)]:
                            if sym in live:
                                target[sym] = w * self.exposure
                elif spread < self.spread_low:
                    # REM lagging by >5%: stealth duration+credit shock → TLT+IEF
                    for sym, w in [(_TLT, 0.60), (_IEF, 0.37)]:
                        if sym in live:
                            target[sym] = w * self.exposure
                else:
                    # Neutral spread: SPY 60% + IEF 37%
                    for sym, w in [(_SPY, 0.60), (_IEF, 0.37)]:
                        if sym in live:
                            target[sym] = w * self.exposure

        # --- Execute ---
        orders: list[Order] = []

        # Close positions not in target
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


NAME = "opus5_rem_vnq_mreit_canary"
HYPOTHESIS = (
    "REM mortgage-REIT vs VNQ broad-REIT 42d return spread as duration+credit "
    "stress canary: REM leads VNQ by >2% AND SPY bull hold QQQ 97% (benign "
    "duration+credit); REM lags VNQ by >5% hold TLT 60%+IEF 37% (stealth "
    "duration+credit shock); neutral spread hold SPY 60%+IEF 37%; SPY<200d "
    "bear gate hold TLT 60%+SHY 37%; weekly rebalance; popular_etfs universe; "
    "mREIT leveraged-yield vehicle uniquely captures duration+credit dynamics "
    "absent from leaderboard"
)

STRATEGY = RemVnqMreitCanary()
