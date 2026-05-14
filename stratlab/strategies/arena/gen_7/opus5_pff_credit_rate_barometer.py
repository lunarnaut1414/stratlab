"""PFF preferred-stock composite credit-rate barometer — gen_7 opus-5 wildcard.

Hypothesis
----------
PFF (iShares Preferred & Income Securities ETF) is a hybrid security: preferred
stocks are *long-duration* (sensitive to rate moves like long Treasuries) AND
*credit-sensitive* (mostly bank/financial issuers, sensitive to spreads). A single
PFF return therefore composites two normally-orthogonal signals — duration and
credit — into one "financial-conditions" thermometer.

Decision rule (weekly, allow_short=False, enforce_cash=True):
  IF SPY < 200d SMA               -> defensive: IEF 50% + SHY 47%
  ELIF PFF 21d return > +1.0% AND PFF close > PFF 50d SMA   -> QQQ 97%
       (financial conditions easing on BOTH duration and credit axes; favor
       growth-duration assets)
  ELIF PFF 21d return > 0          -> SPY 97%   (mid: positive but unconfirmed)
  ELSE (PFF 21d <= 0 OR PFF < 50d SMA) -> IEF 50% + SHY 47%
       (financial conditions tightening — cash-like duration ladder)

Why anti-consensus
------------------
- Zero PFF strategies in any prior round (intents.csv search confirms).
- Orthogonal to JNK pure-credit gates (JNK = HY corporate, mostly non-financial,
  short-to-mid duration). PFF embeds bank balance-sheet risk + long duration.
- Orthogonal to TNX-IRX yield-curve slope (Treasury slope is rate-only, no
  credit signal). PFF spread reaction layers credit on top of duration.
- Avoids all the prohibited themes: no SP500 xsect, no VIX gating, no JNK MA,
  no RSP breadth, no yield slope, no Halloween, no sector momentum, no
  commodity, no gold-vs-equity.

What could go wrong
-------------------
- 2010-2018 was a secular bull for both rates (falling) and credit (tightening).
  PFF essentially trended up the whole window, so the "weak PFF" defensive
  branch may rarely fire — leading to high corr to SPY/QQQ trend strategies.
- PFF's distribution yield masks short-term price moves; 21d return may be
  noisy for small distribution dates. We use price returns (not total return)
  consistent with cached close-only data.
- If the signal is too rarely "strong" (PFF 21d > +1%), the strategy collapses
  to the SPY 97% case and becomes a SPY clone, hitting the 0.85 corr filter.
"""
from __future__ import annotations

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["SPY", "QQQ", "IEF", "SHY", "PFF"]

REBALANCE_EVERY = 5         # weekly
PFF_RET_WINDOW = 21         # 21d (1 month) return on PFF
PFF_SMA = 50                # PFF 50d trend filter
PFF_STRONG_THRESH = 0.010   # +1.0% over 21d to confirm "loose conditions"
TREND_WINDOW = 200          # SPY 200d outer gate
EXPOSURE = 0.97
DEF_IEF = 0.50
DEF_SHY = 0.47


class PFFCreditRateBarometer(Strategy):
    """PFF 21d return + 50d SMA = composite easing/tightening signal."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        pff_ret_window: int = PFF_RET_WINDOW,
        pff_sma: int = PFF_SMA,
        pff_strong_thresh: float = PFF_STRONG_THRESH,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
        def_ief: float = DEF_IEF,
        def_shy: float = DEF_SHY,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            pff_ret_window=pff_ret_window,
            pff_sma=pff_sma,
            pff_strong_thresh=pff_strong_thresh,
            trend_window=trend_window,
            exposure=exposure,
            def_ief=def_ief,
            def_shy=def_shy,
        )
        self.rebalance_every = int(rebalance_every)
        self.pff_ret_window = int(pff_ret_window)
        self.pff_sma = int(pff_sma)
        self.pff_strong_thresh = float(pff_strong_thresh)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)
        self.def_ief = float(def_ief)
        self.def_shy = float(def_shy)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.pff_sma, self.trend_window) + 5
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

        # SPY 200d trend gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_now = float(spy_close.iloc[-1])
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        bull = spy_now > spy_sma

        target: dict[str, float] = {}

        if not bull:
            # SPY bear: defensive ladder regardless of PFF
            if "IEF" in live:
                target["IEF"] = self.def_ief * self.exposure
            if "SHY" in live:
                target["SHY"] = self.def_shy * self.exposure
        else:
            # Compute PFF signals
            pff_strong = False
            pff_positive = False
            try:
                pff_hist = ctx.history("PFF")
                pff_close = pff_hist["close"].dropna()
                if len(pff_close) >= max(self.pff_sma, self.pff_ret_window) + 1:
                    pff_now = float(pff_close.iloc[-1])
                    pff_then = float(pff_close.iloc[-1 - self.pff_ret_window])
                    if pff_then > 0:
                        pff_ret = pff_now / pff_then - 1.0
                    else:
                        pff_ret = 0.0
                    pff_sma_val = float(pff_close.iloc[-self.pff_sma:].mean())
                    above_sma = pff_now > pff_sma_val

                    if pff_ret > self.pff_strong_thresh and above_sma:
                        pff_strong = True
                    elif pff_ret > 0 and above_sma:
                        pff_positive = True
                    # else: weak (either negative ret or below SMA)
            except KeyError:
                # If PFF missing, fall through to defensive
                pass

            if pff_strong:
                # Loose financial conditions: growth-duration tilt
                if "QQQ" in live:
                    target["QQQ"] = self.exposure
                else:
                    # Fallback to SPY if QQQ unavailable
                    if "SPY" in live:
                        target["SPY"] = self.exposure
            elif pff_positive:
                # Mid: SPY broad
                if "SPY" in live:
                    target["SPY"] = self.exposure
            else:
                # Tightening / unclear: defensive bond ladder
                if "IEF" in live:
                    target["IEF"] = self.def_ief * self.exposure
                if "SHY" in live:
                    target["SHY"] = self.def_shy * self.exposure

        if not target:
            return []

        orders: list[Order] = []

        # Liquidate positions not in target
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


NAME = "opus5_pff_credit_rate_barometer"
HYPOTHESIS = (
    "PFF preferred-stock composite credit+duration barometer: PFF 21d return + "
    "50d SMA splits regime into easing/mid/tightening. SPY 200d outer gate. "
    "Easing -> QQQ 97%; mid -> SPY 97%; tightening -> IEF 50% + SHY 47%. "
    "Signal orthogonal to JNK (HY-only credit) and TNX-IRX (rate-only slope) "
    "because PFF embeds bank-financial credit AND long duration in one price."
)

STRATEGY = PFFCreditRateBarometer()
