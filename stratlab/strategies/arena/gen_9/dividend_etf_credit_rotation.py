"""Dividend/Income ETF Momentum Rotation with JNK Credit Gate.

Hypothesis (sonnet-5, gen_9):
    Equal-weight momentum rotation across dividend/income ETFs
    (DVY, VIG, SCHD, SDY, VYM) with JNK credit regime gate:
    - JNK above 42d SMA (risk-on credit) -> top-2 dividend ETFs by 63d momentum
    - JNK below 42d SMA (risk-off credit) -> TLT 60% + IEF 37%
    Income-focused ETF universe captures dividend yield factor uncorrelated
    to pure growth momentum. Biweekly rebalance.

Diversification angle vs leaderboard:
  - No prior round strategy uses dividend/income ETFs (DVY, VIG, SCHD, SDY, VYM)
    as the primary investment universe.
  - gen8_opus1_dividend_rate_credit_rotation used DVY+VIG+SCHD but only in one
    branch of a more complex strategy; here they ARE the universe.
  - Factor captured: dividend growth / yield factor. Distinct from growth (QQQ),
    credit (JNK/HYG), pure price-momentum (SP500 stocks).
  - Credit gate via JNK 42d SMA (not 30d or 60d) for moderate responsiveness.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10        # biweekly
CREDIT_SMA = 42             # JNK SMA lookback for credit regime
MOM_WINDOW = 63             # dividend ETF momentum window
TOP_K = 2                   # hold top-2 dividend ETFs
EXPOSURE = 0.97

# Dividend/income ETF universe
DIVIDEND_ETFS = ["DVY", "VIG", "SCHD", "SDY", "VYM"]
# Defensive allocation
DEFENSIVE_BOND = "TLT"
DEFENSIVE_MID = "IEF"
DEFENSIVE_BOND_WEIGHT = 0.60
DEFENSIVE_MID_WEIGHT = 0.37


class DividendEtfCreditRotation(Strategy):
    """Dividend ETF 63d momentum rotation gated by JNK credit trend."""

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(CREDIT_SMA, MOM_WINDOW) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % REBALANCE_EVERY != 0:
            return []

        # --- JNK credit regime gate ---
        try:
            jnk_hist = ctx.history("JNK")
        except KeyError:
            return []
        if len(jnk_hist) < CREDIT_SMA + 2:
            return []

        jnk_close = jnk_hist["close"].dropna()
        if len(jnk_close) < CREDIT_SMA:
            return []
        jnk_sma = float(jnk_close.iloc[-CREDIT_SMA:].mean())
        jnk_price = float(jnk_close.iloc[-1])
        credit_risk_on = jnk_price > jnk_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if p > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if credit_risk_on:
            # Compute 63d momentum for dividend ETFs
            prices = ctx.closes_window(MOM_WINDOW + 5)
            if len(prices) < MOM_WINDOW:
                # Fallback: equal-weight available dividend ETFs
                avail = [s for s in DIVIDEND_ETFS if s in closes_now.index and closes_now[s] > 0]
                if avail:
                    wt = EXPOSURE / len(avail[:TOP_K])
                    for sym in avail[:TOP_K]:
                        target[sym] = wt
            else:
                scores: dict[str, float] = {}
                for sym in DIVIDEND_ETFS:
                    if sym not in prices.columns:
                        continue
                    col = prices[sym].dropna()
                    if len(col) < MOM_WINDOW:
                        continue
                    p_end = float(col.iloc[-1])
                    p_start = float(col.iloc[-MOM_WINDOW])
                    if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                        continue
                    scores[sym] = p_end / p_start - 1.0

                if not scores:
                    # All dividend ETFs missing -> defensive
                    credit_risk_on = False
                else:
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)
                    selected = ranked[:TOP_K]
                    per_slot = EXPOSURE / len(selected)
                    for sym in selected:
                        target[sym] = per_slot

        if not credit_risk_on:
            # Risk-off: TLT + IEF
            if DEFENSIVE_BOND in closes_now.index:
                target[DEFENSIVE_BOND] = DEFENSIVE_BOND_WEIGHT
            if DEFENSIVE_MID in closes_now.index:
                target[DEFENSIVE_MID] = DEFENSIVE_MID_WEIGHT

        # --- Generate orders ---
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


UNIVERSE = DIVIDEND_ETFS + ["JNK", "TLT", "IEF"]

NAME = "gen9_dividend_etf_credit_rotation"
HYPOTHESIS = (
    "Momentum rotation across dividend ETFs (DVY,VIG,SCHD,SDY,VYM) with JNK credit gate: "
    "JNK above 42d SMA -> top-2 dividend ETFs by 63d momentum; "
    "JNK below 42d SMA -> TLT 60%+IEF 37%; biweekly rebalance."
)

STRATEGY = DividendEtfCreditRotation()
