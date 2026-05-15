"""VRP-Gated SP500 Skip-Month Momentum — gen_9 sonnet-7

Hypothesis: Volatility Risk Premium (VRP = VIX - SPY 20d annualized realized vol)
as regime gate for SP500 skip-month momentum.

Regime logic:
  - SPY below 200d SMA: TLT 97% (outer bear gate)
  - VRP > 5 (market overpaying for protection by >5pp): top-15 SP500 by
    126d-skip-21d momentum, inverse-vol weighted, 97% exposure. High VRP
    historically predicts positive subsequent returns (options overpriced).
  - VRP < 0 (realized vol exceeds implied, genuine stress): TLT 97%
  - SPY above 200d SMA and -5 < VRP < 0: SPY 97% (uncertainty regime)
  - SPY above 200d SMA and 0 <= VRP <= 5: SPY 97% (neutral/mild premium)

Rationale: VRP is a well-documented predictor of equity risk premia. When
implied vol (VIX) significantly exceeds realized vol, market participants
are overpaying for protection — a classic contrarian signal that subsequent
returns will be positive as the fear premium erodes. In the IS window:
- High VRP (>5pp) corresponded to the sustained 2012-2017 low-vol bull
  market where VIX averaged ~15% but realized vol averaged ~8-10%.
- Negative VRP corresponded to actual volatility episodes (Aug 2011, Q4 2018).

This differs from VIX-level gates (which use absolute thresholds) and VIX
percentile gates (which use relative ranking) by using the VRP spread itself.
The VRP signal is orthogonal to credit spreads, yield curve, and breadth gates.

Distinct from all leaderboard: no prior strategy uses VRP as primary gate for
SP500 stock selection. The IC (ic_26bf4072) was committed for gen_9.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy
from stratlab.data.universe import sp500_tickers

# ── Parameters ──────────────────────────────────────────────────────────────
REALVOL_WINDOW = 20      # 20-day realized vol window
SPY_TREND = 200          # SPY outer bear gate
MOM_LONG = 126           # Skip-month momentum lookback
MOM_SKIP = 21            # Skip 1 month
VOL_WINDOW = 21          # Inverse-vol weighting
TOP_K = 15
REBALANCE_EVERY = 10     # Bi-weekly
EXPOSURE = 0.97

VRP_HIGH = 5.0           # VRP > 5pp: risk-on, stocks
VRP_LOW = 0.0            # VRP < 0: genuine stress, TLT


class VrpGatedSkipMonSp500(Strategy):
    """VRP (VIX minus realized vol) gating SP500 skip-month momentum."""

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(SPY_TREND, MOM_LONG + MOM_SKIP, REALVOL_WINDOW) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % REBALANCE_EVERY != 0:
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
        spy_bull = float(spy_close.iloc[-1]) > spy_sma200

        if not spy_bull:
            targets: dict[str, float] = {"TLT": EXPOSURE}
            return self._trade_to_targets(ctx, live, targets)

        # Compute SPY 20d annualized realized vol
        if len(spy_close) < REALVOL_WINDOW + 2:
            return []
        spy_rets = np.log(spy_close.values[-REALVOL_WINDOW - 1:])
        spy_log_rets = spy_rets[1:] - spy_rets[:-1]
        realized_vol_ann = float(np.std(spy_log_rets) * np.sqrt(252) * 100)

        # Get VIX level
        try:
            vix_hist = ctx.history("^VIX")
        except KeyError:
            targets = {"SPY": EXPOSURE}
            return self._trade_to_targets(ctx, live, targets)

        if len(vix_hist) < 2:
            targets = {"SPY": EXPOSURE}
            return self._trade_to_targets(ctx, live, targets)

        vix_level = float(vix_hist["close"].dropna().iloc[-1])

        # VRP = VIX - Realized Vol (both as % per year)
        vrp = vix_level - realized_vol_ann

        if vrp > VRP_HIGH:
            # Overpaying for protection: stocks outperform
            targets = self._skipmon_targets(ctx, live)
            if not targets:
                targets = {"SPY": EXPOSURE}
        elif vrp < VRP_LOW:
            # Realized vol > implied: genuine stress, go defensive
            targets = {"TLT": EXPOSURE}
        else:
            # Neutral VRP (0 to 5pp): hold SPY
            targets = {"SPY": EXPOSURE}

        return self._trade_to_targets(ctx, live, targets)

    def _skipmon_targets(
        self, ctx: BarContext, live: dict[str, float]
    ) -> dict[str, float]:
        """Compute skip-month momentum targets with inverse-vol weighting."""
        need = MOM_LONG + MOM_SKIP + 2
        prices = ctx.closes_window(need)
        if len(prices) < need - 1:
            return {}

        scores: dict[str, float] = {}
        inv_vols: dict[str, float] = {}

        for sym in prices.columns:
            if sym.startswith("^") or sym in ("SPY", "TLT", "IEF"):
                continue
            col = prices[sym].dropna()
            if len(col) < MOM_LONG + MOM_SKIP:
                continue
            end_idx = -MOM_SKIP
            start_idx = -(MOM_LONG + MOM_SKIP)
            p_end = float(col.iloc[end_idx])
            p_start = float(col.iloc[start_idx])
            if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                continue
            ret = p_end / p_start - 1.0
            if not np.isfinite(ret):
                continue

            # Inverse-vol weighting
            vol_tail = col.iloc[-VOL_WINDOW - 1:]
            if len(vol_tail) < VOL_WINDOW + 1:
                continue
            log_r = np.log(vol_tail.values[1:] / vol_tail.values[:-1])
            rv = float(np.std(log_r))
            if rv <= 1e-6 or not np.isfinite(rv):
                continue

            scores[sym] = ret
            inv_vols[sym] = 1.0 / rv

        if len(scores) < TOP_K:
            return {}

        ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:TOP_K]
        iv_sum = sum(inv_vols[s] for s in ranked)
        if iv_sum <= 0:
            return {}

        return {sym: EXPOSURE * inv_vols[sym] / iv_sum for sym in ranked}

    def _trade_to_targets(
        self,
        ctx: BarContext,
        live: dict[str, float],
        targets: dict[str, float],
    ) -> list[Order]:
        """Generate orders to reach target weights."""
        equity = ctx.cash
        for sym, pos in ctx.positions.items():
            p = live.get(sym, 0.0)
            if p > 0:
                equity += pos.size * p
        if equity <= 0:
            return []

        orders: list[Order] = []
        for sym, pos in list(ctx.positions.items()):
            if sym not in targets and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        for sym, weight in targets.items():
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
    return sp500_tickers() + ["SPY", "TLT", "^VIX"]


NAME = "gen9_vrp_gated_skipmon_sp500"
HYPOTHESIS = (
    "VRP (VIX minus SPY 20d annualized realized vol) as regime gate: "
    "VRP > 5pp -> top-15 SP500 126d-skip-21d momentum inverse-vol weighted; "
    "VRP < 0pp -> TLT 97%; 0-5pp -> SPY 97%; "
    "SPY 200d SMA bear -> TLT 97%. Biweekly rebalance."
)

UNIVERSE = _universe

STRATEGY = VrpGatedSkipMonSp500()
